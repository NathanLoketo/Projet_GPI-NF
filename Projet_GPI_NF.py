import os
import urllib.request

import gdown
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mrcfile
import numpy as np
from Bio.PDB import PDBParser
from skimage.feature import peak_local_max
from skimage.filters import gaussian


# =========================
# Parameters to adjust
# =========================

FILE_ID = "1Qj30jSXcHEpkzE04cisbP6ljtnQ2Ausr"
MRC_FILE = "map.mrc"

PDB_ID = "6BDF"
PDB_FILE = f"{PDB_ID}.pdb"

USE_CROP = False   # True = faster test on cropped image, False = full image

CROP_X_START = 1
CROP_X_END = 500
CROP_Y_START = 1
CROP_Y_END = 500

BIN_SIZE = 4
PAD_WIDTH = 25
PIXEL_SIZE = 2.64  # Angstrom / pixel

IMAGE_DENOISE_SIGMA = 1
TEMPLATE_BLUR_SIGMA = 1

PARTICLE_THRESHOLD = 0.36
MIN_DISTANCE = 10
WINDOW_SIZE = 60
MAX_PARTICLES_TO_SHOW = 40

SAVE_INDIVIDUAL_PARTICLES = False
PARTICLE_OUTPUT_FOLDER = "detected_particles"

# =========================
# Image preparation
# =========================

def download_mrc_file(file_id, output_file):
    if not os.path.exists(output_file):
        gdown.download(id=file_id, output=output_file, quiet=False)


def load_mrc_image(mrc_file):
    with mrcfile.open(mrc_file, permissive=True) as mrc:
        data = mrc.data.copy()

    if data.ndim == 3:
        image = np.sum(data, axis=0)
    elif data.ndim == 2:
        image = data
    else:
        raise ValueError(f"Unsupported MRC data shape: {data.shape}")

    return image


def bin_image(image, bin_size):
    h, w = image.shape
    new_h = h // bin_size
    new_w = w // bin_size

    binned = image[:new_h * bin_size, :new_w * bin_size]
    binned = binned.reshape(new_h, bin_size, new_w, bin_size)
    binned = binned.mean(axis=(1, 3))

    return binned


def normalize_image(image):
    image = image.astype(np.float32)

    if image.max() == image.min():
        return np.zeros_like(image)

    return (image - image.min()) / (image.max() - image.min())


def apply_padding(image, pad_width, constant_value=0):
    return np.pad(
        image,
        ((pad_width, pad_width), (pad_width, pad_width)),
        mode="constant",
        constant_values=constant_value
    )

def crop_image(image, x_start, x_end, y_start, y_end):
    return image[y_start:y_end, x_start:x_end]


# =========================
# Template preparation
# =========================

def download_pdb_file(pdb_id, pdb_file):
    if not os.path.exists(pdb_file):
        pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"

        with urllib.request.urlopen(pdb_url, timeout=10) as response:
            pdb_content = response.read().decode("utf-8")

        with open(pdb_file, "w") as f:
            f.write(pdb_content)


def extract_pdb_coordinates(pdb_id, pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_file)
    model = structure[0]

    all_coords = []

    for chain in model.get_chains():
        for residue in chain.get_residues():
            for atom in residue.get_atoms():
                all_coords.append(atom.get_coord())

    return np.array(all_coords)


def create_projection(coord1, coord2, padding=10):
    min1 = np.floor(coord1.min()) - padding
    max1 = np.ceil(coord1.max()) + padding

    min2 = np.floor(coord2.min()) - padding
    max2 = np.ceil(coord2.max()) + padding

    bins1 = int(max1 - min1)
    bins2 = int(max2 - min2)

    projection, _, _ = np.histogram2d(
        coord1,
        coord2,
        bins=[bins1, bins2],
        range=[[min1, max1], [min2, max2]]
    )

    projection = projection / projection.max()
    projection = 1 - projection

    return projection


def create_templates_from_pdb(pdb_id, pdb_file, pixel_size):
    download_pdb_file(pdb_id, pdb_file)

    coords = extract_pdb_coordinates(pdb_id, pdb_file)
    coords_centered = coords - coords.mean(axis=0)
    coords_scaled = coords_centered / pixel_size

    proj_xy = create_projection(coords_scaled[:, 0], coords_scaled[:, 1])
    proj_xz = create_projection(coords_scaled[:, 0], coords_scaled[:, 2])

    proj_xy_blurred = gaussian(proj_xy, sigma=TEMPLATE_BLUR_SIGMA, preserve_range=True)
    proj_xz_blurred = gaussian(proj_xz, sigma=TEMPLATE_BLUR_SIGMA, preserve_range=True)

    return proj_xy_blurred, proj_xz_blurred


# =========================
# NCC detection
# =========================

def normalized_cross_correlation(image, template):
    img_h, img_w = image.shape
    tmp_h, tmp_w = template.shape

    out_h = img_h - tmp_h + 1
    out_w = img_w - tmp_w + 1

    output = np.zeros((out_h, out_w))

    template_norm = template - template.mean()
    template_std = np.sqrt(np.sum(template_norm ** 2))

    if template_std == 0:
        return output

    for i in range(out_h):
        for j in range(out_w):
            region = image[i:i + tmp_h, j:j + tmp_w]

            region_norm = region - region.mean()
            region_std = np.sqrt(np.sum(region_norm ** 2))

            if region_std == 0:
                output[i, j] = 0
            else:
                output[i, j] = np.sum(region_norm * template_norm) / (region_std * template_std)

    return output


def filter_border_coords(coords, image_shape, window_size):
    coords = np.squeeze(coords)

    if coords.size == 0:
        return np.empty((0, 2), dtype=int)

    if coords.ndim == 1:
        coords = coords.reshape(1, -1)

    half = window_size // 2
    height, width = image_shape

    filtered = []

    for y, x in coords:
        if (
            y - half >= 0 and
            y + half < height and
            x - half >= 0 and
            x + half < width
        ):
            filtered.append([y, x])

    return np.array(filtered, dtype=int)


def detect_particles(ncc_result, template, image_shape, threshold, min_distance, window_size):
    
    # Detect only local maxima in the NCC map.
    # min_distance prevents detecting the same particle several times
    # by forcing detected peaks to be separated by at least MIN_DISTANCE pixels.

    coords = peak_local_max(
        ncc_result,
        threshold_abs=threshold,
        min_distance=min_distance
    )

    # Convert NCC top-left coordinates to particle center coordinates
    coords = coords + np.array([
        template.shape[0] // 2,
        template.shape[1] // 2
    ])

    coords = filter_border_coords(
        coords,
        image_shape=image_shape,
        window_size=window_size
    )

    return coords


# =========================
# Final plots
# =========================

def show_detected_particles(base_img, coords, window_size):
    half = window_size // 2

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(base_img, cmap="gray")
    ax.axis("off")

    for y, x in coords:
        rect = patches.Rectangle(
            (x - half, y - half),
            window_size,
            window_size,
            linewidth=1.5,
            edgecolor="red",
            facecolor="none"
        )
        ax.add_patch(rect)

    ax.set_title(f"Detected particles: {len(coords)}")
    plt.tight_layout()
    plt.show()


def show_particle_grid(base_img, coords, window_size=60, particles_per_grid=40):
    half = window_size // 2

    if len(coords) == 0:
        print("No particles to show.")
        return

    for grid_index, start in enumerate(range(0, len(coords), particles_per_grid), start=1):
        coords_subset = coords[start:start + particles_per_grid]

        particles = []

        for y, x in coords_subset:
            y = int(y)
            x = int(x)

            particle = base_img[
                y - half:y + half,
                x - half:x + half
            ]

            particles.append(particle)

        n_particles = len(particles)
        n_cols = 8
        n_rows = int(np.ceil(n_particles / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 1.5 * n_rows))

        axes = np.array(axes).reshape(-1)

        for i, ax in enumerate(axes):
            if i < n_particles:
                ax.imshow(particles[i], cmap="gray", interpolation="bicubic")
                ax.set_title(f"Particle {start + i + 1}")
                ax.axis("off")
            else:
                ax.axis("off")

        plt.suptitle(f"Grid {grid_index} - particles {start + 1} to {start + n_particles}")
        plt.tight_layout()
        plt.show()

def save_individual_particles(base_img, coords, window_size=60, output_folder="detected_particles"):
    half = window_size // 2

    os.makedirs(output_folder, exist_ok=True)

    for i, (y, x) in enumerate(coords, start=1):
        y = int(y)
        x = int(x)

        particle = base_img[
            y - half:y + half,
            x - half:x + half
        ]

        output_path = os.path.join(output_folder, f"particle_{i:03d}.png")

        plt.imsave(
            output_path,
            particle,
            cmap="gray"
        )

    print(f"Saved {len(coords)} particles in folder: {output_folder}")

# =========================
# Main pipeline
# =========================

def main():
    download_mrc_file(FILE_ID, MRC_FILE)

    raw_image = load_mrc_image(MRC_FILE)

    binned_image = bin_image(raw_image, BIN_SIZE)

    if USE_CROP:
        image_to_analyze = crop_image(
            binned_image,
            CROP_X_START,
            CROP_X_END,
            CROP_Y_START,
            CROP_Y_END
        )
    else:
        image_to_analyze = binned_image

    normalized_image = normalize_image(image_to_analyze)

    padded_img = apply_padding(
        normalized_image,
        PAD_WIDTH
    )

    padded_img_denoised = gaussian(
        padded_img,
        sigma=IMAGE_DENOISE_SIGMA,
        preserve_range=True
    )

    proj_xy_blurred, proj_xz_blurred = create_templates_from_pdb(
        PDB_ID,
        PDB_FILE,
        PIXEL_SIZE
    )

    result1 = normalized_cross_correlation(
        padded_img_denoised,
        proj_xy_blurred
    )

    coords = detect_particles(
        ncc_result=result1,
        template=proj_xy_blurred,
        image_shape=padded_img_denoised.shape,
        threshold=PARTICLE_THRESHOLD,
        min_distance=MIN_DISTANCE,
        window_size=WINDOW_SIZE
    )

    print(f"Detected particles: {len(coords)}")

    show_detected_particles(
        base_img=padded_img_denoised,
        coords=coords,
        window_size=WINDOW_SIZE
    )

    show_particle_grid(
        base_img=padded_img_denoised,
        coords=coords,
        window_size=WINDOW_SIZE,
        particles_per_grid=40
    )

    if SAVE_INDIVIDUAL_PARTICLES:
        save_individual_particles(
            base_img=padded_img_denoised,
            coords=coords,
            window_size=WINDOW_SIZE,
            output_folder=PARTICLE_OUTPUT_FOLDER
        )


if __name__ == "__main__":
    main()