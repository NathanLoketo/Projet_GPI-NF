# Projet_GPI-NF
Nathan Fracchiolla
## Projet M1 GPI

This project was carried out as part of the M1 course unit *Gestion de projet informatique*.

The aim of the project is to develop a Python script for particle detection in microscopy images using normalized cross-correlation.

# Particle Detection Script

This script detects particles in an MRC microscopy image using normalized cross-correlation (NCC).

The script:
- downloads and loads an MRC image;
- bins the image to make the computation faster;
- normalizes and denoises the image;
- creates a 2D reference projection from a PDB structure;
- compares the reference with the image using NCC;
- detects particles from the NCC result;
- displays the analyzed image with squares around detected particles;
- displays a grid containing the detected particles individually.

- optionally uses a cropped region for faster parameter testing;

## Required packages

Install the required Python packages before running the script.

```bash
pip install numpy matplotlib scipy scikit-image mrcfile biopython gdown
```

## Run the script

Open a terminal in the folder containing the script, then run:

```bash
python Projet_GPI_NF.py
```

## Main parameters

The main parameters can be modified at the beginning of the script.

```python
USE_CROP = True
```

Set `USE_CROP = True` to test the detection faster on a cropped image.  
Set `USE_CROP = False` to run the detection on the full image.

```python
PARTICLE_THRESHOLD = 0.35
```

This controls the detection threshold.  
A higher value detects fewer particles but usually gives cleaner detections.  
A lower value detects more particles but may include more false positives.

```python
BIN_SIZE = 4
```

This controls the binning of the image.  
A higher value makes the image smaller and the computation faster, but it reduces image detail.  
A lower value keeps more detail, but the script will be slower.

```python
PAD_WIDTH = 25
```

This controls the padding added around the image before particle detection. 

## Save individual particle images

The script can optionally save each detected particle as a separate `.png` image.

To activate this option, change this parameter at the beginning of the script:

```python
SAVE_INDIVIDUAL_PARTICLES = True
```

If you do not want to save individual particle images, keep:

```python
SAVE_INDIVIDUAL_PARTICLES = False
```

The images are saved in a folder defined by:

```python
PARTICLE_OUTPUT_FOLDER = "detected_particles"
```

This folder will be created in the directory from which you run the script.
