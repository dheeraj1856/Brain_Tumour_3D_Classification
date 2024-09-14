# Brain Tumor 3D Classification

This repository contains code for analyzing and visualizing brain MRI data using various medical image processing techniques. The code explores 3D MRI scans, extracts features, and applies machine learning models to predict brain tumor classes based on the MGMT promoter methylation status from the BraTS2021 dataset.

## Project Overview

The project utilizes MRI scans in DICOM and NIfTI formats and integrates several powerful libraries for image processing and data analysis:

- **pydicom**: To read and handle DICOM files.
- **nibabel**: To handle NIfTI images.
- **albumentations**: For image augmentation and resizing.
- **plotly**: For interactive 3D visualizations of brain scans.
- **matplotlib** and **seaborn**: For data visualization and exploratory data analysis.
- **skimage**: For morphological operations in image processing.

## Features

1. **Exploratory Data Analysis**:
   - A quick visualization of the distribution of MGMT values.
   - Sample image visualizations for MRI scans from different sequences (FLAIR, T1w, T1wCE, T2w).

2. **MRI Data Loading and Normalization**:
   - Functions to load DICOM and NIfTI files, normalize them, and prepare them for analysis.

3. **Image Augmentation**:
   - Image resizing and padding using the `albumentations` library to standardize input sizes.

4. **3D Scatter Plot Visualizations**:
   - 3D visualization of brain scans using Plotly, showcasing tumor areas in different classes.

5. **Feature Extraction**:
   - Calculation of tumor pixel percentages, tumor centroids, and other key image features for further analysis and model training.

6. **Animation Creation**:
   - Dynamic visualizations of MRI slices using `matplotlib.animation`.

7. **File Extraction**:
   - Automatic extraction of MRI data files from compressed tar archives.

## File Structure

- `train_labels.csv`: The CSV file containing target labels (MGMT values) for training.
- `train/`: Directory containing MRI scan data in DICOM format.
- `BraTS2021_Training_Data.tar`: The tar file containing the entire training dataset.
- `ImageReader`: Class to load and process MRI scans from NIfTI files.
- `ImageViewer3d`: Class for visualizing 3D MRI data and tumor segmentations.

## Getting Started

### Prerequisites

Make sure you have the following libraries installed:

- `numpy`
- `pandas`
- `pydicom`
- `nibabel`
- `matplotlib`
- `seaborn`
- `plotly`
- `albumentations`
- `scikit-image`
- `opencv-python`

## Dataset
The project uses the BraTS2021 and RSNA-MICCAI Brain Tumor Radiogenomic Classification dataset for brain tumor segmentation. The MRI scans and associated metadata (e.g., MGMT values) should be downloaded and extracted into a train/ directory.

### Usage
Exploratory Data Analysis: Use the code in the script to visualize the distribution of labels and sample MRI scans.

MRI Visualization: Run the visualize_sample() function to visualize 2D MRI slices for a particular patient ID.

3D MRI Visualization: Use the ImageViewer3d class to generate 3D visualizations of MRI scans and tumor segmentations.

Feature Extraction: Extract features like pixel counts and tumor centroids from the MRI scans using the provided functions.

## Extract Data

You can extract the training data using the extract_task1_files() function

## Results
The project includes functions to generate plots and statistics on tumor characteristics, as well as 3D visualizations of brain scans, which can be used to train machine learning models for brain tumor classification.


