import os
import json
import glob
import random
import collections
import tarfile

import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px
import nibabel as nib
import albumentations as A
from matplotlib import animation, rc
from skimage.morphology import binary_closing

# Load the CSV file
train_df = pd.read_csv("train_labels.csv")

# Exploratory Data Analysis
plt.figure(figsize=(5, 5))
sns.countplot(data=train_df, x="MGMT_value")


# Load DICOM images
def load_dicom(path):
    dicom = pydicom.dcmread(path)
    data = dicom.pixel_array
    data = (data - np.min(data)) / (np.max(data) or 1)  # Normalization to [0,1]
    data = (data * 255).astype(np.uint8)  # Convert to 8-bit image
    return data


# Visualize sample images
def visualize_sample(brats21id, slice_i, mgmt_value, types=("FLAIR", "T1w", "T1wCE", "T2w")):
    plt.figure(figsize=(16, 5))
    patient_path = os.path.join("train/", str(brats21id).zfill(5))
    for i, t in enumerate(types, 1):
        t_paths = sorted(
            glob.glob(os.path.join(patient_path, t, "*")),
            key=lambda x: int(x[:-4].split("-")[-1]),
        )
        data = load_dicom(t_paths[int(len(t_paths) * slice_i)])
        plt.subplot(1, 4, i)
        plt.imshow(data, cmap="gray")
        plt.title(f"{t}", fontsize=16)
        plt.axis("off")

    plt.suptitle(f"MGMT_value: {mgmt_value}", fontsize=16)
    plt.show()


for i in random.sample(range(train_df.shape[0]), 5):
    _brats21id = train_df.iloc[i]["BraTS21ID"]
    _mgmt_value = train_df.iloc[i]["MGMT_value"]
    visualize_sample(brats21id=_brats21id, mgmt_value=_mgmt_value, slice_i=0.5)

# Animation setup
rc('animation', html='jshtml')


def create_animation(ims):
    fig = plt.figure(figsize=(6, 6))
    plt.axis('off')
    im = plt.imshow(ims[0], cmap="gray")

    def animate_func(i):
        im.set_array(ims[i])
        return [im]

    anim = animation.FuncAnimation(fig, animate_func, frames=len(ims), interval=1000 // 24)

    # Show the animation
    plt.show()

    return anim


def load_dicom_line(path):
    t_paths = sorted(
        glob.glob(os.path.join(path, "*")),
        key=lambda x: int(x[:-4].split("-")[-1]),
    )
    images = []
    for filename in t_paths:
        data = load_dicom(filename)
        if data.max() == 0:
            continue
        images.append(data)

    return images


# Load and display the animations
images = load_dicom_line("train/00000/FLAIR")
anim = create_animation(images)

images = load_dicom_line("train/00000/T2w")
anim = create_animation(images)


# 3D MRI Scan and file extraction function
import os
import tarfile


def extract_task1_files(root="./data"):
    try:
        tar_path = "BraTS2021_Training_Data.tar"  # Updated with the correct filename

        # Print the absolute path for debugging
        full_path = os.path.abspath(tar_path)
        print(f"Attempting to access: {full_path}")

        if not os.path.exists(tar_path):
            raise FileNotFoundError(f"File not found: {full_path}")

        if not os.access(tar_path, os.R_OK):
            raise PermissionError(f"Permission denied for file: {full_path}")

        # Open the .tar file and extract with progress
        with tarfile.open(tar_path, "r") as tar:
            members = tar.getmembers()
            total_files = len(members)
            print(f"Total files to extract: {total_files}")

            for i, member in enumerate(members):
                tar.extract(member, path=root)
                # Calculate progress
                progress = (i + 1) / total_files * 100
                print(f"Extracted: {i + 1}/{total_files} ({progress:.2f}% complete) - {member.name}")

            print(f"Successfully extracted files to {root}")

    except PermissionError as e:
        print(f"Permission Error: {e}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except tarfile.ReadError as e:
        print(f"Read Error: {e} - not a valid tar file or the format is incorrect")
    except Exception as e:
        print(f"An error occurred: {e}")

extract_task1_files()


# ImageReader Class
class ImageReader:
    def __init__(self, root: str, img_size: int = 256, normalize: bool = False, single_class: bool = False):
        pad_size = 256 if img_size > 256 else 224
        self.resize = A.Compose(
            [
                A.PadIfNeeded(min_height=pad_size, min_width=pad_size, value=0),
                A.Resize(img_size, img_size)
            ]
        )
        self.normalize = normalize
        self.single_class = single_class
        self.root = root

    def read_file(self, path: str) -> dict:
        scan_type = path.split('_')[-1]
        raw_image = nib.load(path).get_fdata()
        raw_mask = nib.load(path.replace(scan_type, 'seg.nii.gz')).get_fdata()
        processed_frames, processed_masks = [], []
        for frame_idx in range(raw_image.shape[2]):
            frame = raw_image[:, :, frame_idx]
            mask = raw_mask[:, :, frame_idx]
            if self.normalize:
                if frame.max() > 0:
                    frame = frame / frame.max()
                frame = frame.astype(np.float32)
            else:
                frame = frame.astype(np.uint8)
            resized = self.resize(image=frame, mask=mask)
            processed_frames.append(resized['image'])
            processed_masks.append(1 * (resized['mask'] > 0) if self.single_class else resized['mask'])
        return {
            'scan': np.stack(processed_frames, 0),
            'segmentation': np.stack(processed_masks, 0),
            'orig_shape': raw_image.shape
        }

    def load_patient_scan(self, idx: int, scan_type: str = 'flair') -> dict:
        patient_id = str(idx).zfill(5)
        scan_filename = f'{self.root}/BraTS2021_{patient_id}/BraTS2021_{patient_id}_{scan_type}.nii.gz'
        return self.read_file(scan_filename)


# 3D Scatter Plot Functions
def generate_3d_scatter(
        x: np.array, y: np.array, z: np.array, colors: np.array,
        size: int = 3, opacity: float = 0.2, scale: str = 'Teal',
        hover: str = 'skip', name: str = 'MRI'
) -> go.Scatter3d:
    return go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers', hoverinfo=hover,
        marker=dict(
            size=size, opacity=opacity,
            color=colors, colorscale=scale
        ),
        name=name
    )


class ImageViewer3d:
    def __init__(self, reader: ImageReader, mri_downsample: int = 10, mri_colorscale: str = 'Ice') -> None:
        self.reader = reader
        self.mri_downsample = mri_downsample
        self.mri_colorscale = mri_colorscale

    def load_clean_mri(self, image: np.array, orig_dim: int) -> dict:
        shape_offset = image.shape[1] / orig_dim
        z, x, y = (image > 0).nonzero()
        x, y, z = x[::self.mri_downsample], y[::self.mri_downsample], z[::self.mri_downsample]
        colors = image[z, x, y]
        return dict(x=x / shape_offset, y=y / shape_offset, z=z, colors=colors)

    def load_tumor_segmentation(self, image: np.array, orig_dim: int) -> dict:
        tumors = {}
        shape_offset = image.shape[1] / orig_dim
        sampling = {1: 1, 2: 3, 4: 5}
        for class_idx in sampling:
            z, x, y = (image == class_idx).nonzero()
            x, y, z = x[::sampling[class_idx]], y[::sampling[class_idx]], z[::sampling[class_idx]]
            tumors[class_idx] = dict(
                x=x / shape_offset, y=y / shape_offset, z=z,
                colors=class_idx / 4
            )
        return tumors

    def collect_patient_data(self, scan: dict) -> tuple:
        clean_mri = self.load_clean_mri(scan['scan'], scan['orig_shape'][0])
        tumors = self.load_tumor_segmentation(scan['segmentation'], scan['orig_shape'][0])
        markers_created = clean_mri['x'].shape[0] + sum(tumors[class_idx]['x'].shape[0] for class_idx in tumors)
        return [
            generate_3d_scatter(**clean_mri, scale=self.mri_colorscale, opacity=0.3, hover='skip', name='Brain MRI'),
            generate_3d_scatter(**tumors[1], opacity=0.8, hover='all', name='Necrotic tumor core'),
            generate_3d_scatter(**tumors[2], opacity=0.4, hover='all', name='Peritumoral invaded tissue'),
            generate_3d_scatter(**tumors[4], opacity=0.4, hover='all', name='GD-enhancing tumor'),
        ], markers_created

    def get_3d_scan(self, patient_idx: int, scan_type: str = 'flair') -> go.Figure:
        scan = self.reader.load_patient_scan(patient_idx, scan_type)
        data, num_markers = self.collect_patient_data(scan)
        fig = go.Figure(data=data)
        fig.update_layout(
            title=f"[Patient id:{patient_idx}] brain MRI scan ({num_markers} points)",
            legend_title="Pixel class (click to enable/disable)",
            font=dict(
                family="Courier New, monospace",
                size=14,
            ),
            margin=dict(
                l=0, r=0, b=0, t=30
            ),
            legend=dict(itemsizing='constant')
        )
        return fig


# Instantiate ImageReader and ImageViewer3d
reader = ImageReader('./data', img_size=128, normalize=True, single_class=False)
viewer = ImageViewer3d(reader, mri_downsample=25)

# Generate and display 3D scan plots
fig = viewer.get_3d_scan(0, 't1')
plot(fig)  # Replacing iplot with plot to generate an HTML file

fig = viewer.get_3d_scan(9, 'flair')
plot(fig)  # Replacing iplot with plot to generate an HTML file


# Feature Extraction Functions
def get_approx_pixel_count(scan: np.array, close: bool = False, mask: bool = False, mask_idx: int = -1) -> int:
    slice_areas = []
    for slice_idx in range(scan.shape[0]):
        if close:
            mri = 1 * binary_closing(scan[slice_idx, :, :])
        elif mask_idx >= 0:
            mri = 1 * (scan[slice_idx, :, :] == mask_idx)
        elif mask:
            mri = 1 * (scan[slice_idx, :, :] > 0)
        else:
            raise ValueError('Masking mechanism should be specified')
        mri_area = mri.sum()
        slice_areas.append(mri_area)
    return np.sum(slice_areas)


def get_centroid(scan: np.array, mask_idx: int = 1) -> list:
    z, x, y = (scan == mask_idx).nonzero()
    x, y, z = np.median(x), np.median(y), np.median(z)
    return [x / scan.shape[1], y / scan.shape[2], z / scan.shape[0]]


df = pd.read_csv('train_labels.csv')
targets = dict(zip(df.BraTS21ID, df.MGMT_value))

features = []
for patient_idx in targets:
    try:
        data = reader.load_patient_scan(patient_idx)
        scan_px = get_approx_pixel_count(data['scan'], mask=True)
        tumor_px = get_approx_pixel_count(data['segmentation'], mask=True)
        dimension = np.prod(data['scan'].shape)
        patient_features = [patient_idx, targets[patient_idx]]
        patient_features.extend([scan_px / dimension, tumor_px / dimension, tumor_px / scan_px])
        patient_features.extend(get_centroid(data['segmentation'], 4))
        features.append(patient_features)
    except FileNotFoundError:
        continue

df = pd.DataFrame(features, columns=['idx', 'target', 'scan_pct', 'tumor_pct', 'tumor_ratio', 'x', 'y', 'z']).set_index(
    'idx')
fig = px.histogram(df, x="tumor_pct", color="target", marginal="box", nbins=100, barmode='relative')
fig.show()
