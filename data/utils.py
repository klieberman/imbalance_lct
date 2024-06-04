from pathlib import Path

from torch.utils.data import Dataset
import torch
from PIL import Image

import numpy as np
import pandas as pd
from tqdm import tqdm


IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".JPEG",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def print_class_distribution(dataset, val_dataset=None):
    '''
    Prints the number of samples with each target value and name in a dataset.

    Args:
    dataset (torch.utils.data.Dataset): dataset where to find target distribution
    label_map (dictionary): map from target value to target name
    val_dataset (torch.utils.data.Dataset): optional validation dataset. If given,
        function will print it's distribution as well and dataset will be labeled 'train'
        in the output table.
    '''
    unique, counts = np.unique(dataset.targets, return_counts=True)
    if val_dataset is None:
        columns = ['Target', 'Label', 'Samples']
    else:
        columns = ['Target', 'Label', 'Train Samples', 'Val Samples']

    rows = []
    for label, count in zip(unique, counts):
        class_name = dataset.label_map[label]
        row = [label, class_name, count]
        if val_dataset is not None:
            val_count = np.sum(val_dataset.targets == label)
            row.append(val_count)
        rows.append(row)
    df = pd.DataFrame(rows, columns=columns)
    print(df.to_string(index=False))
    return None


def print_dataset_size(dataset_name, dataset, loader, batch_size):
    print(f"{dataset_name} dataset has {len(dataset)} images" \
          f"({len(loader)} batches of {batch_size}).")


class SubsetWithTargets(Dataset):
    '''
    Custom subset class which includes targets.
    Necessary because pytorch's Subset class does not include the targets attribute.
    '''

    def __init__(self, dataset, indices):
        self.dataset = torch.utils.data.Subset(dataset, indices)
        self.targets = [dataset.targets[idx] for idx in indices]

        # Additional attributes needed for ImbalancedCIFAR
        self.data = np.array([dataset.data[idx] for idx in indices])
        self.transform = dataset.transform
        self.target_transform = dataset.target_transform

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        target = self.targets[idx]
        return (image, target)

    def __len__(self):
        return len(self.targets)


def split_evenly(dataset, val_proportion):
    '''
    Splits a dataset into a training and validation set, maintaining
    the imbalance ratio in both sets.

    Args:
    dataset (torch.utils.data.Dataset): dataset to split
    val_proportion (float): proportion of dataset to put in validation split.
    '''
    class_labels = np.unique(dataset.targets)
    targets = np.array(dataset.targets)
    all_train_indices, all_val_indices = [], []
    for class_label in class_labels:
        # Get indices with these classes
        class_indices = np.argwhere(targets == class_label).flatten()
        n_samples = len(class_indices)
        n_val = round(n_samples * val_proportion)
        val_indices = np.random.choice(class_indices, n_val, replace=False)
        train_indices = class_indices[np.isin(
            class_indices, val_indices, invert=True)]
        all_train_indices.extend(train_indices)
        all_val_indices.extend(val_indices)
    return SubsetWithTargets(dataset, all_train_indices), SubsetWithTargets(dataset, all_val_indices)


def compute_mean_std(dataset_path):
    image_paths = []
    for ext in IMG_EXTENSIONS:
        image_paths.extend(Path(dataset_path).rglob(f"*{ext}"))
    
    mean = np.zeros(3)
    std = np.zeros(3)
    for image_path in tqdm(image_paths):
        try:
            img = np.array(Image.open(image_path)) / 255.0  # Assuming pixel values are in the range [0, 255]
            mean += np.mean(img, axis=(0, 1))
            std += np.std(img, axis=(0, 1))
        except:
            print(image_path)

    mean /= len(image_paths)
    std /= len(image_paths)

    return mean, std