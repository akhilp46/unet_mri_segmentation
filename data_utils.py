import torch
from torch.utils.data import random_split, DataLoader
from brain_mri_dataset import BrainMRIDataset

def split_dataset(dataset, train_size=0.9, random_seed=72):
    """
    Splits the given dataset into training and validation sets.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to split.
        train_size (float): Proportion of the dataset to include in the training set (0 < train_size < 1).
        random_seed (int): Seed for reproducibility.

    Returns:
        trainset, valset: Subsets of the original dataset for training and validation.
    """
    # Calculate sizes for the splits
    total_size = len(dataset)
    train_len = int(total_size * train_size)
    val_len = total_size - train_len

    # Perform the split
    trainset, valset = random_split(
        dataset, [train_len, val_len], generator=torch.Generator().manual_seed(random_seed)
    )

    return trainset, valset


def get_data_loaders(dataset, batch_size=10, train_size=0.9, preload=False, shuffle=True, num_workers=4,random_seed=72):
    """
    Creates DataLoader objects for training and validation datasets.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to split and load.
        batch_size (int): Number of samples per batch.
        train_size (float): Proportion of the dataset for training (0 < train_size < 1).
        shuffle (bool): Whether to shuffle the training data.
        random_seed (int): Seed for reproducibility.

    Returns:
        train_loader, val_loader: DataLoader objects for training and validation.
    """
    # Split the dataset into training and validation sets
    trainset, valset = split_dataset(dataset, train_size=train_size, random_seed=random_seed)

    # Create DataLoaders
    if preload: 
        train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=0)
        val_loader = DataLoader(dataset=valset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    else:
        train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=4)
        val_loader = DataLoader(dataset=valset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    return train_loader, val_loader