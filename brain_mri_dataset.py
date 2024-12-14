import os
from torch.utils.data import Dataset
from skimage import io, transform
import numpy as np
import torch

class BrainMRIDataset(Dataset):
    def __init__(self, root_path, preload=False):
        """
        Initialize the BrainMRIDataset with the dataset root path.

        Args:
            root_path (str): Path to the root directory containing patient folders.
            preload (bool): If True, preload all images and masks into memory.
        """
        self.root_path = root_path
        self.patients = [
            file for file in os.listdir(root_path) if file not in ['data.csv', 'README.md', '.DS_Store']
        ]
        self.masks, self.images = [], []

        # Collect images and masks
        for patient in self.patients:
            for file in os.listdir(os.path.join(self.root_path, patient)):
                if 'mask' in file.split('.')[0].split('_'):
                    self.masks.append(os.path.join(self.root_path, patient, file))
                else:
                    self.images.append(os.path.join(self.root_path, patient, file))

        # Sort to maintain consistent order
        self.images = sorted(self.images)
        self.masks = sorted(self.masks)

        self.preload = preload
        if self.preload:
            # Preload all images and masks into memory
            self.preloaded_data = []
            for idx in range(len(self.images)):
                print(f"Preloading {idx + 1}/{len(self.images)}: {self.images[idx]}")  # Progress log
                self.preloaded_data.append(self._load_data(idx))
        else:
            self.preloaded_data = None

    def __len__(self):
        """
        Return the total number of image-mask pairs in the dataset.
        """
        return len(self.images)

    def _load_data(self, idx):
        """
        Internal helper method to load and preprocess a single image-mask pair.

        Args:
            idx (int): Index of the image-mask pair to load.

        Returns:
            tuple: (image, mask) where both are PyTorch tensors.
        """
        image_path = self.images[idx]
        mask_path = self.masks[idx]

        # Load image
        image = io.imread(image_path)
        if image.shape[:2] != (256, 256):  # Resize to 256x256 if not already
            image = transform.resize(image, (256, 256))
        image = image.astype(np.float32) / 255  # Normalize to [0, 1]
        image = image.transpose((2, 0, 1))  # Rearrange to channel-first format

        # Load mask
        mask = io.imread(mask_path)
        if mask.shape[:2] != (256, 256):  # Resize to 256x256 if not already
            mask = transform.resize(mask, (256, 256))
        mask = mask.astype(np.float32) / 255  # Normalize to [0, 1]
        mask = np.expand_dims(mask, axis=-1).transpose((2, 0, 1))  # Add channel dimension and rearrange

        # Convert to PyTorch tensors
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        return (image, mask)

    def __getitem__(self, idx):
        """
        Retrieve an image and its corresponding mask by index.

        Args:
            idx (int): Index of the image-mask pair to retrieve.

        Returns:
            tuple: (image, mask) where both are PyTorch tensors.
        """
        if self.preload:
            # Return preloaded data
            return self.preloaded_data[idx]
        else:
            # Load on-the-fly
            return self._load_data(idx)

# Debuging step
if __name__ == "__main__":
    data_folder = './data/lgg-mri-segmentation/kaggle_3m'
    data = BrainMRIDataset(data_folder)

    print('Length of dataset is {}'. format(data.__len__()))
    print('sample data: ')
    print(data.__getitem__(0))

    img, msk = data[0] 
    print(img.shape)
    print(img.dtype)  # Expected: torch.float32 (MPS required float32 instead of float64) (Dec 2024)
    print(msk.shape)
    print(msk.dtype)
    print("Image exists:", os.path.exists(data.images[0]))
    print("Mask exists:", os.path.exists(data.masks[0]))