import numpy as np
import matplotlib.pyplot as plt
import torch

def image_convert(image):
    """
    Convert a PyTorch tensor to a numpy image array for visualization.
    
    Args:
        image (torch.Tensor): A tensor representing the image in (C, H, W) format.

    Returns:
        numpy.ndarray: The image in (H, W, C) format scaled to 0-255.
    """
    image = image.clone().cpu().numpy()  # Detach from computation graph and convert to numpy
    image = image.transpose((1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
    image = (image * 255)  # Scale to 0-255 
    return image

def mask_convert(mask):
    """
    Convert a PyTorch tensor mask to a numpy array for visualization.
    
    Args:
        mask (torch.Tensor): A tensor representing the mask in (1, H, W) format.

    Returns:
        numpy.ndarray: A 2D mask in (H, W) format.
    """
    mask = mask.clone().cpu().detach().numpy()  # Detach from computation graph and convert to numpy
    return np.squeeze(mask)  # Remove single channel dimension

def plot_img(num_images, data_loader, device='cpu'):
    """
    Plot a specified number of images and their corresponding masks from a DataLoader.
    
    Args:
        num_images (int): Number of images and masks to display.
        data_loader (torch.utils.data.DataLoader): DataLoader object to fetch data from.
        device (str): Device ('cpu', 'cuda' or 'mps'(for mac)) to move the tensors to for processing.
    """

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Fetch one batch of data
    iter_loader = iter(data_loader)
    images, masks = next(iter_loader)

    # Move tensors to the specified device
    images = images.to(device)
    masks = masks.to(device)

    plt.figure(figsize=(20, 10))

    # Plot images
    for idx in range(num_images):
        image = image_convert(images[idx])

        # Clip image data to valid range [0, 1] for imshow
        image = np.clip(image / 255.0, 0, 1)

        plt.subplot(2, num_images, idx + 1)
        plt.imshow(image)
        plt.title("Image", fontsize=14)
        plt.axis('off')

    # Plot masks
    for idx in range(num_images):
        mask = mask_convert(masks[idx])

        # Ensure mask is properly scaled to [0, 1] for visualization
        mask = np.clip(mask, 0, 1)

        plt.subplot(2, num_images, idx + num_images + 1)
        plt.imshow(mask, cmap='gray')
        plt.title("Mask", fontsize=14)
        plt.axis('off')

    plt.tight_layout()
    plt.show()
