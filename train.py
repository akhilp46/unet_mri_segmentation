# This script was used to train UNet on M2 Pro 

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from evaluate import DiceBCELoss  
from unet import UNet 
from brain_mri_dataset import BrainMRIDataset 
from data_utils import get_data_loaders 
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from tqdm import tqdm  # progress bar

# Configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"Using device: {device}")

batch_size = 10  
learning_rate = 0.001
num_epochs = 25
checkpoint_dir = "checkpoints"

# Ensure the checkpoint directory exists
os.makedirs(checkpoint_dir, exist_ok=True)

# Initialize dataset and dataloaders
dataset_path = './data/lgg-mri-segmentation/kaggle_3m'
dataset = BrainMRIDataset(root_path=dataset_path)
train_loader, val_loader = get_data_loaders(dataset, batch_size=batch_size, train_size=0.9)
print("\nDataset Loaded.")

# Initialize the model, loss function, and optimizer
model = UNet(in_channels=3, num_classes=1)  
model.to(device)
criterion = DiceBCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print("Model Initiated. Starting Training...\n")

# Initialize lists to store losses for plotting
train_loss_history = []
val_loss_history = []

# Training loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    epoch_start_time = time.time()
    
    # Training
    model.train()
    running_train_loss = []
    train_progress_bar = tqdm(train_loader, desc="Training", leave=False)  # Progress bar for training
    for images, masks in train_progress_bar:
        images, masks = images.to(device), masks.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_train_loss.append(loss.item())
        train_progress_bar.set_postfix({"Batch Train Loss": loss.item()})  # Display batch loss

    # Validation
    model.eval()
    running_val_loss = []
    val_progress_bar = tqdm(val_loader, desc="Validation", leave=False)  # Progress bar for validation
    with torch.no_grad():
        for images, masks in val_progress_bar:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_val_loss.append(loss.item())
            val_progress_bar.set_postfix({"Batch Val Loss": loss.item()})  # Display batch loss

    # Training and Validation Loss
    epoch_train_loss = np.mean(running_train_loss)
    train_loss_history.append(epoch_train_loss)  # Append the epoch loss to history
    print(f"Training Loss: {epoch_train_loss:.4f}")

    epoch_val_loss = np.mean(running_val_loss)
    val_loss_history.append(epoch_val_loss)  # Append the epoch loss to history
    print(f"Validation Loss: {epoch_val_loss:.4f}")

    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"unet_epoch_{epoch+1}.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': epoch_train_loss,
        'val_loss': epoch_val_loss,
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history,
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

    # Timing
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    print(f"Epoch duration: {epoch_duration:.2f} seconds\n")

print("Training completed!")

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_loss_history, label="Training Loss")
plt.plot(range(1, num_epochs + 1), val_loss_history, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss over Epochs")
plt.legend()
plt.grid()
plt.savefig("loss_plot.png")  # Save the plot to a file
plt.show()

# Example on how to load the model for inference
def load_checkpoint(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded checkpoint from '{checkpoint_path}' at epoch {checkpoint['epoch']}")
    return checkpoint

# Usage example for inference:
# model = UNet(in_channels=1, num_classes=1)
# model.to(device)
# checkpoint = load_checkpoint("path_to_checkpoint.pth", model)
# model.eval()