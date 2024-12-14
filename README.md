# UNet Image Segmentation

This repository contains a project for training, evaluating, and exploring the use of a U-Net model for image segmentation tasks. The implementation is modular and allows for easy customization and experimentation with datasets, model configurations, and training settings.

## Features

- **Customizable U-Net Implementation**: A robust U-Net model supporting RGB input and binary segmentation tasks.
- **Training Pipeline**: End-to-end training with support for GPU acceleration (CUDA and MPS devices).
- **Dataset Exploration**: Tools to visualize training and validation data for debugging and insights.
- **Checkpoints**: Automatic saving of model checkpoints for training continuity.
- **Loss Visualization**: Plotting of training and validation losses over epochs.

## Files and Directories

- `training_unet_gcolab.ipynb`:
  - Notebook for training the U-Net model with configurable hyperparameters.
  - Saves checkpoints and generates loss plots for performance tracking.

- `dataset_explore_n_model_eval.ipynb`:
  - Tools to load, explore, and visualize datasets.
  - Evaluate trained models on validation data.

- `utils.py`:
  - Utility functions for data visualization and processing.

- `data_utils.py`:
  - Functions for loading and preprocessing datasets, including PyTorch `DataLoader` objects.

- `unet.py`:
  - Implementation of the U-Net architecture.

## Usage

### Training the Model
Run the `training_unet_gcolab.ipynb` notebook to train the U-Net model. Adjust parameters such as batch size, learning rate, and number of epochs directly in the notebook.

### Exploring the Dataset
Use the `dataset_explore_n_model_eval.ipynb` notebook to visualize and explore the dataset. This notebook also has model evaluation on validation data.

### Visualizing Loss
Loss plots are saved as `loss_plot.png` in the checkpoint directory. Use these plots to monitor training and validation performance.

## Outputs

- **Model Checkpoints**: Saved during training for later evaluation or retraining.
- **Loss Plots**: Saved to the checkpoint directory for reference.

## Example Commands

To evaluate a specific checkpoint:
1. Update the `checkpoint_paths` list in `dataset_explore_n_model_eval.ipynb` with the desired checkpoint file.
2. Run the notebook to analyze model performance.

## Acknowledgments

This project is inspired by the U-Net architecture described in the paper [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).
