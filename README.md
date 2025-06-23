# Drivable Area Segmentation with DINO and U-Net

This repository contains a PyTorch implementation of a U-Net-based segmentation model for identifying the drivable road area. The model leverages a pre-trained DINO (self-**DI**stillation with **NO** labels) ResNet-50 as the encoder backbone, which is known for its powerful feature extraction capabilities. The primary dataset used for this task is the [Indian Driving Dataset (IDD)](http://idd.insaan.iiit.ac.in/).

## Features

  * **U-Net Architecture:** A robust U-Net architecture is used for semantic segmentation.
  * **DINO Pre-trained Encoder:** Utilizes the powerful DINO ResNet-50 model as the feature extractor, improving performance with less training.
  * **Indian Driving Dataset (IDD) Support:** Includes scripts for preparing the IDD dataset, from generating segmentation labels to creating data loaders.
  * **End-to-End Pipeline:** Provides a complete workflow from data preparation and training to inference on video files.
  * **Loss & Metrics:** Combines Dice Loss and Cross-Entropy Loss for stable training and uses the Jaccard Index (IoU) for evaluation.
  * **Video Inference:** Comes with an inference script that can process a video, apply the segmentation model to each frame, and produce an overlayed output video.
  * **Temporal Smoothing:** Implements Exponential Moving Average (EMA) during inference to create smoother and more stable video segmentation.

## Model Architecture

The model, `UNetResNet50`, is a U-Net with a frozen DINO ResNet-50 encoder.

  * **Encoder:** The encoder consists of the convolutional layers from the pre-trained `dino_resnet50` model available via `torch.hub`. The layers `layer1` through `layer4` produce feature maps at different scales, which serve as skip connections.
  * **Decoder:** The decoder is built with custom `DecoderBlock` modules. Each block takes the output from the previous decoder layer and a skip connection from the corresponding encoder layer. It upsamples the feature map and applies two `Conv2dReLU` layers.
  * **Segmentation Head:** A final 1x1 convolution maps the decoder's output to the desired number of classes (2 in this case: `road` and `background`).

## Getting Started

### 1\. Prerequisites

Clone the repository and install the required dependencies.

```bash
git clone <repository-url>
cd <repository-directory>
pip install -r requirements.txt
```

Make sure you have a `requirements.txt` file with the following contents:

```
torch
torchvision
pandas
opencv-python
numpy
matplotlib
tqdm
torchmetrics
segmentation-models-pytorch
```

### 2\. Data Preparation

1.  **Download the Dataset:** Download the **IDD 20k II** dataset from the [official website](http://idd.insaan.iiit.ac.in/dataset/details/). You will need the `gtFine` (annotations) and `leftImg8bit` (images) parts of the dataset.

2.  **Organize Dataset:** Unzip the files and organize them into the following directory structure:

    ```
    idd20kII/
    ├── gtFine/
    │   ├── train/
    │   └── val/
    └── leftImg8bit/
        ├── train/
        └── val/
    ```

3.  **Generate Segmentation Masks:** The `gtFine` folder contains JSON files with polygon annotations. Run the `create_labels.py` script to convert these into pixel-wise segmentation masks (`.png` files).

    ```bash
    python dino/create_labels.py
    ```

    This script will process the JSON files and save the corresponding label images with the `_newlevel3Id.png` suffix in the same directories.

4.  **Create CSV Files:** Run the `IDD_Dataset.py` script to match the images with their corresponding masks and generate `train_IDD.csv` and `val_IDD.csv`. These files are essential for the data loaders.

    ```bash
    python dino/IDD_Dataset.py
    ```

    This will create a `code_files/` directory and save the CSVs inside.

### 3\. Training the Model

To start training the model, run the `train_dino.py` script.

```bash
python dino/train_dino.py
```

  * The script will load the datasets using the CSV files created earlier.
  * Checkpoints containing the model state, optimizer, and training history will be saved to the `code_files/dino_checkpoints/` directory after each epoch.
  * You can resume training from a checkpoint by setting `load_from_ckpt = True` and providing the checkpoint path in the script.

### 4\. Running Inference

The `dino_inference.py` script is used to generate segmentation videos. (Inference was done on another dataset: https://github.com/SullyChen/driving-datasets)

1.  **Prepare Input Video:** The script is currently configured to read a CSV file of image paths (`ncp_steering_data.csv`) and create an input video (`sullychen.mp4`). You can adapt this to use your own video or sequence of images.

2.  **Run Inference:** Execute the script to process the input video and generate an overlayed output.

    ```bash
    python dino/dino_inference.py
    ```

<!-- end list -->

  * The script requires a trained model checkpoint. Make sure to update the `ckpt` path in `dino_inference.py` to point to your saved model (e.g., `code_files/dino_checkpoints/model_epoch20.pth`).
  * The output video (`sullychen_overlayed_10x10_200.mp4`) will show the original video with the segmented drivable area highlighted in red.
  * The script includes post-processing steps like morphological operations and temporal smoothing (EMA) to reduce flicker and improve the visual quality of the output.

## Acknowledgements

  * The label creation script (`create_labels.py`) is adapted from the official [AutoNUE public-code repository](https://github.com/AutoNUE/public-code).
  * This project relies on the excellent work from the DINO authors and the creators of the Indian Driving Dataset.
  * Parts of the Script was inherited from the learnopencv website
