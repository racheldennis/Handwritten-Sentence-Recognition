import os
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from dataset import HandwrittenSentenceDataset
from configs import Configs

from utils.visualization import show_before_after

configs = Configs()

print(f"Device: {configs.DEVICE}\n")

## Load data

images, labels, vocab = [], [], set()

with open(configs.LABEL_FILE, "r") as file:
    for line in tqdm(file.readlines(), desc="Loading dataset"):
        # skip documentation lines
        if line.startswith("#"):
            continue

        line = line.split()

        # skip sentences with segmentation errors
        if line[2] == "err":
            continue

        image_folder_1 = line[0][:3]
        image_folder_2 = line[0][:8].rstrip("-")
        image_file_name = line[0] + ".png"

        image_path = os.path.join(configs.DATA_FOLDER, image_folder_1, image_folder_2, image_file_name)
        label = line[-1].rstrip("\n").replace("|", " ")

        images.append(image_path)
        labels.append(label)
        vocab.update(set(label))

vocab = "".join(sorted(vocab))

print("|", len(images), "images found.")
print("|", len(labels), "labels found.")

## Split the data into training, validation, and test sets

print("\nSplitting data:")

images_train, images_temp, labels_train, labels_temp = train_test_split(images, labels, test_size=0.2)
images_val, images_test, labels_val, labels_test = train_test_split(images_temp, labels_temp, test_size=0.5)

print("| Data split.")

## Create transforms

train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),

    transforms.Resize(32),

    transforms.RandomApply([
        transforms.RandomAffine(degrees=3, translate=(0.03, 0.03), scale=(0.95, 1.05), shear=3)], p=0.7
    ),

    transforms.RandomPerspective(distortion_scale=0.05, p=0.3),

    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5
    ),

    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.6))], p=0.2
    ),

    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

val_test_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

## Visualize transforms

image_path = images_train[0]

raw_image = Image.open(image_path).convert("L")
transformed_image = train_transforms(raw_image)

show_before_after(raw_image, transformed_image)

## Create datasets

print("\nCreating datasets:")

train_dataset = HandwrittenSentenceDataset(images_train, labels_train, vocab, train_transforms)
val_dataset = HandwrittenSentenceDataset(images_val, labels_val, vocab, val_test_transforms)
test_dataset = HandwrittenSentenceDataset(images_test, labels_test, vocab, val_test_transforms)

print("| Datasets ceated.")

## Create dataloaders

print("\nCreating dataloaders:")

train_loader = DataLoader(train_dataset, batch_size=configs.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=configs.BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=configs.BATCH_SIZE, shuffle=False)

print(f"| Training dataloader: {len(train_loader.dataset)} samples in {len(train_loader)} batches of size {configs.BATCH_SIZE}")
print(f"| Validation dataloader: {len(val_loader.dataset)} samples in {len(val_loader)} batches of size {configs.BATCH_SIZE}")
print(f"| Test dataloader: {len(test_loader.dataset)} samples in {len(test_loader)} batches of size {configs.BATCH_SIZE}")
print("Dataloaders created.")
