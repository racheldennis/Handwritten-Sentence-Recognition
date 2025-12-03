import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.model_selection import train_test_split

from dataset import HandwrittenSentenceDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

## Load data

LABEL_PATH = os.path.join("data", "ascii", "sentences.txt")
IMAGE_FOLDER_PATH = os.path.join("data", "sentences")

images, labels = [], []

with open(LABEL_PATH, "r") as file:
    for line in tqdm(file.readlines(), desc="Loading dataset"):
        # skip documentation lines
        if line.startswith("#"):
            continue

        line = line.split()

        # skip sentences with segmentation errors
        if line[2] == "err":
            continue

        image_folder_1 = line[0][:3]
        image_folder_2 = line[0][3:8].rstrip("-")
        image_file_name = line[0] + ".png"

        image_path = os.path.join(IMAGE_FOLDER_PATH, image_folder_1, image_folder_2, image_file_name)
        label = line[-1].rstrip("\n").replace("|", " ")

        images.append(image_path)
        labels.append(label)

print("|", len(images), "images found.")
print("|", len(labels), "labels found.")

## Split the data into training, validation, and test sets

print("\nSplitting data:")

images_train, images_temp, labels_train, labels_temp = train_test_split(images, labels, test_size=0.2)
images_val, images_test, labels_val, labels_test = train_test_split(images_temp, labels_temp, test_size=0.5)

print("| Data split.")

## Create datasets

print("\nCreating datasets:")

train_dataset = HandwrittenSentenceDataset(images_train, labels_train)
val_dataset = HandwrittenSentenceDataset(images_val, labels_val)
test_dataset = HandwrittenSentenceDataset(images_test, labels_test)

print("| Datasets ceated.")

## Create dataloaders

BATCH_SIZE = 32

print("\nCreating dataloaders:")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"| Training dataloader: {len(train_loader.dataset)} samples in {len(train_loader)} batches of size {BATCH_SIZE}")
print(f"| Validation dataloader: {len(val_loader.dataset)} samples in {len(val_loader)} batches of size {BATCH_SIZE}")
print(f"| Test dataloader: {len(test_loader.dataset)} samples in {len(test_loader)} batches of size {BATCH_SIZE}")
print("Dataloaders created.")
