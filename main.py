import os
import argparse
from tqdm import tqdm
from PIL import Image
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from dataset import HandwrittenSentenceDataset, collate_fn
from configs import Configs
from model import CRNN, train_step, eval_step
from inference_model import infer, CER, WER

from utils.visualization import show_transformations, show_predictions
from utils.transforms import ResizeHeight
from utils.vocab import Vocab

## Setup argparse

parser = argparse.ArgumentParser(description="Handwritten Sentence Recognition")

parser.add_argument("--train", action="store_true", help="Run training loop")
parser.add_argument("--test", action="store_true", help="Run inference on test set")
parser.add_argument("--show-transforms", action="store_true", help="Visualize transformations before and after")
parser.add_argument("--show-predictions", action="store_true", help="Visualize model predictions")

args = parser.parse_args()

# enforce dependencies
if args.show_predictions and not args.test:
    raise ValueError("--show-predictions requires --test to be specified.")

## Setup configs

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

print("|", len(images), "images found.")
print("|", len(labels), "labels found.")

## Save vocab

vocab = Vocab("".join(sorted(vocab)), configs.BLANK_LABEL)

## Split the data into training, validation, and test sets

print("\nSplitting data:")

images_train, images_temp, labels_train, labels_temp = train_test_split(images, labels, test_size=0.2)
images_val, images_test, labels_val, labels_test = train_test_split(images_temp, labels_temp, test_size=0.5)

print("| Data split.")

## Create transforms

train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),

    ResizeHeight(configs.IMG_HEIGHT),

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
    ResizeHeight(configs.IMG_HEIGHT),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

if args.show_transforms:
    ## Visualize transforms

    image_path = images_train[0]

    raw_image = Image.open(image_path).convert("L")
    transformed_image = train_transforms(raw_image)

    show_transformations(raw_image, transformed_image)

## Create datasets

print("\nCreating datasets:")

train_dataset = HandwrittenSentenceDataset(images_train, labels_train, vocab, train_transforms)
val_dataset = HandwrittenSentenceDataset(images_val, labels_val, vocab, val_test_transforms)
test_dataset = HandwrittenSentenceDataset(images_test, labels_test, vocab, val_test_transforms)

print("| Datasets ceated.")

## Create model

num_classes = len(vocab) + 1  # +1 for CTC blank label

model = CRNN(num_classes).to(configs.DEVICE)

## Create dataloaders

print("\nCreating dataloaders:")

train_loader = DataLoader(train_dataset, batch_size=configs.BATCH_SIZE, shuffle=True, collate_fn=partial(collate_fn, model=model))
val_loader = DataLoader(val_dataset, batch_size=configs.BATCH_SIZE, shuffle=False, collate_fn=partial(collate_fn, model=model))
test_loader = DataLoader(test_dataset, batch_size=configs.BATCH_SIZE, shuffle=False, collate_fn=partial(collate_fn, model=model))

print(f"| Training dataloader: {len(train_loader.dataset)} samples in {len(train_loader)} batches of size {configs.BATCH_SIZE}")
print(f"| Validation dataloader: {len(val_loader.dataset)} samples in {len(val_loader)} batches of size {configs.BATCH_SIZE}")
print(f"| Test dataloader: {len(test_loader.dataset)} samples in {len(test_loader)} batches of size {configs.BATCH_SIZE}")
print("Dataloaders created.\n")

## TRAINING LOOP

if args.train:
    ## Define loss and optimizer

    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=configs.LEARNING_RATE, weight_decay=configs.WEIGHT_DECAY)

    ## Training loop

    os.makedirs(configs.MODEL_PATH, exist_ok=True)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in tqdm(range(1, configs.EPOCHS + 1), desc="Training epochs"):
        train_loss = train_step(model, train_loader, ctc_loss, optimizer, configs.DEVICE)
        val_loss = eval_step(model, val_loader, ctc_loss, configs.DEVICE)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{configs.MODEL_PATH}/model_{epoch}.pth")
            print(f"| New best model saved with val loss {best_val_loss:.4f} at epoch {epoch}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch}/{configs.EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

## TODO: Plot losses

## INFERENCE ON TEST SET

if args.test:
    ## Run inference

    print("Running inference:")

    inference_model = CRNN(num_classes).to(configs.DEVICE)
    inference_model.load_state_dict(torch.load(f"models/20251204-0016/model_88.pth", map_location=configs.DEVICE))

    predictions, truth = infer(inference_model, test_loader, vocab, configs.DEVICE)

    print("Inference complete.\n")

    ## Calculate accuracy

    print("Calculating accuracy:")

    cer_score = CER(predictions, truth)
    wer_score = WER(predictions, truth)

    print("| Character accuracy:", cer_score)
    print("| Word accuracy:", wer_score)

    ## TODO: Visualize predictions

    if args.show_predictions:
        images_batch, _, _, _ = next(iter(test_loader))

        show_predictions(images_batch, predictions, truth)
