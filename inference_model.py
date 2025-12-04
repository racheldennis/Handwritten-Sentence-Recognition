from tqdm import tqdm

import torch

from model import CRNN
from configs import Configs

@torch.no_grad()
def infer(model, dataloader, vocab, device):
    predictions, truth = [], []
    model.eval()

    for images, labels, _, _ in tqdm(dataloader, desc="| Inferring"):
        images = images.to(device)

        outputs = model(images) # (W, B, C)

        predicted_indices = outputs.softmax(2).argmax(2)  # (W, B)
        predicted_indices = predicted_indices.permute(1, 0)  # (B, W)

        for i, seq in enumerate(predicted_indices):
            predicted_text = vocab.decode(seq)
            true_text = vocab.decode(labels[i])

            predictions.append(predicted_text)
            truth.append(true_text)
    
    return predictions, truth

def CER(predictions, truth):
    correct_chars = 0
    total_chars = 0

    for pred, label in zip(predictions, truth):
        correct_chars += sum(p == t for p, t in zip(pred, label))
        total_chars += len(label)
    
    return 1 - (correct_chars / total_chars)

def WER(predictions, truth):
    correct_words = sum(p == t for p, t in zip(predictions, truth))

    return 1 - (correct_words / len(truth))
