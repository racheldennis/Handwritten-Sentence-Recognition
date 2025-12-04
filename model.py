from tqdm import tqdm

import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_classes, lstm_hidden_size=256, lstm_layers=2):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 32 -> 16

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 16 -> 8

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(8, 1), stride=(8, 1)), # only downsample height: 8 -> 1
        )

        cnn_output_channels = 256
        self.lstm = nn.LSTM(
            input_size=cnn_output_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=False
        )

        self.fc = nn.Linear(in_features=lstm_hidden_size * 2, out_features=num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, channels, height, width) -> (batch_size, 1, 32, W)
        conv = self.cnn(x)
        _, _, h, _ = conv.size()

        assert h == 1, f"Height must be 1 after CNN, but got {h}"

        conv = conv.squeeze(2)  # (batch_size, channels, width)
        conv = conv.permute(2, 0, 1)  # (width, batch_size, channels)

        lstm_out, _ = self.lstm(conv)  # (width, batch_size, hidden_size*2)
        out = self.fc(lstm_out)  # (width, batch_size, num_classes)

        return out.log_softmax(2)  # for CTC loss

def train_step(model, dataloader, criterion, optimizer, device):
    model.train()
    step_loss = 0

    for images, labels, image_lengths, label_lengths in tqdm(dataloader, desc="| Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels, image_lengths, label_lengths)
        
        loss.backward()
        optimizer.step()

        step_loss += loss.item()
    
    return step_loss / len(dataloader)

@torch.no_grad()
def eval_step(model, dataloader, criterion, device):
    model.eval()
    step_loss = 0

    for images, labels, image_lengths, label_lengths in tqdm(dataloader, desc="| Validating", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels, image_lengths, label_lengths)

        step_loss += loss.item()
    
    return step_loss / len(dataloader)
