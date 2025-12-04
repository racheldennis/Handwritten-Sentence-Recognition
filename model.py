import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, image_height, num_classes, lstm_hidden_size=256, lstm_layers=2):
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
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)), # only downsample height: 8 -> 4
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
        b, c, h, w = conv.size()

        assert h == 1, "Height must be 1 after CNN"

        conv = conv.squeeze(2)  # (batch_size, channels, width)
        conv = conv.permute(2, 0, 1)  # (width, batch_size, channels)

        lstm_out, _ = self.lstm(conv)  # (width, batch_size, hidden_size*2)
        out = self.fc(lstm_out)  # (width, batch_size, num_classes)

        return out.log_softmax(2)  # for CTC loss
