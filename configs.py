import os
from datetime import datetime

import torch

class Configs():
    # device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # training hyperparameters
    BATCH_SIZE = 32 # 16
    LEARNING_RATE = 3e-4 # 1e-4
    EPOCHS = 50
    WEIGHT_DECAY = 1e-4

    # model hyperparameters
    IMG_HEIGHT = 32 # 32

    # CTC parameters
    BLANK_LABEL = "-"

    # paths
    MODEL_PATH = os.path.join("models", datetime.strftime(datetime.now(), "%Y%m%d-%H%M"))
    DATA_FOLDER = "data/sentences"
    LABEL_FILE = "data/ascii/sentences.txt"
