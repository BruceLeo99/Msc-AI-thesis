from Food101_dataloader import *
from vgg16_model import *
from utils import print_gpu_memory_status
import time
import pandas as pd
import torch 
import torch.nn as nn
import torch.optim as optim

import gc
import torch.cuda

### LOAD DATA

train_dataset, val_dataset, test_dataset = lazy_load_customized(load_captions=False)

### SET UP DEVICE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### SET UP PARAMETERS

batch_size = 64
num_epochs = 70
experiment_name = "vgg16_regular"
result_foldername = experiment_name

### TRAIN MODEL

result = train_vgg16(
    train_data=train_dataset,
    val_data=val_dataset,
    model_name=experiment_name,
    device=device,
    num_epochs=num_epochs,
    batch_size=batch_size,
    early_stopping_patience=5,
    result_foldername=result_foldername,
    save_result=True
)

### TEST MODEL

test_vgg16(
    result,
    experiment_name,
    test_dataset,
    device,
    result_foldername=result_foldername,
    save_result=True,
    verbose=True
)