from Food101_dataloader import *
from unimodal_ProtoPNet import *
from resnet34 import *
from utils import print_gpu_memory_status
import time
import pandas as pd
import torch 
import torch.nn as nn
import torch.optim as optim

import gc
import torch.cuda

### SET UP PARAMETERS

batch_size = 64
num_epochs = 100
num_prototypes = 5
experiment_name = "PBN_r34_5p_blip"
result_foldername = experiment_name
caption_type = "blip"

### LOAD DATA

train_dataset, val_dataset, test_dataset = lazy_load_customized(load_captions=False, caption_type=caption_type)

### SET UP DEVICE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### TRAIN MODEL

result = train_protopnet(
    train_data=train_dataset,
    val_data=val_dataset,
    model_name=experiment_name,
    device=device,
    num_prototypes=num_prototypes,
    batch_size=batch_size,
    num_epochs=num_epochs,
    use_warmup="default",
    convex_optim="default",
    result_foldername=result_foldername,
    early_stopping_patience=10,
    base_architecture='resnet34',
    save_result=True
)

### TEST MODEL

test_protopnet(
    result,
    experiment_name,
    test_dataset,
    device,
    result_foldername=result_foldername,
    save_result=True)

