from Food101_dataloader import *
from multimodal_PBN import *
from utils import print_gpu_memory_status
import time
import pandas as pd
import torch 
import torch.nn as nn
import torch.optim as optim

import gc
import torch.cuda

### SET UP PARAMETERS

batch_size = 32
num_epochs = 100
num_prototypes_per_class = 2
experiment_name = "mPBN_2p_blip"
result_foldername = experiment_name
caption_type = "blip"

### SET UP DEVICE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### LOAD DATA

train_dataset, val_dataset, test_dataset = lazy_load_customized(load_captions=True, caption_type=caption_type)

### TRAIN MODEL

result = train_multimodal_PBN(
    train_dataset,
    val_dataset,
    experiment_name,
    device=device,
    num_prototypes_per_class=num_prototypes_per_class,
    batch_size=batch_size,
    num_epochs=num_epochs,
    use_warmup="default",
    convex_optim="default",
    result_foldername=result_foldername,
    early_stopping_patience=10,
    save_result=True
)

### TEST MODEL
test_multimodal_PBN(
    result,
    experiment_name,
    test_dataset,
    device,
    result_foldername=result_foldername,
    save_result=True)

