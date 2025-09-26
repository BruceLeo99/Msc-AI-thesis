from Food101_dataloader import *
from resnet34 import *
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
num_epochs = 100
experiment_name = "resnet34_baseline_food101"
result_foldername = "final_baseline_temp"

### TRAIN MODEL

# result = train_resnet34(
#     train_data=train_dataset,
#     val_data=val_dataset,
#     model_name=experiment_name,
#     device=device,
#     num_epochs=num_epochs,
#     batch_size=batch_size,
#     result_foldername=result_foldername,
#     save_result=True
# )

### TEST MODEL

test_resnet34(
    model_path="example_models_SHAPDEV/resnet34_baseline_food101_best.pth",
    experiment_name=experiment_name,
    test_data=test_dataset,
    device=device,
    result_foldername=result_foldername,
    save_result=True,
    verbose=True
)
