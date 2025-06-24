import torch, torchvision
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import numpy as np
import os
import shap
from outdated_models.MSCOCO_preprocessing_local import *
import time
import json
import sys
# Update the path to use the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

chosen_dataset_20classes = "dataset_infos/singleLabel_chosen_categories_6_20_v2.csv"
chosen_dataset_30classes = "dataset_infos/singleLabel_chosen_categories_7_30.csv"

chosen_dataset_df_20classes = pd.read_csv(chosen_dataset_20classes)
chosen_dataset_categories_20classes = chosen_dataset_df_20classes['Category Name'].unique().tolist()
test_json_20classes = "dataset_infos/singleLabel_test_data_20classes.json"

start_time = time.time()
test_data_20classes = MSCOCOCustomDataset(transform='vgg16', target_transform='integer', load_from_json=test_json_20classes)
end_time = time.time()
print(f"Time taken to load test_data_20classes: {end_time - start_time} seconds")


model_path = "best_models/vgg16_baseline_20_categories_0.0001lr_best.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load(model_path)
model.to(device)

shap_explainer = shap.DeepExplainer(model, test_data_20classes)

test_loader = DataLoader(test_data_20classes, batch_size=1, shuffle=False)
batch = next(iter(test_loader))
images, _ = batch
images = images.view(-1, 1, 28, 28)

background = images[:100]
test_images= images[100:110]

print("Calculating SHAP values...")
shap_values = shap_explainer.shap_values(test_images)

print("Converting SHAP values to numpy...")
shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

print("Plotting SHAP values...")
shap.image_plot(shap_numpy, -test_numpy)




