### This is a temporal file to find the best number of CPU cores to use for data loading
from MSCOCO_preprocessing_local import *
from unimodal_ProtoPNet import *
from vgg16_model import *
from resnet18_model import *

import pandas as pd
import torch 
import torch.nn as nn
import torch.optim as optim

import logging

logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model1 = VGG16(num_classes=10).to(device)
model2 = VGG16(num_classes=20).to(device)
model3 = VGG16(num_classes=30).to(device)

chosen_dataset_10classes = "chosen_categories_3_10.csv"
chosen_dataset_20classes = "chosen_categories_6_20.csv"
chosen_dataset_30classes = "chosen_categories_10_30.csv"

chosen_dataset_df_10classes = pd.read_csv(chosen_dataset_10classes)
chosen_dataset_categories_10classes = chosen_dataset_df_10classes['Category Name'].unique().tolist()

chosen_dataset_df_20classes = pd.read_csv(chosen_dataset_20classes)
chosen_dataset_categories_20classes = chosen_dataset_df_20classes['Category Name'].unique().tolist()

chosen_dataset_df_30classes = pd.read_csv(chosen_dataset_30classes)
chosen_dataset_categories_30classes = chosen_dataset_df_30classes['Category Name'].unique().tolist()


experiment_name = "cpuTest"

### 10 classes dataset
train_data_10classes, val_data_10classes = prepare_data_from_preselected_categories(
    chosen_dataset_10classes,
    "train",
    load_from_local=True,
    split_val=True,
    val_size=0.2,
    experiment_name=experiment_name,
    save_result=False,
    transform='vgg16',
    target_transform='integer',
)

test_data_10classes = prepare_data_from_preselected_categories(
    chosen_dataset_10classes,
    "test",
    load_from_local=True,
    split_val=False,
    experiment_name=experiment_name,
    transform='vgg16',
    target_transform='integer',
)

train_data_10classes, val_data_10classes, test_data_10classes = eliminate_leaked_data(
    experiment_name,
    train_data_10classes,
    val_data_10classes,
    test_data_10classes
)

### 20 classes dataset
train_data_20classes, val_data_20classes = prepare_data_from_preselected_categories(
    chosen_dataset_20classes,
    "train",
    load_from_local=True,
    split_val=True,
    val_size=0.2,
    experiment_name=experiment_name,
    save_result=False,
    transform='vgg16',
    target_transform='integer',
)

test_data_20classes = prepare_data_from_preselected_categories(
    chosen_dataset_20classes,
    "test",
    load_from_local=True,
    split_val=False,
    experiment_name=experiment_name,
    transform='vgg16',
    target_transform='integer',
)

train_data_20classes, val_data_20classes, test_data_20classes = eliminate_leaked_data(
    experiment_name,
    train_data_20classes,
    val_data_20classes,
    test_data_20classes
)

### 30 classes dataset
train_data_30classes, val_data_30classes = prepare_data_from_preselected_categories(
    chosen_dataset_30classes,
    "train",
    load_from_local=True,
    split_val=True,
    val_size=0.2,
    experiment_name=experiment_name,
    save_result=False,
    transform='vgg16',
    target_transform='integer',
)

test_data_30classes = prepare_data_from_preselected_categories(
    chosen_dataset_30classes,
    "test",
    load_from_local=True,
    split_val=False,
    experiment_name=experiment_name,
    transform='vgg16',
    target_transform='integer',
)

train_data_30classes, val_data_30classes, test_data_30classes = eliminate_leaked_data(
    experiment_name,
    train_data_30classes,
    val_data_30classes,
    test_data_30classes
)

### Set up training loop
num_epochs = 3


if os.path.exists(f"results/parallel_dataloading_test_Results.txt"):
    result_mode = "a"
else:
    result_mode = "w"

# Test with 10 classes
with open(f"results/parallel_dataloading_test_Results.txt", result_mode) as f:
    f.write(f"Start testing with 10 classes\n")

for num_workers in [0,2,4,8,16,32]:
    print(f"Training with {num_workers} workers")

    with open(f"results/parallel_dataloading_test_Results.txt", "a") as f:
        f.write(f"Start testing with {num_workers} workers\n")

    train_loader_10classes = DataLoader(train_data_10classes, batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model1.parameters(), lr=0.0001)

    start_time = time.time()
    model1.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader_10classes):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model1(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    end_time = time.time()
    time_taken = end_time - start_time
    conclusion = f"Time taken to load data with {num_workers} workers: {time_taken:.2f} seconds"
    print(conclusion)
    with open(f"results/parallel_dataloading_test_Results.txt", "a") as f:
        f.write(conclusion + "\n")


# Test with 20 classes
with open(f"results/parallel_dataloading_test_Results.txt", "a") as f:
    f.write(f"Start testing with 20 classes\n")

for num_workers in [0,2,4,8,16,32]:
    print(f"Training with {num_workers} workers")
    
    with open(f"results/parallel_dataloading_test_Results.txt", "a") as f:
        f.write(f"Start testing with {num_workers} workers\n")

    train_loader_20classes = DataLoader(train_data_20classes, batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model2.parameters(), lr=0.0001)

    start_time = time.time()
    model2.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader_20classes):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model2(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    end_time = time.time()
    time_taken = end_time - start_time
    conclusion = f"Time taken to load data with {num_workers} workers: {time_taken:.2f} seconds"
    print(conclusion)
    with open(f"results/parallel_dataloading_test_Results.txt", "a") as f:
        f.write(conclusion + "\n")
    
# Test with 30 classes
with open(f"results/parallel_dataloading_test_Results.txt", "a") as f:
    f.write(f"Start testing with 30 classes\n")

for num_workers in [0,2,4,8,16,32]:
    print(f"Training with {num_workers} workers")
    
    with open(f"results/parallel_dataloading_test_Results.txt", "a") as f:
        f.write(f"Start testing with {num_workers} workers\n")

    train_loader_30classes = DataLoader(train_data_30classes, batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model3.parameters(), lr=0.0001)

    start_time = time.time()
    model3.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader_30classes):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model3(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    end_time = time.time()
    time_taken = end_time - start_time
    conclusion = f"Time taken to load data with {num_workers} workers: {time_taken:.2f} seconds"
    print(conclusion)
    with open(f"results/parallel_dataloading_test_Results.txt", "a") as f:
        f.write(conclusion + "\n")


