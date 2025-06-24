from outdated_models.MSCOCO_preprocessing_local import *
from unimodal_ProtoPNet import *
from vgg16_model import *
from resnet18_model import *

import pandas as pd
import torch 
import torch.nn as nn
import torch.optim as optim

chosen_dataset_10classes = "dataset_infos/chosen_categories_3_10.csv"
chosen_dataset_20classes = "dataset_infos/chosen_categories_6_20.csv"
chosen_dataset_30classes = "dataset_infos/chosen_categories_10_30.csv"

chosen_dataset_df_10classes = pd.read_csv(chosen_dataset_10classes)
chosen_dataset_categories_10classes = chosen_dataset_df_10classes['Category Name'].unique().tolist()
train_json_10classes = "dataset_infos/train_data_10classes.json"
val_json_10classes = "dataset_infos/val_data_10classes.json"
test_json_10classes = "dataset_infos/test_data_10classes.json"

chosen_dataset_df_20classes = pd.read_csv(chosen_dataset_20classes)
chosen_dataset_categories_20classes = chosen_dataset_df_20classes['Category Name'].unique().tolist()
train_json_20classes = "dataset_infos/train_data_20classes.json"
val_json_20classes = "dataset_infos/val_data_20classes.json"
test_json_20classes = "dataset_infos/test_data_20classes.json"

chosen_dataset_df_30classes = pd.read_csv(chosen_dataset_30classes)
chosen_dataset_categories_30classes = chosen_dataset_df_30classes['Category Name'].unique().tolist()
train_json_30classes = "dataset_infos/train_data_30classes.json"
val_json_30classes = "dataset_infos/val_data_30classes.json"
test_json_30classes = "dataset_infos/test_data_30classes.json"


experiment_name1 = "vgg16_baseline_10_categories_0.0001lr"
experiment_name2 = "resnet18_baseline_10_categories_0.0001lr"
experiment_name3 = "ProtoPNet_vgg16_10_categories_20prototypes_0.0001lr"
experiment_name4 = "ProtoPNet_resnet18_10_categories_20prototypes_0.0001lr"
experiment_name5 = "ProtoPNet_vgg16_20_categories_20prototypes_0.0001lr"
experiment_name6 = "ProtoPNet_resnet18_20_categories_30prototypes_0.0001lr"

########################################################
# DATA PREPARATION
########################################################

## 10 classes
start_time = time.time()
train_data_10classes = MSCOCOCustomDataset(transform='vgg16', target_transform='integer', load_from_json=train_json_10classes)
val_data_10classes = MSCOCOCustomDataset(transform='vgg16', target_transform='integer', load_from_json=val_json_10classes)
test_data_10classes = MSCOCOCustomDataset(transform='vgg16', target_transform='integer', load_from_json=test_json_10classes)
end_time = time.time()
print(f"Time taken to load train_data_10classes: {end_time - start_time} seconds")

## 20 classes
start_time = time.time()
train_data_20classes = MSCOCOCustomDataset(transform='vgg16', target_transform='integer', load_from_json=train_json_20classes)
val_data_20classes = MSCOCOCustomDataset(transform='vgg16', target_transform='integer', load_from_json=val_json_20classes)
test_data_20classes = MSCOCOCustomDataset(transform='vgg16', target_transform='integer', load_from_json=test_json_20classes)
end_time = time.time()
print(f"Time taken to load train_data_20classes: {end_time - start_time} seconds")

## 30 classes
start_time = time.time()
train_data_30classes = MSCOCOCustomDataset(transform='vgg16', target_transform='integer', load_from_json=train_json_30classes)
val_data_30classes = MSCOCOCustomDataset(transform='vgg16', target_transform='integer', load_from_json=val_json_30classes)
test_data_30classes = MSCOCOCustomDataset(transform='vgg16', target_transform='integer', load_from_json=test_json_30classes)
end_time = time.time()

print("Data preparation complete")

########################################################
# MODEL PREPARATION
########################################################

num_epochs = 120
batch_size = 64
learning_rate = 0.0001
lr_adjustment_rate = 0.0001
num_folds = 5
early_stopping_patience = 10
lr_adjustment_patience = 5
num_gpus = torch.cuda.device_count()

print(f"Using batch size {batch_size} with {num_gpus} GPUs")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


########################################################
# MODEL TRAINING AND TESTING
########################################################

### EXPERIMENT 1: VGG16 BASELINE
best_vgg16_baseline_path = train_vgg16(
    train_data_10classes,
    val_data_10classes,
    experiment_name1,
    device,
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    batch_size=batch_size,
    lr_adjustment_rate=lr_adjustment_rate,
    save_result=True,
    early_stopping_patience=early_stopping_patience,
    lr_adjustment_patience=lr_adjustment_patience,
    num_workers=4
)


test_vgg16(
    best_vgg16_baseline_path,
    experiment_name1,
    test_data_10classes,
    device,
    num_classes=10,
    save_result=True,
)


### EXPERIMENT 2: RESNET18 BASELINE
best_resnet18_baseline_path = train_resnet18(
    train_data_10classes,
    val_data_10classes,
    experiment_name2,
    device,
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    batch_size=batch_size,
    lr_adjustment_rate=lr_adjustment_rate,
    save_result=True,
    early_stopping_patience=early_stopping_patience,
    lr_adjustment_patience=lr_adjustment_patience,
    num_workers=4
)

test_resnet18(
    best_resnet18_baseline_path,
    experiment_name2,
    test_data_10classes,
    device,
    num_classes=10,
    save_result=True
)

### EXPERIMENT 3: ProtoPNet VGG16 10 classes
best_vgg16_protopnet_path_10classes = train_protopnet(
    train_data_10classes,
    val_data_10classes,
    experiment_name3,
    device,
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    batch_size=batch_size,
    lr_adjustment_rate=lr_adjustment_rate,
    save_result=True,
    early_stopping_patience=early_stopping_patience,
    lr_adjustment_patience=lr_adjustment_patience,
    class_specific=True,
    num_workers=4
)

test_protopnet(
    best_vgg16_protopnet_path_10classes,
    experiment_name3,
    test_data_10classes,
    device,
    num_classes=10,
    prototype_shape=(200, 512, 1, 1),
    base_architecture='vgg16',
    save_result=True
)

### EXPERIMENT 4: ProtoPNet ResNet18 10 classes
best_resnet18_protopnet_path_10classes = train_protopnet(
    train_data_10classes,
    val_data_10classes,
    experiment_name4,
    device,
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    batch_size=batch_size,
    lr_adjustment_rate=lr_adjustment_rate,
    save_result=True,
    early_stopping_patience=early_stopping_patience,
    lr_adjustment_patience=lr_adjustment_patience,
    class_specific=True,
    num_workers=4
)

test_protopnet(
    best_resnet18_protopnet_path_10classes,
    experiment_name4,
    test_data_10classes,
    device,
    num_classes=10,
    prototype_shape=(300, 512, 1, 1),
    base_architecture='resnet18',
    save_result=True,
)

### EXPERIMENT 5: ProtoPNet VGG16 20 classes
best_vgg16_protopnet_path_20classes = train_protopnet(
    train_data_20classes,
    val_data_20classes,
    experiment_name5,
    device,
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    batch_size=batch_size,
    lr_adjustment_rate=lr_adjustment_rate,
    save_result=True,
    early_stopping_patience=early_stopping_patience,
    lr_adjustment_patience=lr_adjustment_patience,
    class_specific=True,
    num_workers=4
)

test_protopnet(
    best_vgg16_protopnet_path_20classes,
    experiment_name5,
    test_data_20classes,
    device,
    num_classes=20,
    prototype_shape=(400, 512, 1, 1),
    base_architecture='vgg16',
    save_result=True
)

### EXPERIMENT 6: ProtoPNet ResNet18 20 classes
best_resnet18_protopnet_path_20classes = train_protopnet(
    train_data_20classes,
    val_data_20classes,
    experiment_name6,
    device,
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    batch_size=batch_size,
    lr_adjustment_rate=lr_adjustment_rate,
    save_result=True,
    early_stopping_patience=early_stopping_patience,
    lr_adjustment_patience=lr_adjustment_patience,
    class_specific=True,
    num_workers=4
)

test_protopnet(
    best_resnet18_protopnet_path_20classes,
    experiment_name6,
    test_data_20classes,
    device,
    num_classes=20,
    prototype_shape=(600, 512, 1, 1),
    base_architecture='resnet18',
    save_result=True
)


