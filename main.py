from MSCOCO_preprocessing_local import *
from unimodal_ProtoPNet import *
from vgg16_model import *
from resnet18_model import *

import pandas as pd
import torch 
import torch.nn as nn
import torch.optim as optim

chosen_dataset_10classes = "chosen_categories_3_10.csv"
chosen_dataset_20classes = "chosen_categories_6_20.csv"

chosen_dataset_df_10classes = pd.read_csv(chosen_dataset_10classes)
chosen_dataset_categories_10classes = chosen_dataset_df_10classes['Category Name'].unique().tolist()

chosen_dataset_df_20classes = pd.read_csv(chosen_dataset_20classes)
chosen_dataset_categories_20classes = chosen_dataset_df_20classes['Category Name'].unique().tolist()


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
train_data_10classes, val_data_10classes = prepare_data_from_preselected_categories(
    chosen_dataset_10classes,
    data_type="train",
    split_val=True,
    val_size=0.2,
    transform='vgg16',
    target_transform='integer',
    load_captions=False,
    save_result=False
)

test_data_10classes = prepare_data_from_preselected_categories(
    chosen_dataset_10classes,
    data_type="test",
    transform='vgg16',
    target_transform='integer',
    load_captions=False,
    save_result=False
)

train_data_10classes,val_data_10classes,test_data_10classes = eliminate_leaked_data(
    experiment_name=experiment_name1,
    train_data=train_data_10classes,
    val_data=val_data_10classes,
    test_data=test_data_10classes,
    verbose=True,
    save_result=True
)


### 20 classes
train_data_20classes, val_data_20classes = prepare_data_from_preselected_categories(
    chosen_dataset_20classes,
    data_type="train",
    split_val=True,
    val_size=0.2,
    transform='vgg16',
    target_transform='integer',
    load_captions=False,
    save_result=False
)

test_data_20classes = prepare_data_from_preselected_categories(
    chosen_dataset_20classes,
    data_type="test",
    transform='resnet18',
    target_transform='integer',
    load_captions=False,
    save_result=False
)

train_data_20classes, val_data_20classes, test_data_20classes = eliminate_leaked_data(
    experiment_name=experiment_name2,
    train_data=train_data_20classes,
    val_data=val_data_20classes,
    test_data=test_data_20classes,
    verbose=True,
    save_result=True
)

print("Data preparation complete")

########################################################
# MODEL PREPARATION
########################################################

num_epochs = 150
base_batch_size = 16
num_gpus = torch.cuda.device_count()
batch_size = base_batch_size * max(1, num_gpus)  # Scale batch size with GPU count
print(f"Using batch size {batch_size} with {num_gpus} GPUs")
learning_rate = 0.0001
lr_increment_rate = 0.0001
num_folds = 5
early_stopping_patience = 10
lr_increase_patience = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_vgg16_baseline = VGG16(num_classes=10)
model_resnet18_baseline = ResNet18(num_classes=10)

# Enable multi-GPU for baseline models
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for baseline models")
    model_vgg16_baseline = torch.nn.DataParallel(model_vgg16_baseline)
    model_resnet18_baseline = torch.nn.DataParallel(model_resnet18_baseline)

model_protopnet_vgg16_10classes = construct_PPNet(
    base_architecture='vgg16',
    num_classes=10,
    prototype_shape=(200, 512, 1, 1),
)

model_protopnet_resnet18_10classes = construct_PPNet(
    base_architecture='resnet18',
    num_classes=10,
    prototype_shape=(300, 512, 1, 1),
)

model_protopnet_vgg16_20classes = construct_PPNet(
    base_architecture='vgg16',
    num_classes=20,
    prototype_shape=(400, 512, 1, 1),
)

model_protopnet_resnet18_20classes = construct_PPNet(
    base_architecture='resnet18',
    num_classes=20,
    prototype_shape=(600, 512, 1, 1),
)

########################################################
# MODEL TRAINING AND TESTING
########################################################

### EXPERIMENT 1: VGG16 BASELINE
best_vgg16_baseline_path = train_vgg16(
    model_vgg16_baseline,
    train_data_10classes,
    val_data_10classes,
    experiment_name1,
    device,
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    batch_size=batch_size,
    lr_increment_rate=lr_increment_rate,
    save_result=True,
    early_stopping_patience=early_stopping_patience,
    lr_increase_patience=lr_increase_patience
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
    model_resnet18_baseline,
    train_data_10classes,
    val_data_10classes,
    experiment_name2,
    device,
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    batch_size=batch_size,
    lr_increment_rate=lr_increment_rate,
    save_result=True,
    early_stopping_patience=early_stopping_patience,
    lr_increase_patience=lr_increase_patience
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
    model_protopnet_vgg16_10classes,
    train_data_10classes,
    val_data_10classes,
    experiment_name3,
    device,
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    batch_size=batch_size,
    lr_increment_rate=lr_increment_rate,
    save_result=True,
    early_stopping_patience=early_stopping_patience,
    lr_increase_patience=lr_increase_patience,
    class_specific=True
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
    model_protopnet_resnet18_10classes,
    train_data_10classes,
    val_data_10classes,
    experiment_name4,
    device,
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    batch_size=batch_size,
    lr_increment_rate=lr_increment_rate,
    save_result=True,
    early_stopping_patience=early_stopping_patience,
    lr_increase_patience=lr_increase_patience,
    class_specific=True
)

test_protopnet(
    best_resnet18_protopnet_path_10classes,
    experiment_name4,
    test_data_10classes,
    device,
    num_classes=10,
    prototype_shape=(300, 512, 1, 1),
    base_architecture='resnet18',
    save_result=True
)

### EXPERIMENT 5: ProtoPNet VGG16 20 classes
best_vgg16_protopnet_path_20classes = train_protopnet(
    model_protopnet_vgg16_20classes,
    train_data_20classes,
    val_data_20classes,
    experiment_name5,
    device,
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    batch_size=batch_size,
    lr_increment_rate=lr_increment_rate,
    save_result=True,
    early_stopping_patience=early_stopping_patience,
    lr_increase_patience=lr_increase_patience,
    class_specific=True
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
    model_protopnet_resnet18_20classes,
    train_data_20classes,
    val_data_20classes,
    experiment_name6,
    device,
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    batch_size=batch_size,
    lr_increment_rate=lr_increment_rate,
    save_result=True,
    early_stopping_patience=early_stopping_patience,
    lr_increase_patience=lr_increase_patience,
    class_specific=True
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


