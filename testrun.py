from outdated_models.MSCOCO_preprocessing_local import *
from unimodal_ProtoPNet import *
from vgg16_model import *
from resnet18_model import *
from utils import print_gpu_memory_status
import time
import pandas as pd
import torch 
import torch.nn as nn
import torch.optim as optim

import gc
import torch.cuda

from utils import set_seed

set_seed(42)

########################################################
# DATA PREPARATION
########################################################

# chosen_dataset_20classes = "dataset_infos/singleLabel_chosen_categories_6_20_v2.csv"
# chosen_dataset_30classes = "dataset_infos/singleLabel_chosen_categories_7_30.csv"

train_json_20classes = "dataset_infos/singleLabel_train_data_20classes.json"
val_json_20classes = "dataset_infos/singleLabel_val_data_20classes.json"
test_json_20classes = "dataset_infos/singleLabel_test_data_20classes.json"

train_json_30classes = "dataset_infos/singleLabel_train_data_30classes.json"
val_json_30classes = "dataset_infos/singleLabel_val_data_30classes.json"
test_json_30classes = "dataset_infos/singleLabel_test_data_30classes.json"


experiment_name1 = "testrun"
# experiment_name2 = "resnet18_baseline_30_categories_0.0001lr"
# experiment_name3 = "ProtoPNet_resnet18_20_categories_10prototypes_0.0001lr"
# experiment_name4 = "ProtoPNet_resnet18_30_categories_10prototypes_0.0001lr"

# result_foldername = '/var/scratch/yyg760/testrun' 
# result_foldername = '/home/yyg760/testrun' 
result_foldername = 'results'
result_foldername_compare = 'results_compare'

## 20 classes
start_time = time.time()
train_data_20classes = MSCOCOCustomDataset(transform='resnet18', target_transform='integer', load_from_json=train_json_20classes)
val_data_20classes = MSCOCOCustomDataset(transform='resnet18', target_transform='integer', load_from_json=val_json_20classes)
test_data_20classes = MSCOCOCustomDataset(transform='resnet18', target_transform='integer', load_from_json=test_json_20classes)
end_time = time.time()
print(f"Time taken to load train_data_20classes: {end_time - start_time} seconds")

## 30 classes
start_time = time.time()
train_data_30classes = MSCOCOCustomDataset(transform='resnet18', target_transform='integer', load_from_json=train_json_30classes)
val_data_30classes = MSCOCOCustomDataset(transform='resnet18', target_transform='integer', load_from_json=val_json_30classes)
test_data_30classes = MSCOCOCustomDataset(transform='resnet18', target_transform='integer', load_from_json=test_json_30classes)
end_time = time.time()

print("Data preparation complete")

########################################################
# MODEL PREPARATION
########################################################

num_epochs = 5
batch_size = 16
learning_rate = 0.0001
lr_adjustment_rate = 0
lr_adjustment_mode = 'decrease'
num_folds = 5
early_stopping_patience = 5
lr_adjustment_patience = 4
base_architecture = 'resnet18'
num_gpus = torch.cuda.device_count()

print(f"Using batch size {batch_size} with {num_gpus} GPUs")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


########################################################
# MODEL TRAINING AND TESTING
########################################################

### EXPERIMENT 1: RESNET18 BASELINE 20 CLASSES
print("\nStarting Experiment 1: RESNET18 Baseline 20 Classes")
print_gpu_memory_status()

# experiment_path1 = train_resnet18(
#     train_data_20classes,
#     val_data_20classes,
#     experiment_name1,
#     device,
#     num_epochs=num_epochs,
#     learning_rate=learning_rate,
#     batch_size=batch_size,
#     lr_adjustment_rate=lr_adjustment_rate,
#     lr_adjustment_mode=lr_adjustment_mode,
#     lr_adjustment_patience=lr_adjustment_patience,
#     save_result=True,
#     early_stopping_patience=early_stopping_patience,
#     num_workers=4,
#     result_foldername=result_foldername
# )

# print("\nAfter training Experiment 1:")
# print_gpu_memory_status()


test_resnet18(
    "testrun_best.pth",
    experiment_name1,
    test_data_20classes,
    device,
    save_result=True,
    result_foldername=result_foldername
)

test_resnet18(
    "testrun_best.pth",
    experiment_name1,
    test_data_20classes,
    device,
    save_result=True,
    result_foldername=result_foldername_compare
)

# Clear memory after experiment 1
torch.cuda.empty_cache()
gc.collect()

# ### EXPERIMENT 2: RESNET18 BASELINE 30 CLASSES
# print("\nStarting Experiment 2: RESNET18 Baseline 30 Classes")
# print_gpu_memory_status()

# experiment_path2 = train_resnet18(
#     train_data_30classes,
#     val_data_30classes,
#     experiment_name2,
#     device,
#     num_epochs=num_epochs,
#     learning_rate=learning_rate,
#     batch_size=batch_size,
#     lr_adjustment_rate=lr_adjustment_rate,
#     lr_adjustment_mode=lr_adjustment_mode,
#     lr_adjustment_patience=lr_adjustment_patience,
#     save_result=True,
#     early_stopping_patience=early_stopping_patience,
#     num_workers=4,
#     result_foldername=result_foldername
# )

# print("\nAfter training Experiment 2:")
# print_gpu_memory_status()


# test_resnet18(
#     experiment_path2,
#     experiment_name2,
#     test_data_30classes,
#     device,
#     save_result=True,
#     result_foldername=result_foldername
# )

# # Clear memory after experiment 2
# torch.cuda.empty_cache()
# gc.collect()

# ### EXPERIMENT 3: ProtoPNet RESNET18 20 CLASSES 10 PROTOTYPES
# print("\nStarting Experiment 3: ProtoPNet RESNET18 20 Classes")
# print_gpu_memory_status()

# experiment_path3 = train_protopnet(
#     train_data_20classes,
#     val_data_20classes,
#     experiment_name3,
#     device,
#     num_prototypes=10,
#     num_epochs=num_epochs,
#     learning_rate=learning_rate,
#     batch_size=batch_size,
#     lr_adjustment_rate=lr_adjustment_rate,
#     lr_adjustment_mode=lr_adjustment_mode,
#     lr_adjustment_patience=lr_adjustment_patience,
#     save_result=True,
#     early_stopping_patience=early_stopping_patience,
#     base_architecture=base_architecture,
#     class_specific=True,
#     get_full_results=True,
#     num_workers=4,
#     result_foldername=result_foldername
# )

# print("\nAfter training Experiment 3:")
# print_gpu_memory_status()


# test_protopnet(
#     experiment_path3,
#     experiment_name3,
#     test_data_20classes,
#     device,
#     num_prototypes=10,
#     base_architecture=base_architecture,
#     class_specific=True,
#     get_full_results=True,
#     save_result=True,
#     result_foldername=result_foldername
# )

# # Clear memory after experiment 3
# torch.cuda.empty_cache()
# gc.collect()

# ### EXPERIMENT 4: ProtoPNet RESNET18 30 CLASSES 10 PROTOTYPES
# print("\nStarting Experiment 4: ProtoPNet RESNET18 30 Classes")
# print_gpu_memory_status()

# experiment_path4 = train_protopnet(
#     train_data_30classes,
#     val_data_30classes,
#     experiment_name4,
#     device,
#     num_prototypes=10,
#     num_epochs=num_epochs,
#     learning_rate=learning_rate,
#     batch_size=batch_size,
#     lr_adjustment_rate=lr_adjustment_rate,
#     lr_adjustment_mode=lr_adjustment_mode,
#     lr_adjustment_patience=lr_adjustment_patience,
#     save_result=True,
#     early_stopping_patience=early_stopping_patience,
#     base_architecture=base_architecture,
#     class_specific=True,
#     get_full_results=True,
#     num_workers=4,
#     result_foldername=result_foldername
# )

# print("\nAfter training Experiment 4:")
# print_gpu_memory_status()


# test_protopnet(
#     experiment_path4,
#     experiment_name4,
#     test_data_30classes,
#     device,
#     num_prototypes=10,
#     base_architecture=base_architecture,
#     class_specific=True,
#     get_full_results=True,
#     save_result=True,
#     result_foldername=result_foldername
# )

# # Clear memory after experiment 4
# torch.cuda.empty_cache()
# gc.collect()



