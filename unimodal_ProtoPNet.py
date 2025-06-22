import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import json
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import socket
import keyboard 
import threading
import torch.nn.functional as F
from torch.utils.data import Subset, random_split
from sklearn.metrics import classification_report, confusion_matrix
import json

from MSCOCO_preprocessing_local import *

from ProtoPNet.model import construct_PPNet
from ProtoPNet.train_and_test_PBN import train, validate
from ProtoPNet.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from ProtoPNet.vgg_features import vgg16_features, vgg19_features
from ProtoPNet import settings
from ProtoPNet import push
from ProtoPNet.helpers import makedir
from torchvision import transforms

import pandas as pd

from collections import OrderedDict
 

def train_protopnet(
        train_data,
        val_data,
        model_name,
        device,
        num_prototypes=10,
        num_epochs=50,
        learning_rate=0.0001,
        batch_size=32,
        num_out_channels=128,
        lr_adjustment_rate=0.0001,
        lr_adjustment_mode='decrease',
        lr_adjustment_patience=5,
        save_result=False,
        early_stopping_patience=10,
        base_architecture='vgg16',
        class_specific=True,
        get_full_results=True,
        num_workers=4,
        result_foldername='results',
        push_prototypes=False  
):
    """Train ProtoPNet following the original paper's three-stage training process using settings.py.
    
    This implementation uses the original ProtoPNet settings while preserving all existing 
    variable names and function signatures for compatibility.
    
    Args:
        push_prototypes (bool): If True, performs prototype pushing during joint training.
                               This makes prototypes interpretable but is computationally expensive.
                               Pushes happen at epochs defined in settings.push_epochs.
    """

    if not os.path.exists("/var/scratch/yyg760/best_models"):
        os.makedirs("/var/scratch/yyg760/best_models")

    best_models_foldername = f"/var/scratch/yyg760/best_models"

    if not os.path.exists(result_foldername):
        os.makedirs(result_foldername)

    #_____________________________________________________________________________________________________# 

    ### LOAD MODEL ###

    num_classes = train_data.get_num_classes()

    prototype_shape = (num_classes*num_prototypes, num_out_channels, 1, 1)

    model = construct_PPNet(base_architecture=base_architecture, 
                            pretrained=True, 
                            prototype_shape=prototype_shape, 
                            num_classes=num_classes,
                            add_on_layers_type='bottleneck',
                            img_size=224)
    
    model = model.to(device)
    # Always wrap in DataParallel since ProtoPNet code expects model.module access
    model = torch.nn.DataParallel(model)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
    else:
        print(f"Using 1 GPU for training (wrapped in DataParallel for ProtoPNet compatibility)")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    ### END LOAD MODEL ###
    #_____________________________________________________________________________________________________# 

    ### LOAD DATA ###

    if num_workers == 0:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        # Push loader for prototype pushing (no normalization)
        if push_prototypes:
            push_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    else:
        train_loader = DataLoader(train_data, 
                                  batch_size=batch_size, 
                                  shuffle=True, 
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  prefetch_factor=2
                                  )
        
        val_loader = DataLoader(val_data, 
                                batch_size=batch_size, 
                                shuffle=False, 
                                num_workers=num_workers,
                                pin_memory=True,
                                prefetch_factor=2
                                )
        
        # Push loader for prototype pushing (no normalization)
        if push_prototypes:
            push_loader = DataLoader(train_data, 
                                    batch_size=batch_size, 
                                    shuffle=False, 
                                    num_workers=num_workers,
                                    pin_memory=True,
                                    prefetch_factor=2
                                    )

    # ADDED: Setup for prototype pushing if enabled
    if push_prototypes:
        img_dir = os.path.join(result_foldername, 'prototype_imgs')
        makedir(img_dir)
        prototype_img_filename_prefix = 'prototype-img'
        prototype_self_act_filename_prefix = 'prototype-self-act'
        proto_bound_boxes_filename_prefix = 'bb'
        print("Prototype pushing enabled - prototypes will be saved to:", img_dir)

    ### END LOAD DATA ###
    #_____________________________________________________________________________________________________# 

    ### SETUP MODEL HYPERPARAMS ###

    # CSV header
    if save_result:
        with open(f"{result_foldername}/{model_name}_result.csv", "w") as f:
            f.write("Stage,Epoch,Mode,Time,Cross Entropy,Cluster,Separation,Avg Cluster,Accuracy,L1,P Avg Pair Dist,Learning Rate\n")

    best_accuracy = 0.0
    best_epoch = 0
    current_lr = learning_rate
    epochs_without_improvement = 0
    lr_adjustments_made = 0
    best_model_state = None
    
    best_stage = ""

    ### END SETUP MODEL HYPERPARAMS ###
    #_____________________________________________________________________________________________________# 

    print("Starting ProtoPNet training with original paper's three-stage process...")
    if push_prototypes:
        print("Prototype pushing is ENABLED - this will increase training time but provide interpretability")
    else:
        print("Prototype pushing is DISABLED - faster training, no interpretability")
    
    ### STAGE 1: WARM-UP TRAINING - Using settings.num_warm_epochs and settings.warm_optimizer_lrs ###
    print(f"Stage 1: Warm-up training ({settings.num_warm_epochs} epochs)")
    
    # warm_optimizer using settings.warm_optimizer_lrs
    warm_optimizer = optim.Adam([
        {'params': model.module.add_on_layers.parameters(), 'lr': settings.warm_optimizer_lrs['add_on_layers']},
        {'params': model.module.prototype_vectors, 'lr': settings.warm_optimizer_lrs['prototype_vectors']}
    ])
    
    # Freeze pretrained layers for warm-up
    for p in model.module.features.parameters():
        p.requires_grad = False

    
    for epoch in range(settings.num_warm_epochs):


        print(f"Warm-up Epoch {epoch+1}/{settings.num_warm_epochs}")

        # Clear GPU memory before each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        mode, running_time, train_loss, cluster_cost, separation_cost, avg_cluster_cost, train_accuracy, l1, p_avg_pair_dist = \
        train(model, 
              train_loader, 
              warm_optimizer,  # Using warm_optimizer with settings
              class_specific=class_specific,
              get_full_results=get_full_results,
              coefs=settings.coefs) 
        
        if save_result:
            with open(f"{result_foldername}/{model_name}_result.csv", "a") as f:
                f.write(f"warm,{epoch+1},train,{running_time},{train_loss},{cluster_cost},{separation_cost},{avg_cluster_cost},{train_accuracy},{l1},{p_avg_pair_dist},{settings.warm_optimizer_lrs['add_on_layers']}\n")

        # Validation
        mode, running_time, val_loss, cluster_cost, separation_cost, avg_cluster_cost, val_accuracy, l1, p_avg_pair_dist = \
            validate(model,
             val_loader,
             class_specific=class_specific,
             get_full_results=True)
        
        # CSV logging for validation - MODIFIED: Added stage info  
        if save_result:
            with open(f"{result_foldername}/{model_name}_result.csv", "a") as f:
                f.write(f"warm,{epoch+1},validation,{running_time},{val_loss},{cluster_cost},{separation_cost},{avg_cluster_cost},{val_accuracy},{l1},{p_avg_pair_dist},{settings.warm_optimizer_lrs['add_on_layers']}\n")

        # Early-stopping & model saving logic (preserved)
        if val_accuracy > best_accuracy:
            print("Model improved → saving")
            best_accuracy = val_accuracy
            best_epoch = epoch + 1
            best_stage = "warm"  # ADDED: Track stage
            epochs_without_improvement = 0
            
            # Save best model based on validation accuracy
            best_model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            print(f"Best model saved for epoch {epoch+1} with accuracy {val_accuracy:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs (best: {best_accuracy:.4f} at epoch {best_epoch})")

    ### END STAGE 1 ###

    #_____________________________________________________________________________________________________# 

    ### STAGE 2: JOINT TRAINING - Using settings.joint_optimizer_lrs ###
    print(f"\nStage 2: Joint training ({num_epochs} epochs)")
    
    # ADDED: joint_optimizer using settings.joint_optimizer_lrs
    joint_optimizer = optim.Adam([
        {'params': model.module.features.parameters(), 'lr': settings.joint_optimizer_lrs['features']},
        {'params': model.module.add_on_layers.parameters(), 'lr': settings.joint_optimizer_lrs['add_on_layers']},
        {'params': model.module.prototype_vectors, 'lr': settings.joint_optimizer_lrs['prototype_vectors']}
    ])
    
    # Unfreeze pretrained layers for joint training
    for p in model.module.features.parameters():
        p.requires_grad = True

    # Reset epochs_without_improvement for joint training stage
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):

        
        print(f"Joint Training Epoch {epoch+1}/{num_epochs}")

        # Clear GPU memory before each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        mode, running_time, train_loss, cluster_cost, separation_cost, avg_cluster_cost, train_accuracy, l1, p_avg_pair_dist = \
        train(model, 
              train_loader, 
              joint_optimizer,
              class_specific=class_specific,
              get_full_results=get_full_results,
              coefs=settings.coefs) 
        

        # CSV logging for training 
        if save_result:
            with open(f"{result_foldername}/{model_name}_result.csv", "a") as f:
                f.write(f"joint,{epoch+1},train,{running_time},{train_loss},{cluster_cost},{separation_cost},{avg_cluster_cost},{train_accuracy},{l1},{p_avg_pair_dist},mixed\n")

        # Validation
        mode, running_time, val_loss, cluster_cost, separation_cost, avg_cluster_cost, val_accuracy, l1, p_avg_pair_dist = \
            validate(model,
             val_loader,
             class_specific=class_specific,
             get_full_results=True)
        
        # CSV logging for validation
        if save_result:
            with open(f"{result_foldername}/{model_name}_result.csv", "a") as f:
                f.write(f"joint,{epoch+1},validation,{running_time},{val_loss},{cluster_cost},{separation_cost},{avg_cluster_cost},{val_accuracy},{l1},{p_avg_pair_dist},mixed\n")

        # Prototype pushing logic (optional)
        if push_prototypes and epoch >= settings.push_start and epoch in settings.push_epochs:
            print(f"PUSHING PROTOTYPES at epoch {epoch}...")
            
            # Create epoch-specific directory for prototype images
            proto_epoch_dir = os.path.join(img_dir, f'epoch-{epoch}')
            makedir(proto_epoch_dir)
            
            # Perform prototype pushing
            push.push_prototypes(
                push_loader,  # Use push_loader (no normalization)
                prototype_network_parallel=model,
                class_specific=class_specific,
                preprocess_input_function=None,  # No preprocessing needed
                prototype_layer_stride=1,
                root_dir_for_saving_prototypes=img_dir,
                epoch_number=epoch,
                prototype_img_filename_prefix=prototype_img_filename_prefix,
                prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
                save_prototype_class_identity=True,
                log=print
            )
            
            # Test model after pushing
            print("Testing model after prototype pushing...")
            mode, running_time, push_val_loss, cluster_cost, separation_cost, avg_cluster_cost, push_val_accuracy, l1, p_avg_pair_dist = \
                validate(model,
                        val_loader,
                        class_specific=class_specific,
                        get_full_results=True)
            
            # CSV logging for post-push validation
            if save_result:
                with open(f"{result_foldername}/{model_name}_result.csv", "a") as f:
                    f.write(f"push,{epoch},validation,{running_time},{push_val_loss},{cluster_cost},{separation_cost},{avg_cluster_cost},{push_val_accuracy},{l1},{p_avg_pair_dist},push\n")
            
            print(f"Post-push validation accuracy: {push_val_accuracy:.4f}")
            
            # Check if push improved the model
            if push_val_accuracy > best_accuracy:
                print("Model improved after prototype pushing → saving")
                best_accuracy = push_val_accuracy
                best_epoch = epoch
                best_stage = "push"
                best_model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()


    ## END STAGE 2 ###

    #_____________________________________________________________________________________________________# 

    ### STAGE 3: LAST LAYER OPTIMIZATION - Using settings.last_layer_optimizer_lr ###
    
    print("\nStage 3: Last layer optimization")

    last_layer_optimizer = optim.Adam([
        {'params': model.module.last_layer.parameters(), 'lr': settings.last_layer_optimizer_lr}
    ])
    
    # Freeze all layers except last layer
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    # Run one epoch of last layer optimization with L1 regularization using settings.coefs
    # ADDED: last_layer_coefs for last layer optimization (only cross entropy and L1)
    last_layer_coefs = {
        'crs_ent': settings.coefs['crs_ent'],
        'clst': 0,  # No cluster cost for last layer
        'sep': 0,   # No separation cost for last layer  
        'l1': settings.coefs['l1']
    }
    
    mode, running_time, train_loss, cluster_cost, separation_cost, avg_cluster_cost, train_accuracy, l1, p_avg_pair_dist = \
    train(model, 
         train_loader, 
         last_layer_optimizer,
         class_specific=class_specific,
         get_full_results=get_full_results,
         coefs=last_layer_coefs)  # ADDED: Using last_layer_coefs

    # Final validation
    mode, running_time, val_loss, cluster_cost, separation_cost, avg_cluster_cost, val_accuracy, l1, p_avg_pair_dist = \
    validate(model,
            val_loader,
            class_specific=class_specific,
            get_full_results=True)

    # CSV logging for last layer optimization - ADDED: Last stage logging
    if save_result:
        with open(f"{result_foldername}/{model_name}_result.csv", "a") as f:
            f.write(f"last,1,train,{running_time},{train_loss},{cluster_cost},{separation_cost},{avg_cluster_cost},{train_accuracy},{l1},{p_avg_pair_dist},{settings.last_layer_optimizer_lr}\n")
            f.write(f"last,1,validation,{running_time},{val_loss},{cluster_cost},{separation_cost},{avg_cluster_cost},{val_accuracy},{l1},{p_avg_pair_dist},{settings.last_layer_optimizer_lr}\n")

    # Check if last layer optimization improved the model
    if val_accuracy > best_accuracy:
        print("Model improved in last layer optimization → saving")
        best_accuracy = val_accuracy
        best_stage = "last"  # ADDED: Track stage
        best_model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

    print(f"Best model saved from epoch {best_epoch} with validation accuracy {best_accuracy:.4f}")
    print(f"Best model came from {best_stage} stage")  # ADDED: Show which stage was best

    if best_model_state is not None:
        torch.save(best_model_state, f"{best_models_foldername}/{model_name}_best.pth")
        print(f"Final best model confirmed saved: {model_name}_best.pth")

    print("Training complete!")
    if push_prototypes:
        print(f"Prototype images saved in: {img_dir}")
    
    return f"{best_models_foldername}/{model_name}_best.pth"


def test_protopnet(model_path, 
                   experiment_name, 
                   test_data, 
                   device, 
                   num_prototypes=10,
                   base_architecture='vgg16',
                   class_specific=True,
                   get_full_results=True,
                   save_result=False, 
                   verbose=False,
                   use_l1_mask=False,
                   coefs=None,
                   result_foldername='results'
                   ):
    
    """Test a saved ProtoPNet model on held-out data (mirrors VGG16 test)."""

    if not os.path.exists(result_foldername):
        os.makedirs(result_foldername)

    num_classes = test_data.get_num_classes()
    prototype_shape = (num_classes*num_prototypes, 512, 1, 1)

    model = construct_PPNet(base_architecture=base_architecture, 
                            pretrained=True, 
                            prototype_shape=prototype_shape, 
                            num_classes=num_classes,
                            add_on_layers_type='bottleneck',
                            img_size=224)

    # Load state dict and handle DataParallel prefix
    state_dict = torch.load(model_path, map_location=device)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:] # remove 'module.' prefix
        else:
            name = k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    # Always wrap in DataParallel since ProtoPNet code expects model.module access
    model = torch.nn.DataParallel(model)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for testing")
    else:
        print(f"Using 1 GPU for testing (wrapped in DataParallel for ProtoPNet compatibility)")

    test_loader = DataLoader(test_data, shuffle=False)

    print("Starting model testing...")
    label_to_idx = test_data.get_dataset_labels()
    print(f"Label names and indices: {label_to_idx}")
    
    # Create reverse mapping from index to name
    idx_to_label = {idx: name for name, idx in label_to_idx.items()}
    print(f"Index to label mapping: {idx_to_label}")

    model.eval()
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0


    y_true = []
    y_pred = []
    y_img_ids = []
    confusion_mapping = {}

    for i, (image, label) in enumerate(test_loader):
        input = image.to(device)
        target = label.to(device)

        with torch.no_grad():
            output, min_distances = model(input)

            # compute loss
            cross_entropy = torch.nn.functional.cross_entropy(output, target)

            if class_specific:
                max_dist = (model.module.prototype_shape[1]
                            * model.module.prototype_shape[2]
                            * model.module.prototype_shape[3])
                
                prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,label]).to(device)
                inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
                cluster_cost = torch.mean(max_dist - inverted_distances)

                # calculate separation cost
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = \
                    torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

                # calculate avg cluster cost
                avg_separation_cost = \
                    torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
                avg_separation_cost = torch.mean(avg_separation_cost)
                
                if use_l1_mask:
                    l1_mask = 1 - torch.t(model.module.prototype_class_identity).to(device)
                    m = model.module if hasattr(model, 'module') else model
                    l1 = (m.last_layer.weight * l1_mask).norm(p=1)
                else:
                    m = model.module if hasattr(model, 'module') else model
                    l1 = m.last_layer.weight.norm(p=1) 

            else:
                min_distance, _ = torch.min(min_distances, dim=1)
                cluster_cost = torch.mean(min_distance)
                m = model.module if hasattr(model, 'module') else model
                l1 = m.last_layer.weight.norm(p=1)

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()
            
            # Store the raw integer labels for confusion matrix
            y_true.append(target.item())
            y_pred.append(predicted.item())
            y_img_ids.append(test_data.get_image_id(i))
            
            # Store mapping for confusion matrix visualization
            key = f"{target.item()}_{predicted.item()}"
            confusion_mapping.setdefault(key, []).append(test_data.get_image_id(i))

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()

        del input, target, output, predicted, min_distances


    # Convert labels to class names for individual results
    individual_prediction_results = dict()
    for test_image_id, test_label, test_pred in zip(y_img_ids, y_true, y_pred):
        individual_prediction_results[test_image_id] = {
            'true_label': idx_to_label[test_label],
            'pred_label': idx_to_label[test_pred]
        }

    # Get unique labels in sorted order for consistent matrix
    unique_labels = sorted(list(set(y_true + y_pred)))
    label_names = [idx_to_label[idx] for idx in unique_labels]

    # Generate classification report and confusion matrix using integer labels
    classi_report = classification_report(y_true, y_pred, labels=list(label_to_idx.keys()), target_names=list(label_to_idx.keys()), output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    if verbose:
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, labels=unique_labels, target_names=label_names))
        print("\nConfusion Matrix:")
        print(conf_matrix)

    test_accuracy = 100 * n_correct / n_examples
    test_loss = total_cross_entropy / n_batches
    test_cluster_cost = total_cluster_cost / n_batches
    test_separation_cost = total_separation_cost / n_batches
    test_avg_separation_cost = total_avg_separation_cost / n_batches
    test_l1 = model.module.last_layer.weight.norm(p=1).item()

    results = {
        'experiment_name': experiment_name,
        'pth_filepath': model_path,
        'accuracy': test_accuracy,
        'loss': test_loss,
        'cluster_cost': test_cluster_cost,
        'separation_cost': test_separation_cost,
        'avg_separation_cost': test_avg_separation_cost,
        'l1': test_l1,
        'confusion_matrix_images': confusion_mapping,
        'individual_prediction_results': individual_prediction_results,
        'classification_report': classi_report,
    }

    if save_result:
        with open(f"{result_foldername}/{experiment_name}_test_result.json", "w") as f:
            json.dump(results, f, indent=2)
        pd.DataFrame(classi_report).T.to_csv(
            f"{result_foldername}/classification_reports/{experiment_name}_classification_report.csv")
        pd.DataFrame(conf_matrix,
                     index=list(idx_to_label.values()),
                     columns=list(idx_to_label.values())).to_csv(
                         f"{result_foldername}/confusion_matrices/{experiment_name}_confusion_matrix.csv")
    return results


# if __name__ == "__main__":

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # Create results directory if it doesn't exist
#     if not os.path.exists("results"):
#         os.makedirs("results")

#     # Memory optimization settings
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.backends.cudnn.benchmark = True
#         torch.cuda.set_per_process_memory_fraction(0.9)

#     # Define model parameters
#     base_architecture = 'vgg16'
#     class_specific = True
#     get_full_results = True
#     num_epochs = 50
#     learning_rate = 0.0001
#     batch_size = 4
#     lr_adjustment_rate = 0.0001
#     save_result = True
#     early_stopping_patience = 10
#     lr_adjustment_patience = 5
#     experiment_name = 'ProtoPNet_testrun'

#     # Define dataset parameters
#     dataset_info = 'chosen_categories_3_10_v3.csv'
#     df_dataset_info = pd.read_csv(dataset_info)
#     categories = df_dataset_info["Category Name"].unique()
#     num_classes = len(categories)

#     # # Define dataset parameters
#     # train_data, val_data = prepare_data_manually(*categories, 
#     #                                              num_instances=15, 
#     #                                              for_test=False, 
#     #                                              split=True, 
#     #                                              split_size=0.15, 
#     #                                              experiment_name=experiment_name,
#     #                                              transform=base_architecture,
#     #                                              target_transform='integer',
#     #                                              load_captions=False,
#     #                                              save_result=save_result)
    
#     test_data = prepare_data_manually(*categories, 
#                                       num_instances=10, 
#                                       for_test=True, 
#                                       split=False, 
#                                       experiment_name=experiment_name,
#                                       transform=base_architecture,
#                                       target_transform='integer',
#                                       load_captions=False,
#                                       save_result=save_result)
    
#     # train_data, val_data, test_data = eliminate_leaked_data(experiment_name, train_data, val_data, test_data, save_result=save_result)
    

#     # best_model_path = train_protopnet(train_data, 
#     #                                   val_data, 
#     #                                   experiment_name, 
#     #                                   device, 
#     #                                   num_epochs, 
#     #                                   learning_rate, 
#     #                                   batch_size, 
#     #                                   lr_adjustment_rate, 
#     #                                   save_result, 
#     #                                   early_stopping_patience, 
#     #                                   lr_adjustment_patience, 
#     #                                   base_architecture, 
#     #                                   prototype_shape, 
#     #                                   class_specific, 
#     #                                   get_full_results=True)

#     best_model_path = "best_models/ProtoPNet_testrun_best.pth"
    
#     test_results = test_protopnet(best_model_path, 
#                                 experiment_name, 
#                                 test_data, 
#                                 device, 
#                                 base_architecture=base_architecture,
#                                 class_specific=class_specific,
#                                 get_full_results=get_full_results,
#                                 save_result=save_result, 
#                                 verbose=False,
#                                 use_l1_mask=False,
#                                 coefs=None)