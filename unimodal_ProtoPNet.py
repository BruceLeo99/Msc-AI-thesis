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

from ProtoPNet.model import construct_PPNet
from ProtoPNet.train_and_test_PBN import train, validate, warm_only, joint, last_only
from ProtoPNet.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from ProtoPNet.vgg_features import vgg16_features, vgg19_features
from ProtoPNet import settings

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
        lr_adjustment_rate=0,
        lr_adjustment_mode='none',
        lr_adjustment_patience=0,
        use_warmup=None,
        convex_optim=None,
        save_result=False,
        early_stopping_patience=10,
        base_architecture='vgg16',
        class_specific=True,
        get_full_results=True,
        num_workers=4,
        result_foldername='results'
):
    """Train ProtoPNet following the original paper's three-stage training process using settings.py.
    
    This implementation uses the original ProtoPNet settings while preserving all existing 
    variable names and function signatures for compatibility.
    """

    if not os.path.exists("/var/scratch/yyg760/best_models"):
        os.makedirs("/var/scratch/yyg760/best_models")

    best_models_foldername = f"/var/scratch/yyg760/best_models"

    if not os.path.exists(result_foldername):
        os.makedirs(result_foldername)

    #_____________________________________________________________________________________________________# 

    ### LOAD MODEL ###

    num_classes = train_data.get_num_classes()
    # Following original ProtoPNet paper specifications for different architectures
    if base_architecture == 'vgg16' or base_architecture == 'vgg19' or base_architecture == 'densenet121' or base_architecture == 'densenet161':
        num_channels = 128  
    # elif base_architecture == 'resnet34':
    #     num_channels = 256  
    else:
        num_channels = 512  

    if num_prototypes == 1:
        prototype_shape = (num_classes*num_prototypes, num_channels, 7, 7)
    else:
        prototype_shape = (num_classes*num_prototypes, num_channels, 1, 1)

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
    
    ### END LOAD MODEL ###
    #_____________________________________________________________________________________________________# 

    ### LOAD DATA ###

    if num_workers == 0:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

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

    ### END LOAD DATA ###
    #_____________________________________________________________________________________________________# 

    ### SETUP MODEL HYPERPARAMS ###

    # CSV header
    if save_result:
        with open(f"{result_foldername}/{model_name}_result.csv", "w") as f:
            f.write("Stage,Epoch,Mode,Time,Cross Entropy,Cluster,Separation,Avg Cluster,Accuracy,L1,P Avg Pair Dist\n")

    best_joint_accuracy = 0.0
    best_joint_epoch = 0
    epochs_without_improvement_joint = 0

    best_last_accuracy = 0.0
    best_last_epoch = 0
    epochs_without_improvement_last = 0

    current_lr = learning_rate
    
    lr_adjustments_made = 0

    best_model_state = None
    
    ### END SETUP MODEL HYPERPARAMS ###
    #_____________________________________________________________________________________________________# 

    print("Starting ProtoPNet training with original paper's three-stage process...")
    
    ### STAGE 1: WARM-UP TRAINING - Using settings.num_warm_epochs and settings.warm_optimizer_lrs ###
    if use_warmup is None:
        print("Stage 1: Warm-up training: skipped")
    else:
        print(f"Stage 1: Warm-up training ({settings.num_warm_epochs} epochs)")
        
        # warm_optimizer using settings.warm_optimizer_lrs
        warm_optimizer = optim.Adam([
            {'params': model.module.add_on_layers.parameters(), 'lr': settings.warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
            {'params': model.module.prototype_vectors, 'lr': settings.warm_optimizer_lrs['prototype_vectors']}
        ])
        
        if use_warmup == "default":
            num_warm_epochs = settings.num_warm_epochs
        else:
            num_warm_epochs = int(use_warmup)

        for epoch in range(num_warm_epochs):
            print(f"Warm-up Epoch {epoch+1}/{num_warm_epochs} (using {use_warmup} settings)")

            # Clear GPU memory before each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Freeze pretrained layers for warm-up
            warm_only(model)

            # Start main training loop
            mode, running_time, train_loss, cluster_cost, separation_cost, avg_cluster_cost, train_accuracy, l1, p_avg_pair_dist = \
            train(model, 
                train_loader, 
                warm_optimizer,  # Using warm_optimizer with settings
                class_specific=class_specific,
                get_full_results=get_full_results,
                coefs=settings.coefs) 
            
            if save_result:
                with open(f"{result_foldername}/{model_name}_result.csv", "a") as f:
                    f.write(f"warm,{epoch+1},train,{running_time},{train_loss},{cluster_cost},{separation_cost},{avg_cluster_cost},{train_accuracy},{l1},{p_avg_pair_dist}\n")

            # Validation
            mode, running_time, val_loss, cluster_cost, separation_cost, avg_cluster_cost, val_accuracy, l1, p_avg_pair_dist = \
                validate(model,
                val_loader,
                class_specific=class_specific,
                get_full_results=True)
            
            if save_result:
                with open(f"{result_foldername}/{model_name}_result.csv", "a") as f:
                    f.write(f"warm,{epoch+1},validation,{running_time},{val_loss},{cluster_cost},{separation_cost},{avg_cluster_cost},{val_accuracy},{l1},{p_avg_pair_dist}\n")

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

    for epoch in range(num_epochs):

        print(f"Joint Training Epoch {epoch+1}/{num_epochs}")

        # Clear GPU memory before each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        joint(model)

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
                f.write(f"joint,{epoch+1},train,{running_time},{train_loss},{cluster_cost},{separation_cost},{avg_cluster_cost},{train_accuracy},{l1},{p_avg_pair_dist}\n")

        # Validation
        mode, running_time, val_loss, cluster_cost, separation_cost, avg_cluster_cost, val_accuracy, l1, p_avg_pair_dist = \
            validate(model,
             val_loader,
             class_specific=class_specific,
             get_full_results=True)
        
        # CSV logging for validation
        if save_result:
            with open(f"{result_foldername}/{model_name}_result.csv", "a") as f:
                f.write(f"joint,{epoch+1},validation,{running_time},{val_loss},{cluster_cost},{separation_cost},{avg_cluster_cost},{val_accuracy},{l1},{p_avg_pair_dist}\n")

        
        # Early stopping: register the best joint accuracy and epoch, then decide whether to stop
        if val_accuracy > best_joint_accuracy:
            best_joint_accuracy = val_accuracy
            best_joint_epoch = epoch + 1
            epochs_without_improvement_joint = 0
            best_model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            print(f"Best joint model saved for epoch {epoch+1} with accuracy {val_accuracy:.2f}")
        else:
            epochs_without_improvement_joint += 1
            print(f"No improvement for {epochs_without_improvement_joint} epochs (best: {best_joint_accuracy:.2f} at epoch {best_joint_epoch})")
        
        if epochs_without_improvement_joint >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1} with best joint accuracy {best_joint_accuracy:.2f} at epoch {best_joint_epoch}")
            break

    ## END STAGE 2 ###

    #_____________________________________________________________________________________________________# 

    ### STAGE 3: LAST LAYER OPTIMIZATION - Using settings.last_layer_optimizer_lr ###
    if convex_optim is None:
        print("Stage 3: Last layer optimization: skipped")
    else:
    
        print("\nStage 3: Last layer optimization")

        last_layer_optimizer = optim.Adam([
            {'params': model.module.last_layer.parameters(), 'lr': settings.last_layer_optimizer_lr}
        ])

        # Run one epoch of last layer optimization with L1 regularization using settings.coefs
        last_layer_coefs = {
            'crs_ent': settings.coefs['crs_ent'],
            'clst': 0,  # No cluster cost for last layer
            'sep': 0,   # No separation cost for last layer  
            'l1': settings.coefs['l1']
        }
        
        if convex_optim == "default":
            num_last_epochs = 5
        else:
            num_last_epochs = int(convex_optim)

        for last_epoch in range(num_last_epochs): 
            last_only(model)
            mode, running_time, train_loss, cluster_cost, separation_cost, avg_cluster_cost, train_accuracy, l1, p_avg_pair_dist = \
            train(model, 
                train_loader, 
                last_layer_optimizer,
                class_specific=class_specific,
                get_full_results=get_full_results,
                coefs=last_layer_coefs)

            # Final validation
            mode, running_time, val_loss, cluster_cost, separation_cost, avg_cluster_cost, val_accuracy, l1, p_avg_pair_dist = \
            validate(model,
                    val_loader,
                    class_specific=class_specific,
                    get_full_results=True)

            # CSV logging for last layer optimization
            if save_result:
                with open(f"{result_foldername}/{model_name}_result.csv", "a") as f:
                    f.write(f"last,{last_epoch+1},train,{running_time},{train_loss},{cluster_cost},{separation_cost},{avg_cluster_cost},{train_accuracy},{l1},{p_avg_pair_dist}\n")
                    f.write(f"last,{last_epoch+1},validation,{running_time},{val_loss},{cluster_cost},{separation_cost},{avg_cluster_cost},{val_accuracy},{l1},{p_avg_pair_dist}\n")

            # early stopping: register the best last accuracy and epoch, then decide whether to stop
            if val_accuracy > best_last_accuracy:
                print(f"Model improved in last layer optimization → saving at epoch {epoch+1} with accuracy {val_accuracy:.2f}")
                best_last_accuracy = val_accuracy
                best_last_epoch = epoch + 1
                epochs_without_improvement_last = 0
                best_model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

            if epochs_without_improvement_last >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1} with best last accuracy {best_last_accuracy:.2f} at epoch {best_last_epoch}")
                break

    ### END STAGE 3 ###
    #_____________________________________________________________________________________________________# 

    if best_model_state is not None:
        model_config = {
            'base_architecture': base_architecture,
            'pretrained': True,
            'prototype_shape': prototype_shape,
            'num_classes': num_classes,
            'add_on_layers_type': 'bottleneck',
            'img_size': 224
        }

        torch.save({'model_config': model_config, 'state_dict': best_model_state}, f"{best_models_foldername}/{model_name}_best.pth")
        print(f"Final best model confirmed saved: {model_name}_best.pth")

    
    return f"{best_models_foldername}/{model_name}_best.pth"


def test_protopnet(model_path, 
                   experiment_name, 
                   test_data, 
                   device, 
                   class_specific=True,
                   get_full_results=True,
                   save_result=False, 
                   verbose=True,
                   use_l1_mask=False,
                   coefs=None,
                   result_foldername='results'
                   ):
    
    """Test a saved ProtoPNet model on held-out data (mirrors VGG16 test)."""

    if not os.path.exists(result_foldername):
        os.makedirs(result_foldername)
    
    if not os.path.exists(f"{result_foldername}/classification_reports"):
        os.makedirs(f"{result_foldername}/classification_reports")
    
    if not os.path.exists(f"{result_foldername}/confusion_matrices"):
        os.makedirs(f"{result_foldername}/confusion_matrices")

    # Load state dict and handle DataParallel prefix
    model_checkpoint = torch.load(model_path, map_location=device)
    model_config = model_checkpoint['model_config']
    state_dict = model_checkpoint['state_dict']

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:] 
        else:
            name = k
        new_state_dict[name] = v

    model = construct_PPNet(
    base_architecture=model_config['base_architecture'],
    pretrained=model_config['pretrained'],
    prototype_shape=model_config['prototype_shape'],
    num_classes=model_config['num_classes'],
    add_on_layers_type=model_config['add_on_layers_type'],
    img_size=model_config['img_size'])
      
    model.load_state_dict(new_state_dict)

    # Always wrap in DataParallel since ProtoPNet code expects model.module access
    model = torch.nn.DataParallel(model)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for testing")
    else:
        print(f"Using 1 GPU for testing (wrapped in DataParallel for ProtoPNet compatibility)")

    test_loader = DataLoader(test_data, shuffle=False)

    print("Starting model testing...")
    label_to_idx = test_data.label_name_idx
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
    classi_report = classification_report(y_true, y_pred, labels=list(label_to_idx.values()), target_names=list(label_to_idx.keys()), output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    if verbose:
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, labels=list(label_to_idx.values()), target_names=list(label_to_idx.keys())))
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
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label,
        'accuracy': test_accuracy,
        'loss': test_loss,
        'cluster_cost': test_cluster_cost,
        'separation_cost': test_separation_cost,
        'avg_separation_cost': test_avg_separation_cost,
        'l1': test_l1,
        'confusion_matrix_for_each_individual': confusion_mapping,
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

