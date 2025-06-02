import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import json
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, KFold
import socket
import keyboard 
import threading
import torch.nn.functional as F
from torch.utils.data import Subset, random_split
from sklearn.metrics import classification_report, confusion_matrix
import json

from MSCOCO_preprocessing import prepare_data, MSCOCOCustomDataset, load_from_COCOAPI, show_image, retrieve_captions, prepare_data_manually

from ProtoPNet.model import construct_PPNet
from ProtoPNet.train_and_test import train as protopnet_train, test as protopnet_test
from ProtoPNet.train_and_test import joint, warm_only, last_only
from ProtoPNet.push import push_prototypes
from ProtoPNet.save import save_model_w_condition
from ProtoPNet.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from ProtoPNet.vgg_features import vgg16_features, vgg19_features
from torchvision import transforms

import pandas as pd

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_protopnet(
        model,
        train_data,
        val_data,
        model_name,
        device,
        num_epochs=50,
        learning_rate=0.0001,
        batch_size=32,
        lr_increment_rate=0.0001,
        save_result=False,
        early_stopping_patience=10,
        lr_increase_patience=5,
        push_epochs=[10, 20, 30, 40],  # epochs to push prototypes
        prototype_img_dir='prototypes/',
        coefs={'crs_ent': 1, 'clst': 0.8, 'sep': -0.08, 'l1': 1e-4}
):
    """
    Train ProtoPNet in the same logging/early-stopping style used for VGG16/ResNet.
    Uses the actual ProtoPNet training functions while maintaining VGG16-style structure.
    
    Returns the path to the best model saved on validation accuracy.
    """

    if not os.path.exists("best_models"):
        os.makedirs("best_models")
    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists(prototype_img_dir):
        os.makedirs(prototype_img_dir)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # CSV header
    if save_result:
        with open(f"results/{model_name}_result.csv", "w") as f:
            f.write("Epoch,Training Loss,Validation Loss,Training Accuracy,Validation Accuracy,Time,Learning Rate\n")

    best_accuracy = 0.0
    current_lr = learning_rate
    non_update = 0
    lr_inc_count = 0

    print("Starting ProtoPNet training…")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        epoch_start = time.time()
        
        # Determine training mode based on epoch
        if epoch < 10:
            warm_only(model)  # Only train add_on_layers and prototype vectors
            print("Training mode: warm only")
        elif epoch < 30:
            joint(model)  # Train all layers
            print("Training mode: joint")
        else:
            last_only(model)  # Only train last layer
            print("Training mode: last layer only")
        
        # ---- Training Phase ----
        train_start = time.time()
        # Use ProtoPNet's training function
        train_results = protopnet_train(
            model=model, 
            dataloader=train_loader, 
            optimizer=optimizer,
            class_specific=True,
            coefs=coefs,
            log=lambda x: None,  # Suppress verbose logging
            get_full_results=True
        )
        
        # Extract results: mode, running_time, cross_entropy, cluster, separation, avg_separation, accu, l1, p_avg_pair_dist
        _, train_time, train_cross_entropy, train_cluster, train_separation, train_avg_separation, train_acc, train_l1, train_p_dist = train_results
        
        # Calculate combined training loss (matching ProtoPNet's loss calculation)
        train_loss = train_cross_entropy + coefs['clst'] * train_cluster + coefs['sep'] * train_separation + coefs['l1'] * train_l1
        
        print(f"Train Acc: {train_acc:.2f}%  Train Loss: {train_loss:.4f}  Cross-Entropy: {train_cross_entropy:.4f}")
        print(f"Cluster Cost: {train_cluster:.4f}  Separation Cost: {train_separation:.4f}  L1: {train_l1:.4f}")
        
        # ---- Validation Phase ----
        val_start = time.time()
        val_results = protopnet_test(
            model=model,
            dataloader=val_loader,
            class_specific=True,
            log=lambda x: None,
            get_full_results=True
        )
        
        _, val_time, val_cross_entropy, val_cluster, val_separation, val_avg_separation, val_acc, val_l1, val_p_dist = val_results
        val_loss = val_cross_entropy  # For validation, we only care about cross-entropy
        
        print(f"Val  Acc: {val_acc:.2f}%  Val Loss: {val_loss:.4f}")
        
        # Push prototypes at specified epochs
        if epoch in push_epochs and epoch > 0:
            print(f"\nPushing prototypes at epoch {epoch+1}...")
            push_prototypes(
                train_loader,
                prototype_network=model,
                class_specific=True,
                preprocess_input_function=None,  # Already preprocessed
                prototype_layer_stride=1,
                root_dir_for_saving_prototypes=prototype_img_dir,
                epoch_number=epoch,
                prototype_img_filename_prefix='prototype-img',
                prototype_self_act_filename_prefix='prototype-self-act',
                proto_bound_boxes_filename_prefix='bb',
                save_prototype_class_identity=True
            )
        
        epoch_time = time.time() - epoch_start
        
        # ---- Early-stopping & LR schedule ----
        if val_acc - best_accuracy > 0.01:
            print("Model improved → saving")
            best_accuracy = val_acc
            non_update = 0
            save_model_w_condition(
                model=model,
                model_dir='best_models/',
                model_name=f'{model_name}_best',
                accu=val_acc,
                target_accu=0.0,  # Always save
                log=print
            )
        else:
            non_update += 1
            print(f"No improvement for {non_update} epochs")

        # Learning rate scheduling
        if non_update >= early_stopping_patience:
            if lr_inc_count < lr_increase_patience:
                current_lr += lr_increment_rate
                for pg in optimizer.param_groups:
                    pg['lr'] = current_lr
                lr_inc_count += 1
                non_update = 0
                print(f"Learning-rate increased to {current_lr}")
            else:
                print("Early stopping triggered")
                break

        # Update scheduler
        scheduler.step()
        
        # ---- CSV logging ----
        if save_result:
            with open(f"results/{model_name}_result.csv", "a") as f:
                f.write(f"{epoch+1},{train_loss:.4f},{val_loss:.4f},{train_acc:.2f},{val_acc:.2f},{epoch_time:.2f},{current_lr}\n")

    print("Training complete!")
    return f"best_models/{model_name}_best.pth"


def test_protopnet(model_path, experiment_name, test_data, device, num_classes, 
                   prototype_shape, base_architecture='vgg16', save_result=False, verbose=False):
    """
    Test a saved ProtoPNet model on held-out data (mirrors VGG16 test).
    Provides the same outputs: classification report, confusion matrix, confusion mapping, and individual predictions.
    """

    # Ensure result dirs
    for d in ["results", "results/classification_reports", "results/confusion_matrices"]:
        os.makedirs(d, exist_ok=True)

    # Reconstruct model with proper prototype shape
    model = construct_PPNet(
        base_architecture=base_architecture, 
        pretrained=False, 
        prototype_shape=prototype_shape, 
        num_classes=num_classes
    )
    
    # Handle DataParallel wrapper
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    # Load saved weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    label_names_idx = test_data.get_dataset_labels()
    idx_to_label = {idx: name for name, idx in label_names_idx.items()}

    y_true, y_pred, y_img_ids = [], [], []
    confusion_matrix_for_each_individual = {}
    individual_prediction_results = {}

    test_correct = 0
    test_total = 0
    
    print("Starting model testing...")
    print(f"Label names and indices: {label_names_idx}")
    print(f"Index to label mapping: {idx_to_label}")
    
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Get model output
            outputs, min_distances = model(images)
            _, predicted = outputs.max(1)
            
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
            # Get metadata
            image_id = test_data.get_image_id(idx)
            image_url = test_data.get_image_url(idx)
            image_caption = test_data.get_image_caption(idx)
            
            # Convert to class names
            t_int = labels.item()
            p_int = predicted.item()
            true_class = idx_to_label[t_int]
            pred_class = idx_to_label[p_int]
            
            y_img_ids.append(image_id)
            y_true.append(true_class)
            y_pred.append(pred_class)
            
            # Create confusion matrix mapping
            cell_key = f"{t_int}_{p_int}"
            if cell_key not in confusion_matrix_for_each_individual:
                confusion_matrix_for_each_individual[cell_key] = []
            
            # Store comprehensive image information
            image_info = {
                'image_id': image_id,
                'image_url': image_url,
                'image_caption': image_caption,
                'true_label': true_class,
                'pred_label': pred_class,
                'correct': true_class == pred_class,
                'min_prototype_distances': min_distances.cpu().numpy().tolist()[0]  # Store prototype distances
            }
            confusion_matrix_for_each_individual[cell_key].append(image_info)
            
            # Individual predictions
            individual_prediction_results[image_id] = {
                'true_label': true_class,
                'pred_label': pred_class
            }
            
            if verbose:
                print(f"Image ID: {image_id}, True: {true_class}, Pred: {pred_class}")
                
            del images, labels, outputs, predicted, min_distances

    test_accuracy = 100 * test_correct / test_total
    
    print(f"\n{'='*60}")
    print("TEST RESULTS")
    print(f"{'='*60}")
    print(f"Test Accuracy: {test_accuracy:.2f}%  (Total samples: {test_total})")

    # Classification report and confusion matrix
    classi_report = classification_report(y_true, y_pred, target_names=list(idx_to_label.values()), output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=list(idx_to_label.values())))
    
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # Create comprehensive test results (matching VGG16 format)
    test_result = {
        'experiment_name': experiment_name,
        'pth_filepath': model_path,
        'label_names_idx': label_names_idx,
        'idx_to_label': idx_to_label,
        'test_accuracy': test_accuracy,
        'confusion_matrix_for_each_individual': confusion_matrix_for_each_individual,
        'individual_prediction_results': individual_prediction_results
    }

    if save_result:
        # Save main results JSON
        with open(f"results/{experiment_name}_test_result.json", "w") as f:
            json.dump(test_result, f)
        
        # Save confusion matrix image mapping separately
        with open(f"results/{experiment_name}_confusion_matrix_images.json", "w") as f:
            json.dump(confusion_matrix_for_each_individual, f, indent=2)
            
        # Save classification report
        df_classification_report = pd.DataFrame(classi_report).T
        df_classification_report.index.name = 'Class'
        df_classification_report.to_csv(f"results/classification_reports/{experiment_name}_classification_report.csv")

        # Save confusion matrix
        df_confusion_matrix = pd.DataFrame(conf_matrix)
        class_names = list(idx_to_label.values())
        df_confusion_matrix.index = class_names
        df_confusion_matrix.columns = class_names
        df_confusion_matrix.index.name = 'True Label'
        df_confusion_matrix.columns.name = 'Predicted Label'
        df_confusion_matrix.to_csv(f"results/confusion_matrices/{experiment_name}_confusion_matrix.csv")

    return test_result


# Example usage
if __name__ == "__main__":
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Memory optimization settings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_per_process_memory_fraction(0.9)
    
    # Training parameters
    num_epochs = 50
    learning_rate = 0.0001
    lr_increment_rate = 0.0001
    batch_size = 8
    validation_size = 0.15
    early_stopping_patience = 10
    lr_increase_patience = 5
    
    # Model parameters
    prototype_shape = (200, 512, 1, 1)  # 20 prototypes per class for 10 classes
    num_classes = 10
    
    experiment_name = "protopnet_vgg16_10classes_testrun"
    
    # Load data
    train_data, val_data = prepare_data_manually(
        'dog', 'cat', 'zebra', 'giraffe', 'horse', 'pizza', 'cup', 'truck', 'train', 'airplane',
        num_instances=50, 
        split=True, 
        split_size=validation_size,
        transform="vgg16", 
        target_transform="integer"
    )
    
    test_data = prepare_data_manually(
        'dog', 'cat', 'zebra', 'giraffe', 'horse', 'pizza', 'cup', 'truck', 'train', 'airplane',
        num_instances=10, 
        for_test=True,
        transform="vgg16", 
        target_transform="integer"
    )
    
    # Build model
    model = construct_PPNet(
        base_architecture='vgg16', 
        pretrained=True,
        prototype_shape=prototype_shape,
        num_classes=num_classes
    )
    
    # Wrap in DataParallel if multiple GPUs
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    # Train model
    best_model_path = train_protopnet(
        model, 
        train_data, 
        val_data, 
        experiment_name, 
        device, 
        num_epochs=num_epochs, 
        learning_rate=learning_rate,
        lr_increment_rate=lr_increment_rate,
        batch_size=batch_size, 
        early_stopping_patience=early_stopping_patience,
        lr_increase_patience=lr_increase_patience,
        save_result=True
    )
    
    # Test model
    test_protopnet(
        best_model_path, 
        experiment_name, 
        test_data, 
        device, 
        num_classes,
        prototype_shape=prototype_shape,
        base_architecture='vgg16',
        save_result=True,
        verbose=True
    ) 