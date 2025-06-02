### Unimodal PBN

# This file contains the unimodal PBN model, specifically:
# - ProtoPNet 

# The model utilizes PBN to do image classification WITHOUT text input.

import os

print(os.getcwd())
print(" ")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())
print(" ")

import sys

# print(" ")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/ProtoPNet")
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "cocoapi")
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "cocoapi/PythonAPI")
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "cocoapi/PythonAPI/coco")
# print(" ")

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import time
import socket
import keyboard 
import threading
import torch.nn.functional as F
from torch.utils.data import Subset, random_split
from sklearn.metrics import classification_report, confusion_matrix
import json

from MSCOCO_preprocessing import prepare_data, MSCOCOCustomDataset, load_from_COCOAPI, show_image, retrieve_captions

from ProtoPNet.model import construct_PPNet
from ProtoPNet.train_and_test import train, test
from ProtoPNet.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features

from torchvision import transforms

import pandas as pd

# Load data 
train_data, test_data = prepare_data('dog', 'cat', 'zebra', 'giraffe', 'horse', 'pizza', 'cup', 'truck', 'train', 'airplane', 
                                     transform="vgg16", target_transform="integer", num_instances=300, test_size=0.2)

# Adjust batch sizes
train_data = DataLoader(train_data, batch_size=20, shuffle=True)  
test_data = DataLoader(test_data, batch_size=20, shuffle=False)  


# Build the model - adjust prototype shape for few-shot learning
model1 = construct_PPNet(base_architecture='vgg16', 
                        pretrained=True,
                        prototype_shape=(200, 512, 1, 1),  # 20 prototypes per class
                        num_classes=10)

# model2 = construct_PPNet(base_architecture='vgg16', 
#                         pretrained=True,
#                         prototype_shape=(300, 512, 1, 1),  # 30 prototypes per class
#                         num_classes=10)

# model3 = construct_PPNet(base_architecture='resnet18', 
#                         pretrained=True,
#                         prototype_shape=(200, 512, 1, 1),  # 20 prototypes per class
#                         num_classes=10)

# model4 = construct_PPNet(base_architecture='resnet18', 
#                         pretrained=True,
#                         prototype_shape=(300, 512, 1, 1),  # 30 prototypes per class
#                         num_classes=10)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(model1.parameters(), lr=0.0001, weight_decay=1e-4)
# optimizer2 = optim.Adam(model2.parameters(), lr=0.0001, weight_decay=1e-4)
# optimizer3 = optim.Adam(model3.parameters(), lr=0.0005, weight_decay=1e-4)
# optimizer4 = optim.Adam(model4.parameters(), lr=0.0001, weight_decay=1e-4)

# Learning rate scheduler
scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode='max', factor=0.1, patience=25)
# scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, mode='max', factor=0.1, patience=25)
# scheduler3 = optim.lr_scheduler.ReduceLROnPlateau(optimizer3, mode='max', factor=0.1, patience=25)
# scheduler4 = optim.lr_scheduler.ReduceLROnPlateau(optimizer4, mode='max', factor=0.1, patience=25)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# CUDA memory management
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
    torch.cuda.empty_cache()  # Clear GPU cache before starting
    # Set memory allocation to TensorFloat32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Print GPU memory info
    print(f"GPU Memory Usage:")
    print(f"Allocated: {torch.cuda.memory_allocated(0)//1024//1024}MB")
    print(f"Cached: {torch.cuda.memory_reserved(0)//1024//1024}MB")
    
    model1 = nn.DataParallel(model1)
    # model2 = nn.DataParallel(model2)
    # model3 = nn.DataParallel(model3)
    # model4 = nn.DataParallel(model4)
    model1 = model1.to(device)
    # model2 = model2.to(device)
    # model3 = model3.to(device)
    # model4 = model4.to(device)
    criterion = criterion.to(device)
else:
    model1 = nn.DataParallel(model1)
    # model2 = nn.DataParallel(model2)
    # model3 = nn.DataParallel(model3)
    # model4 = nn.DataParallel(model4)
    model1 = model1.to(device)
    # model2 = model2.to(device)
    # model3 = model3.to(device)
    # model4 = model4.to(device)

pause_training = False

def listen_for_pause():
    global pause_training
    while True:
        if keyboard.is_pressed('ctrl+p'):
            print("\nPause requested! Saving checkpoint and pausing training...")
            pause_training = True
            while keyboard.is_pressed('ctrl+p'):
                time.sleep(0.1)
        time.sleep(0.1)

listener_thread = threading.Thread(target=listen_for_pause, daemon=True)
listener_thread.start()

def train_and_test(model, result_name, train_data, test_data, optimizer, scheduler, device, criterion,
                   num_epochs = 70, 
                   patience = 20,
                   save_result = True):
    global pause_training
    
    try:
        # Clear GPU memory before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # open a csv file and write result line by line after each epoch
        if save_result:
            get_full_results = True
            if not os.path.exists(f'results/{result_name}.csv'):
                with open(f'results/{result_name}.csv', 'w') as f:
                    f.write("Epoch,Mode,Time,Cross Entropy,Cluster,Separation,Avg Separation,Train Accuracy,Test Accuracy,L1,P Avg Pair Dist\n")

        # Initialize variables
        best_accuracy = 0
        no_improve_count = 0

        # Start training and testing procedure
        for epoch in range(num_epochs):
            try:
                print(f'Epoch {epoch+1} of {num_epochs}')
                
                # Clear GPU memory before each epoch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                # Training
                print("Training...")
                mode, running_time, cross_entropy, cluster, separation, avg_separation, train_accuracy, l1, p_avg_pair_dist = train(model, 
                                                                                                                            train_data, 
                                                                                                                            optimizer, 
                                                                                                                            class_specific=True, 
                                                                                                                            get_full_results=get_full_results)
                accuracy = train_accuracy
                if save_result:
                    with open(f'results/{result_name}.csv', 'a') as f:
                        f.write(f"{epoch+1},{mode},{running_time},{cross_entropy},{cluster},{separation},{avg_separation},{accuracy},{l1},{p_avg_pair_dist}\n")

                # Testing
                print("Testing...")
                mode, running_time, cross_entropy, cluster, separation, avg_separation, test_accuracy, l1, p_avg_pair_dist = test(model, 
                                                                                                                          test_data, 
                                                                                                                          class_specific=True, 
                                                                                                                          get_full_results=get_full_results)
                accuracy = test_accuracy
                if save_result:
                    with open(f'results/{result_name}.csv', 'a') as f:
                        f.write(f"{epoch+1},{mode},{running_time},{cross_entropy},{cluster},{separation},{avg_separation},{accuracy},{l1},{p_avg_pair_dist}\n")

                # Learning rate scheduling
                scheduler.step(train_accuracy)
                
                # Early stopping
                if train_accuracy > best_accuracy:
                    best_accuracy = train_accuracy
                    no_improve_count = 0
                    # Save best model
                    torch.save(model.state_dict(), f'{result_name}_best_model.pth')
                else:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                        break
                
                # Pause handling
                if pause_training:
                    torch.save({
                        'epoch': epoch,
                        'epoch_left': num_epochs - epoch,
                        'model_state_dict': model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': criterion.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_accuracy': best_accuracy,
                        'no_improve_count': no_improve_count,
                        'train_accuracy': train_accuracy,
                        'test_accuracy': test_accuracy,
                        'l1': l1,
                        'p_avg_pair_dist': p_avg_pair_dist,
                        'cluster': cluster,
                        'separation': separation,
                        'avg_separation': avg_separation,
                        'cross_entropy': cross_entropy,
                        'running_time': running_time,
                        'mode': mode
                    }, f'{result_name}_model_in_progress.pth')
                    print("Checkpoint saved. Training paused. Press Ctrl+R to resume.")
                    while not keyboard.is_pressed('ctrl+r'):
                        time.sleep(0.5)
                    print("Resuming training!")
                    pause_training = False
                    
            except RuntimeError as e:
                if "out of memory" in str(e) or "CUDA error" in str(e):
                    print(f"CUDA error in epoch {epoch+1}. Attempting to recover...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        print(f"GPU Memory after error:")
                        print(f"Allocated: {torch.cuda.memory_allocated(0)//1024//1024}MB")
                        print(f"Cached: {torch.cuda.memory_reserved(0)//1024//1024}MB")
                    # Save checkpoint before potential crash
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }, f'{result_name}_error_checkpoint.pth')
                    raise  # Re-raise the exception after saving
                else:
                    raise  # Re-raise other RuntimeErrors
                    
        print(f"Best accuracy achieved: {best_accuracy:.2f}%")
        print("Done!")
        
    except Exception as e:
        print(f"Training interrupted by error: {str(e)}")
        if torch.cuda.is_available():
            print(f"Final GPU Memory State:")
            print(f"Allocated: {torch.cuda.memory_allocated(0)//1024//1024}MB")
            print(f"Cached: {torch.cuda.memory_reserved(0)//1024//1024}MB")
        raise  # Re-raise the exception for debugging

def resume_training(model, optimizer, scheduler, device, criterion, result_name):
    checkpoint = torch.load(f'{result_name}_model_in_progress.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    criterion.load_state_dict(checkpoint['loss'])
    best_accuracy = checkpoint['best_accuracy']
    no_improve_count = checkpoint['no_improve_count']
    train_accuracy = checkpoint['train_accuracy']
    test_accuracy = checkpoint['test_accuracy']
    l1 = checkpoint['l1']
    p_avg_pair_dist = checkpoint['p_avg_pair_dist']
    cluster = checkpoint['cluster']
    separation = checkpoint['separation']
    avg_separation = checkpoint['avg_separation']
    cross_entropy = checkpoint['cross_entropy']
    running_time = checkpoint['running_time']
    mode = checkpoint['mode']
    num_epochs = checkpoint['epoch']
    epoch_left = checkpoint['epoch_left']

    print(f"Resuming training from epoch {checkpoint['epoch']+1}")
    print(f"Best accuracy: {best_accuracy:.2f}%")
    print(f"No improvement count: {no_improve_count}")
    print(f"Train accuracy: {train_accuracy:.2f}%")
    print(f"Test accuracy: {test_accuracy:.2f}%")
    print(f"L1: {l1:.2f}")

    model.to(device)
    criterion.to(device)
    
    return train_and_test(model, result_name, train_data, test_data, optimizer, scheduler, device, criterion,
                   num_epochs = epoch_left, 
                   patience = 10,
                   save_result = True)
    


if __name__ == "__main__":
    try:
        # result_name = "ProtoPNet_vgg16_80train-20test-0.005lr-5classes-20prototypes"
        result_name1 = "ProtoPNet_vgg16_80train-20test-0.0001lr-10classes-20prototypes-rerun"
        # result_name2 = "ProtoPNet_vgg16_80train-20test-0.0001lr-10classes-30prototypes"
        # result_name3 = "ProtoPNet_resnet18_80train-20test-0.0005lr-10classes-20prototypes"
        # result_name4 = "ProtoPNet_resnet18_80train-20test-0.0001lr-10classes-30prototypes"

        # # Train models sequentially to manage memory better
        print("Training Model 1...")
        train_and_test(model1, result_name1, train_data, test_data, optimizer1, scheduler1, device, criterion,
                        num_epochs = 300, 
                        patience = 10,
                        save_result = True)
        
        # Clear memory before next model
        if torch.cuda.is_available():
            del model1
            torch.cuda.empty_cache()
            print("Cleared GPU memory after Model 1")
        
        # print("\nTraining Model 2...")
        # train_and_test(model2, result_name2, train_data, test_data, optimizer2, scheduler2, device, criterion,
        #                 num_epochs = 70, 
        #                 patience = 25,
        #                 save_result = True)
        
        # # Clear memory before next model
        # if torch.cuda.is_available():
        #     del model2
        #     torch.cuda.empty_cache()
        #     print("Cleared GPU memory after Model 2")
        
        # print("\nTraining Model 3...")
        # train_and_test(model3, result_name3, train_data, test_data, optimizer3, scheduler3, device, criterion,
        #                 num_epochs = 70, 
        #                 patience = 25,
        #                 save_result = True)
        
        # # Clear memory before next model
        # if torch.cuda.is_available():
        #     del model3
        #     torch.cuda.empty_cache()
        #     print("Cleared GPU memory after Model 3")


        # print("Resume training Model 2...")
        # resume_training(model2, optimizer2, scheduler2, device, criterion, result_name2)

        # # Clear memory before next model
        # if torch.cuda.is_available():
        #     del model2
        #     torch.cuda.empty_cache()
        #     print("Cleared GPU memory after Model 2")


        # print("\nTraining Model 4...")
        # train_and_test(model4, result_name4, train_data, test_data, optimizer4, scheduler4, device, criterion,
        #                 num_epochs = 70, 
        #                 patience = 25,
        #                 save_result = True)
        
        # if torch.cuda.is_available():
        #     del model4
        #     torch.cuda.empty_cache()
        #     print("Cleared GPU memory after Model 4")

    except Exception as e:
        print(f"Training process interrupted by error: {str(e)}")
        if torch.cuda.is_available():
            print(f"Final GPU Memory State:")
            print(f"Allocated: {torch.cuda.memory_allocated(0)//1024//1024}MB")
            print(f"Cached: {torch.cuda.memory_reserved(0)//1024//1024}MB")
        raise  # Re-raise the exception for debugging

    # resume_training(model, optimizer, scheduler, device, criterion, result_name)


### Before training, set up the following:
# 1. Data
# 2. Model
# 3. Loss function
# 4. Optimizer
# 5. Learning rate scheduler
# 6. Device configuration
# 7. Training loop
# 8. Early stopping
# 9. Save results to CSV (save line-by-line instead of all at once in the end.)

### Next trainings:

# 20 classes, 10, 20, 30 prototypes, 0.0001 lr, 70 epochs, 25 patience, resnet18, choose another encoder
# vgg16 is being trained on PC, collect results later.
# Start developing the multimodal PBN model

# ===============================
# New ProtoPNet Training / Testing Utilities (VGG16-style)
# ===============================

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
):
    """Train ProtoPNet in the same logging/early-stopping style used for VGG16/ResNet.

    Returns the path to the best model saved on validation accuracy.
    """

    if not os.path.exists("best_models"):
        os.makedirs("best_models")
    if not os.path.exists("results"):
        os.makedirs("results")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
        print(f"Epoch {epoch+1}/{num_epochs}")

        # ---- Training ----
        model.train()
        train_start = time.time()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            _, preds = outputs.max(1)
            train_total += labels.size(0)
            train_correct += preds.eq(labels).sum().item()

            # Free mem periodically
            del imgs, labels, outputs, preds
            if i % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        train_time = time.time() - train_start
        train_loss_avg = train_loss_sum / len(train_loader)
        train_acc = 100 * train_correct / train_total
        print(f"Train Acc: {train_acc:.2f}%  Train Loss: {train_loss_avg:.4f}  Time: {train_time:.2f}s")

        # ---- Validation ----
        model.eval()
        val_start = time.time()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss_sum += loss.item()
                _, preds = outputs.max(1)
                val_total += labels.size(0)
                val_correct += preds.eq(labels).sum().item()
                del imgs, labels, outputs, preds
        val_time = time.time() - val_start
        val_loss_avg = val_loss_sum / len(val_loader)
        val_acc = 100 * val_correct / val_total
        print(f"Val  Acc: {val_acc:.2f}%  Val Loss: {val_loss_avg:.4f}  Time: {val_time:.2f}s")

        epoch_time = train_time + val_time

        # ---- Early-stopping & LR schedule ----
        if val_acc - best_accuracy > 0.01:
            print("Model improved → saving")
            best_accuracy = val_acc
            non_update = 0
            torch.save(model.state_dict(), f"best_models/{model_name}_best.pth")
        else:
            non_update += 1
            print(f"No improvement for {non_update} epochs")

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

        # ---- CSV logging ----
        if save_result:
            with open(f"results/{model_name}_result.csv", "a") as f:
                f.write(f"{epoch+1},{train_loss_avg:.4f},{val_loss_avg:.4f},{train_acc:.2f},{val_acc:.2f},{epoch_time:.2f},{current_lr}\n")

    print("Training complete!")
    return f"best_models/{model_name}_best.pth"


def test_protopnet(model_path, experiment_name, test_data, device, num_classes, save_result=False, verbose=False):
    """Test a saved ProtoPNet model on held-out data (mirrors VGG16 test)."""

    # Ensure result dirs
    for d in ["results", "results/classification_reports", "results/confusion_matrices"]:
        os.makedirs(d, exist_ok=True)

    model = construct_PPNet(base_architecture='vgg16', pretrained=False, prototype_shape=(1,1,1,1), num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    label_names_idx = test_data.get_dataset_labels()
    idx_to_label = {idx: name for name, idx in label_names_idx.items()}

    y_true, y_pred, y_img_ids = [], [], []
    confusion_mapping = {}

    correct = 0
    with torch.no_grad():
        for idx, (img, label) in enumerate(test_loader):
            img, label = img.to(device), label.to(device)
            output = model(img)
            _, pred = output.max(1)
            t_int, p_int = label.item(), pred.item()
            true_cls, pred_cls = idx_to_label[t_int], idx_to_label[p_int]
            if t_int == p_int:
                correct += 1
            img_id = test_data.get_image_id(idx)
            y_true.append(true_cls)
            y_pred.append(pred_cls)
            y_img_ids.append(img_id)
            key = f"{t_int}_{p_int}"
            confusion_mapping.setdefault(key, []).append(img_id)
            if verbose:
                print(f"Image {img_id} – True: {true_cls}, Pred: {pred_cls}")
            del img, label, output, pred

    acc = 100 * correct / len(test_loader.dataset)
    print(f"Test Accuracy: {acc:.2f}%  ({correct}/{len(test_loader.dataset)})")

    class_report = classification_report(y_true, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=list(idx_to_label.values()))
    print(classification_report(y_true, y_pred))
    print(conf_matrix)

    results = {
        'experiment_name': experiment_name,
        'pth_filepath': model_path,
        'accuracy': acc,
        'confusion_matrix_images': confusion_mapping,
        'classification_report': class_report,
    }

    if save_result:
        with open(f"results/{experiment_name}_test_result.json", "w") as f:
            json.dump(results, f, indent=2)
        pd.DataFrame(class_report).T.to_csv(
            f"results/classification_reports/{experiment_name}_classification_report.csv")
        pd.DataFrame(conf_matrix,
                     index=list(idx_to_label.values()),
                     columns=list(idx_to_label.values())).to_csv(
                         f"results/confusion_matrices/{experiment_name}_confusion_matrix.csv")
    return results

# ===============================
# (The legacy train_and_test function remains below but is no longer used.)
# ===============================









