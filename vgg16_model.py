# vgg16 model without prototypes as baseline

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CyclicLR
from sklearn.model_selection import train_test_split, KFold
import time
import os
import pandas as pd
from torch.utils.data import random_split, Subset
from sklearn.metrics import classification_report, confusion_matrix
import json



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7*7*512, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))

        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
    
def train_vgg16(
        train_data,
        val_data,
        model_name,
        device,
        num_epochs = 10,
        learning_rate = 0.0001,
        batch_size = 32,
        save_result = False,
        early_stopping_patience = 10,
        num_workers = 0,
        result_foldername='results'
):
    
    """
    Trains a VGG16 model on the training set and validates it on the validation set.
    Data is preloaded and splitted into train and validation sets manually.
    Returns the path to the best model.

    Args:
        train_data: Training data
        val_data: Validation data
        model_name: Name of the model
        device: Device to use for training
        num_epochs: Number of epochs to train
        learning_rate: Learning rate for training
        batch_size: Batch size for training
        save_result: Whether to save the result
        early_stopping_patience: Patience for early stopping
        num_workers: Number of workers for data loading
    """

    if not os.path.exists(f"/var/scratch/yyg760/best_models"):
        os.makedirs(f"/var/scratch/yyg760/best_models")

    best_models_foldername = f"/var/scratch/yyg760/best_models"

    if not os.path.exists(result_foldername):
        os.makedirs(result_foldername)

    model = VGG16(num_classes=train_data.get_num_classes()).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = torch.nn.DataParallel(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Clear GPU cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if num_workers == 0:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, shuffle=False)
    else:
        train_loader = DataLoader(
            train_data, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2
        )
        val_loader = DataLoader(
            val_data, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,
        )
    
    # Original single train/val split training
    print("Starting Training.\nModel: VGG16 baseline with No Prototypes")

    # Train model
    if save_result:
        with open(f"{result_foldername}/{model_name}_result.csv", "w") as f:
            f.write("Epoch,Training Loss,Validation Loss,Training Accuracy,Validation Accuracy,Time\n")

    best_accuracy = 0
    epochs_without_improvement = 0
    best_model_state = None


    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1} of {num_epochs}")

        ### Training phase
        train_start_time = time.time()
        model.train()
        
        train_correct = 0
        train_total = 0
        
        # Train and keep backpropagating on batches
        for i, (images, labels) in enumerate(train_loader, 0):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate training accuracy
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # Clear variables to free memory
            del images, labels, outputs, predicted
            
            # Clear GPU cache periodically during training
            if i % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Calculate training accuracy for this epoch
        train_accuracy = 100 * train_correct / train_total

        # Clear GPU cache after training phase
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        train_end_time = time.time()
        train_time_spent = train_end_time - train_start_time
        print(f"Epoch {epoch+1}, Train Loss: {loss.item()}, Train Accuracy: {train_accuracy:.2f}%, Time: {train_time_spent:.2f} seconds")
        
        ### Validation phase
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            val_loss_total = 0
            val_start_time = time.time()
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                
                # Calculate validation loss
                val_loss = criterion(outputs, labels)
                val_loss_total += val_loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Clear variables to free memory
                del images, labels, outputs
                
            # Clear GPU cache after validation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            val_end_time = time.time()
            val_time = val_end_time - val_start_time
            val_accuracy = 100 * correct / total
            val_loss_avg = val_loss_total / len(val_loader)
            print(f"Accuracy of the model on the {total} validation images: {val_accuracy:.2f}%")
            print(f"Validation Loss: {val_loss_avg:.4f}")
            print(f"Validation Time: {val_time:.2f} seconds")

            ### Early Stopping & LR scheduling
            # Use validation accuracy for early stopping, then save and dynamically update the best model 
            if val_accuracy > best_accuracy:
                print("Model performance improved")
                best_accuracy = val_accuracy
                best_epoch = epoch + 1
                epochs_without_improvement = 0

                # Save best model based on validation accuracy
                best_model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                # torch.save(best_model_state, f"{best_models_foldername}/{model_name}_best.pth")
                print(f"Best model saved for epoch {epoch+1} with accuracy {val_accuracy:.4f}")

            else:
                epochs_without_improvement += 1
                print(f"Model performance did not improve for {epochs_without_improvement} times")

            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1} due to no improvement in validation accuracy after {early_stopping_patience} times")
                break

        # Calculate time spent for one epoch and save the result
        epoch_time = train_time_spent + val_time
        if save_result:
            with open(f"{result_foldername}/{model_name}_result.csv", "a") as f:
                f.write(f"{epoch+1},{loss.item()},{val_loss_avg:.4f},{train_accuracy:.2f},{val_accuracy:.2f},{epoch_time:.2f}\n")
    
    # End of training 
    print("Training Complete!")

    if best_model_state is not None:
        torch.save(best_model_state, f"{best_models_foldername}/{model_name}_best.pth")
        print(f"Final best model confirmed saved: {model_name}_best.pth")

    return f"{best_models_foldername}/{model_name}_best.pth"


def train_vgg16_with_CV(
        train_data,
        n_folds,
        num_epochs,
        learning_rate,
        model_name,
        device,
        batch_size,
        lr_adjustment_rate = 0.0001,
        lr_adjustment_mode = 'increase',
        save_result = False,
        early_stopping_patience = 10,
        lr_adjustment_patience = 5,
        random_state = 42,
        num_workers = 0,
        result_foldername='results'
):
    
    """
    Trains a VGG16 model with K-Fold Cross-Validation on the training set and validates it on the validation set.
    Only training data needed to be passed, which will be split into K-Fold train-validation set pairs.
    Returns the path to the best model.

    Args:
        train_data: Training data
        n_val_splits: Number of folds for cross-validation
        num_epochs: Number of epochs to train
        learning_rate: Learning rate for training
        model_name: Name of the model
        device: Device to use for training
        batch_size: Batch size for training
        save_result: Whether to save the result
        lr_adjustment_rate: Learning rate increment rate
        early_stopping_patience: Number of epochs to wait before early stopping
        lr_adjustment_patience: Number of epochs to wait before increasing learning rate
        random_state: Random state for cross-validation. For reproducibility, set 42 by default.
        num_workers: Number of workers for data loading. For better GPU utilization, set 0 by default.
    """
    if not os.path.exists(f"/var/scratch/yyg760/best_models"):
        os.makedirs(f"/var/scratch/yyg760/best_models")

    best_models_foldername = f"/var/scratch/yyg760/best_models"
    
    if not os.path.exists(result_foldername):
        os.makedirs(result_foldername)

    # Clear GPU cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Starting Training with {n_folds}-fold Cross-Validation.\nModel: VGG16 baseline with No Prototypes")
    
    if save_result:
        with open(f"{result_foldername}/{model_name}_cv_result.csv", "w") as f:
            f.write("Fold,Epoch,Training Loss,Validation Loss,Training Accuracy,Validation Accuracy,Time,Learning Rate\n")

    best_model_state = None

    # Split data into K-Fold 
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    fold_results = []
    
    for fold, (train_index, val_index) in enumerate(kfold.split(train_data)):
        print(f"\n{'='*60}")
        print(f"Start training for FOLD {fold+1}/{n_folds}")
        print(f"{'='*60}")

        # Create fresh model for each fold
        fold_model = VGG16(num_classes=train_data.get_num_classes()).to(device)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for training")
            fold_model = torch.nn.DataParallel(fold_model)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(fold_model.parameters(), lr=learning_rate)

        # Training variables for this fold
        best_val_accuracy = 0
        current_lr = learning_rate
        non_update_count = 0
        lr_increase_count = 0

        train_subset = Subset(train_data, train_index)
        val_subset = Subset(train_data, val_index)

        if num_workers == 0:
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False) 
        else:
            train_loader = DataLoader(
                train_subset, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                prefetch_factor=2
            )
            val_loader = DataLoader(
                val_subset, 
                batch_size=batch_size, 
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                prefetch_factor=2
            )

        print(f"Fold {fold+1} - Train: {len(train_subset)}, Val: {len(val_subset)}")

        # Training loop for this fold
        for epoch in range(num_epochs):
            print(f"Fold {fold+1}, Epoch {epoch+1}/{num_epochs}")

            ### Training phase
            train_start_time = time.time()
            fold_model.train()
            train_correct = 0
            train_total = 0
            epoch_loss = 0

            # Train and keep backpropagating on batches
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = fold_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Calculate training accuracy
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                epoch_loss += loss.item()

                # Clear GPU cache periodically during training
                if i % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Clear variables to free memory
                del images, labels, outputs, predicted

            # Calculate training accuracy for this epoch
            train_accuracy = 100 * train_correct / train_total
            avg_loss = epoch_loss / len(train_loader)

            # Clear GPU cache after training phase
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            train_end_time = time.time()
            train_time_spent = train_end_time - train_start_time

            print(f"Train Acc: {train_accuracy:.2f}%, Loss: {avg_loss:.4f}")
            print(f"Train Time: {train_time_spent:.2f}s")
                
            ### Validation phase
            val_start_time = time.time()
            fold_model.eval()
            val_correct = 0
            val_total = 0
            val_loss_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = fold_model(images)
                    
                    # Calculate validation loss
                    val_loss = criterion(outputs, labels)
                    val_loss_total += val_loss.item()
                    
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    
                    # Clear variables to free memory
                    del images, labels, outputs, predicted

            # Clear GPU cache after validation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Calculate validation accuracy for this epoch
            val_accuracy = 100 * val_correct / val_total
            val_loss_avg = val_loss_total / len(val_loader)
            val_end_time = time.time()
            val_time = val_end_time - val_start_time
            
            print(f"Val Acc: {val_accuracy:.2f}%, Val Loss: {val_loss_avg:.4f}")
            print(f"Val Time: {val_time:.2f}s")

            epoch_time = train_time_spent + val_time
            print(f"Time for this epoch: {epoch_time:.2f}s")

            ### Early stopping and model saving for this fold (SAME AS ORIGINAL)
            if val_accuracy - best_val_accuracy > 0.01:
                print("Model performance improved")
                best_val_accuracy = val_accuracy
                non_update_count = 0

                # Save best model for this fold
                best_model_state = fold_model.module.state_dict() if isinstance(fold_model, nn.DataParallel) else fold_model.state_dict()
                # torch.save(fold_model.state_dict(), f"{best_models_foldername}/{model_name}_fold{fold+1}_best.pth")
                print(f"Best model saved for fold {fold+1}")
            else:
                non_update_count += 1
                print(f"Model performance did not improve for {non_update_count} times")

            # Learning rate increment mechanism (SAME AS ORIGINAL)
            if non_update_count >= early_stopping_patience:
                if lr_increase_count < lr_adjustment_patience:
                    if lr_adjustment_mode == 'increase':
                        current_lr += lr_adjustment_rate
                    elif lr_adjustment_mode == 'decrease':
                        current_lr -= lr_adjustment_rate
                    # Update optimizer learning rate
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr
                    lr_increase_count += 1
                    print(f"Learning rate increased to {current_lr}")
                    non_update_count = 0  # Reset counter after LR increase
                else:
                    print(f"Early stopping at epoch {epoch+1} for fold {fold+1}")
                    break
            
            if save_result:
                with open(f"{result_foldername}/{model_name}_cv_result.csv", "a") as f:
                    f.write(f"{fold+1},{epoch+1},{avg_loss:.4f},{val_loss_avg:.4f},{train_accuracy:.2f},{val_accuracy:.2f},{epoch_time:.2f},{current_lr}\n")
        
        # Store fold results
        fold_results.append({
            'fold': fold + 1,
            'best_val_accuracy': best_val_accuracy,
            'model_path': f"{model_name}_fold{fold+1}_best.pth"
        })
        
        print(f"Fold {fold + 1} completed. Best Val Accuracy: {best_val_accuracy:.2f}%")
    
    # Calculate cross-validation statistics
    val_accuracies = [result['best_val_accuracy'] for result in fold_results]
    mean_accuracy = np.mean(val_accuracies)
    std_accuracy = np.std(val_accuracies)
    
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    for result in fold_results:
        print(f"Fold {result['fold']}: {result['best_val_accuracy']:.2f}%")
    print(f"Mean CV Accuracy: {mean_accuracy:.2f} ± {std_accuracy:.2f}%")
    print(f"{'='*60}")
    
    # Save CV summary
    if save_result:
        cv_summary = pd.DataFrame(fold_results)
        cv_summary.to_csv(f"{result_foldername}/{model_name}_cv_summary.csv", index=False)
        
        with open(f"{result_foldername}/{model_name}_cv_stats.txt", "w") as f:
            f.write(f"{n_folds}-Fold Cross-Validation Results\n")
            f.write(f"Mean Accuracy: {mean_accuracy:.2f}%\n")
            f.write(f"Standard Deviation: {std_accuracy:.2f}%\n")
            f.write(f"Min Accuracy: {min(val_accuracies):.2f}%\n")
            f.write(f"Max Accuracy: {max(val_accuracies):.2f}%\n")
    
    torch.save(best_model_state, f"{best_models_foldername}/{model_name}_best.pth")
    # Return the best model path
    return f"{best_models_foldername}/{model_name}_best.pth"
    


def test_vgg16(model_path, 
               experiment_name, 
               test_data, 
               device, 
               positive_class: int = 1, 
               save_result=False, 
               verbose=False,
               result_foldername='results'
               ):
    """
    Loads a trained VGG16 model from a .pth file and tests it on the test set.
    Additionally returns image-index lists for TP, FP, TN, FN (binary-class assumption).
    Args:
        model_path: Path to saved model
        test_data: Dataset to evaluate
        device: torch device
        num_classes: total number of classes in the model
        positive_class: which class id is treated as the "positive" class for TP/FP definitions
    Returns
        test_accuracy, dict with keys 'tp','fp','tn','fn' mapping to lists of sample indices
    """

    if not os.path.exists(f"{result_foldername}/classification_reports"):
        os.makedirs(f"{result_foldername}/classification_reports")

    if not os.path.exists(f"{result_foldername}/confusion_matrices"):
        os.makedirs(f"{result_foldername}/confusion_matrices")

    model = VGG16(num_classes=test_data.get_num_classes()).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    criterion = nn.CrossEntropyLoss()
    test_loader = DataLoader(test_data, shuffle=False)
    model.eval()

    test_correct = 0
    test_total = 0
    loss_total = 0

    y_img_ids = []
    y_true = []
    y_pred = []

    # Confusion matrix image mapping: (true_class, predicted_class) -> list of image details
    confusion_matrix_for_each_individual = {}

    print("Starting model testing...")
    label_to_idx = test_data.label_name_idx
    print(f"Label names and indices: {label_to_idx}")
    
    # Create reverse mapping from index to name
    idx_to_label = {idx: name for name, idx in label_to_idx.items()}
    print(f"Index to label mapping: {idx_to_label}")

    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            image_id = test_data.get_image_id(idx)
            image_label = test_data.get_image_label(idx)
            image_url = test_data.get_image_filename(idx)
            image_caption = test_data.get_image_caption(idx)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            loss_total += criterion(outputs, labels).item()
            # Store raw integer labels for confusion matrix
            y_true.append(labels.item())
            y_pred.append(predicted.item())
            y_img_ids.append(image_id)

            # Store detailed information for confusion matrix visualization
            cell_key = f"{labels.item()}_{predicted.item()}"
            if cell_key not in confusion_matrix_for_each_individual:
                confusion_matrix_for_each_individual[cell_key] = []
            
            # Store comprehensive image information
            image_info = {
                'image_id': image_id,
                'image_url': image_url,
                'image_caption': image_caption,
                'true_label': idx_to_label[labels.item()],
                'pred_label': idx_to_label[predicted.item()],
                'correct': labels.item() == predicted.item()
            }
            confusion_matrix_for_each_individual[cell_key].append(image_id)

            if verbose:
                print(f"Image ID: {image_id}, Image Label: {image_label}, Image URL: {image_url}, Image Caption: {image_caption}")
                print(f"Predicted: {idx_to_label[predicted.item()]}, True: {idx_to_label[labels.item()]}")

            # Free memory
            del images, labels, outputs, predicted

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    test_accuracy = 100 * test_correct / test_total
    test_loss = loss_total / test_total
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

    print(f"\n{'='*60}")
    print("TEST RESULTS")
    print(f"{'='*60}")
    print(f"Test Accuracy: {test_accuracy:.2f}%  (Total samples: {test_total})")
    print(f"Test Loss: {test_loss:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=list(label_to_idx.values()), target_names=list(label_to_idx.keys())))

    print("\nConfusion Matrix:")
    print(conf_matrix)

    test_result = {
        'experiment_name': experiment_name,
        'pth_filepath': model_path,
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label,
        'test_accuracy': test_accuracy,
        'loss': test_loss,
        'confusion_matrix_images': confusion_matrix_for_each_individual,
        'individual_prediction_results': individual_prediction_results
    }

    if save_result:
        with open(f"{result_foldername}/{experiment_name}_test_result.json", "w") as f:
            json.dump(test_result, f)

        # Save classification report with proper labels
        df_classification_report = pd.DataFrame(classi_report).T
        df_classification_report.index.name = 'Class'
        df_classification_report.to_csv(f"{result_foldername}/classification_reports/{experiment_name}_classification_report.csv")

        # Save confusion matrix with proper labels
        df_confusion_matrix = pd.DataFrame(
            conf_matrix,
            index=label_names,
            columns=label_names
        )
        df_confusion_matrix.index.name = 'True Label'
        df_confusion_matrix.columns.name = 'Predicted Label'
        df_confusion_matrix.to_csv(f"{result_foldername}/confusion_matrices/{experiment_name}_confusion_matrix.csv")

    return test_result


















