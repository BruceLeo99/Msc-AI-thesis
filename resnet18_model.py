# resnet18 model without prototypes as baseline

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
import time
from MSCOCO_preprocessing_local import *
import pandas as pd
from torch.utils.data import random_split, Subset
from sklearn.metrics import classification_report, confusion_matrix
import os
import json

import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
def train_resnet18(
        train_data,
        val_data,
        model_name, 
        device, 
        num_epochs=10,
        learning_rate=0.0001,
        batch_size=32,
        lr_increment_rate=0.0001,
        save_result=False,
        early_stopping_patience=10,
        lr_increase_patience=5,
        num_workers=0
):
    """
    Train ResNet18 on provided train/val splits (no cross-validation).
    Mirrors the behaviour of train_resnet18 in resnet18_model.py so that the same CLI
    arguments & logging strategy work for both backbones.

    Args:
        train_data: The training data.
        val_data: The validation data.
        model_name: The name of the model.
        device: The device to train on.
        num_epochs: The number of epochs to train for.
        learning_rate: The learning rate to use.
        batch_size: The batch size to use.
        lr_increment_rate: The learning rate increment rate.
        save_result: Whether to save the results.
        early_stopping_patience: The early stopping patience.
        lr_increase_patience: The learning rate increase patience.
        num_workers: Number of workers for DataLoader

    Returns the filepath of the best model saved on validation accuracy.
    """

    if not os.path.exists("best_models"):
        os.makedirs("best_models")
    
    if not os.path.exists("results"):
        os.makedirs("results")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    model = ResNet18(num_classes=train_data.get_num_classes()).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = torch.nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if num_workers == 0:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, shuffle=False)
    else:
        train_loader = DataLoader(
            train_data, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_data, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    if save_result:
        with open(f"results/{model_name}_result.csv", "w") as f:
             f.write("Epoch,Training Loss,Validation Loss,Training Accuracy,Validation Accuracy,Time,Learning Rate\n")

    best_accuracy = 0.0
    current_lr = learning_rate
    non_update_count = 0
    lr_increase_count = 0

    print("Starting Training – ResNet18")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        ### Training Phase
        train_start_time = time.time()
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            del images, labels, outputs, predicted
            if i % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        train_acc = 100 * train_correct / train_total
        avg_loss = running_loss / len(train_loader)
        train_time_spent = time.time() - train_start_time
        print(f"Epoch {epoch+1}, Cross-Entropy Loss: {loss.item()}, Training Accuracy: {train_acc:.2f}%, Time: {train_time_spent:.2f} seconds")

        ### Validation Phase
        val_start_time = time.time()
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                
                # Calculate validation loss
                val_loss = criterion(outputs, labels)
                val_loss_total += val_loss.item()
                
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                del images, labels, outputs, predicted
        val_acc = 100 * val_correct / val_total
        val_loss_avg = val_loss_total / len(val_loader)
        val_time_spent = time.time() - val_start_time
        print(f"Epoch {epoch+1}, Validation Accuracy: {val_acc:.2f}%, Time: {val_time_spent:.2f} seconds")
        print(f"Validation Loss: {val_loss_avg:.4f}")

        # Early stopping & LR scheduling 
        if val_acc - best_accuracy > 0.01:
            print("Model performance improved")
            best_accuracy = val_acc
            non_update_count = 0
            torch.save(model.state_dict(), f"best_models/{model_name}_best.pth")
            print(f"Saved best model to best_models/{model_name}_best.pth")
        else:
            non_update_count += 1
            print(f"Model performance did not improve for {non_update_count} times")

        # Increase learning rate if validation accuracy is not improving
        # Increase learning rate every 10 epochs 5 times in total. 
        # After 5 times, if the model still does not improve, stop training.
        if non_update_count >= early_stopping_patience:
            if lr_increase_count < lr_increase_patience:
                learning_rate += lr_increment_rate
                current_lr = learning_rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate
                lr_increase_count += 1
                non_update_count = 0
                print(f"Learning rate increased to {learning_rate}")
            else:
                print(f"Early stopping at epoch {epoch+1} due to no improvement in validation accuracy after increasing learning rate {lr_increase_patience} times")
                break

        # Calculate time spent for one epoch and save the result
        time_epoch_spent = train_time_spent + val_time_spent
        if save_result:
            with open(f"results/{model_name}_result.csv", "a") as f:
                f.write(f"{epoch+1},{avg_loss:.4f},{val_loss_avg:.4f},{train_acc:.2f},{val_acc:.2f},{time_epoch_spent:.2f},{current_lr}\n")

    # End of training
    print("Training complete!")
    return f"best_models/{model_name}_best.pth"


def train_resnet18_with_CV(
        train_data,
        n_folds,
        num_epochs,
        learning_rate,
        model_name,
        device,
        batch_size,
        lr_increment_rate=0.0001,
        save_result=False,
        early_stopping_patience=10,
        lr_increase_patience=5,
        random_state=42,
        num_workers=0
):
    """
    Trains a ResNet18 model with K-Fold Cross-Validation on the training set and validates it on the validation set.
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
        lr_increment_rate: Learning rate increment rate
        early_stopping_patience: Number of epochs to wait before early stopping
        lr_increase_patience: Number of epochs to wait before increasing learning rate
        random_state: Random state for cross-validation. For reproducibility, set 42 by default.
        num_workers: Number of workers for DataLoader
    """

    if not os.path.exists(f"best_models"):
        os.makedirs(f"best_models")
    
    if not os.path.exists(f"results"):
        os.makedirs(f"results")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Starting Training with {n_folds}-fold Cross-Validation.\nModel: ResNet18 baseline")

    if save_result:
        with open(f"results/{model_name}_cv_result.csv", "w") as f:
            f.write("Fold,Epoch,Training Loss,Validation Loss,Training Accuracy,Validation Accuracy,Time,Learning Rate\n")

    
    # Split data into K-fold
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_data)):
        print(f"\n{'='*60}\nStart training for FOLD {fold+1}/{n_folds}\n{'='*60}")

        # Initialize model for each fold
        fold_model = ResNet18(num_classes=train_data.get_num_classes()).to(device)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for training")
            fold_model = torch.nn.DataParallel(fold_model)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(fold_model.parameters(), lr=learning_rate)

        if num_workers == 0:
            train_loader = DataLoader(train_idx, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_idx, shuffle=False)
        else:
            train_loader = DataLoader(
                train_idx, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True
            )
            val_loader = DataLoader(
                val_idx, 
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )

        best_val_accuracy = 0.0
        current_lr = learning_rate
        non_update = 0
        lr_inc_count = 0

        for epoch in range(num_epochs):
            print(f"Fold {fold+1} — Epoch {epoch+1}/{num_epochs}")

            ### Training Phase
            train_start_time = time.time()
            fold_model.train()
            epoch_loss = 0.0
            train_correct = 0
            train_total = 0

            #Backpropagation and minibatch SGD
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = fold_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Calculate loss and accuracy
                epoch_loss += loss.item()
                _, pred = outputs.max(1)
                train_total += labels.size(0)
                train_correct += pred.eq(labels).sum().item()

                # Clear GPU cache periodically during training
                if i % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Clear variables to free memory
                del images, labels, outputs, pred   

            # Calculate training accuracy and loss
            train_acc = 100 * train_correct / train_total
            avg_loss = epoch_loss / len(train_loader)
            train_time_spent = time.time() - train_start_time

            # Clear GPU cache after training phase
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"Train Acc: {train_acc:.2f}%, Loss: {avg_loss:.4f}")
            print(f"Train Time: {train_time_spent:.2f}s")

            ### Validation Phase
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
                    
                    _, pred = outputs.max(1)

                    # Calculate validation accuracy and loss
                    val_total += labels.size(0)
                    val_correct += pred.eq(labels).sum().item()

                    # Clear variables to free memory
                    del images, labels, outputs, pred

            # Clear GPU cache after validation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            val_acc = 100 * val_correct / val_total
            val_loss_avg = val_loss_total / len(val_loader)
            val_time_spent = time.time() - val_start_time
            print(f"Val Acc: {val_acc:.2f}%, Val Loss: {val_loss_avg:.4f}")
            print(f"Val Time: {val_time_spent:.2f}s")

            ### Early stop & LR schedule
            if val_acc - best_val_accuracy > 0.01:
                print("Model improved")
                best_val_accuracy = val_acc
                non_update = 0
                torch.save(fold_model.state_dict(), f"best_models/{model_name}_fold{fold+1}_best.pth")
                print(f"Saved best model to best_models/{model_name}_fold{fold+1}_best.pth")
            else:
                non_update += 1
                print(f"Model performance did not improve for {non_update} times")

            if non_update >= early_stopping_patience:
                if lr_inc_count < lr_increase_patience:
                    learning_rate += lr_increment_rate
                    current_lr = learning_rate
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = learning_rate
                    lr_inc_count += 1
                    non_update = 0
                    print(f"Learning rate increased to {learning_rate}")
                else:
                    print(f"Early stopping at epoch {epoch+1} due to no improvement in validation accuracy after increasing learning rate {lr_increase_patience} times")
                    break

            # Calculate time spent for one epoch and save the result
            time_epoch_spent = train_time_spent + val_time_spent
            if save_result:
                with open(f"results/{model_name}_cv_result.csv", "a") as f:
                    f.write(f"{fold+1},{epoch+1},{avg_loss:.4f},{val_loss_avg:.4f},{train_acc:.2f},{val_acc:.2f},{time_epoch_spent:.2f},{current_lr}\n")

        fold_results.append({'fold': fold+1, 'best_val_accuracy': best_val_accuracy, 'model_path': f"best_models/{model_name}_fold{fold+1}_best.pth"})
        print(f"Fold {fold+1} best Val Acc: {best_val_accuracy:.2f}%")

    # CV stats and save CV summary
    val_accs = [fr['best_val_accuracy'] for fr in fold_results]
    print(f"\n{'='*60}\nCROSS-VALIDATION RESULTS\n{'='*60}")
    for fr in fold_results:
        print(f"Fold {fr['fold']}: {fr['best_val_accuracy']:.2f}%")
    print(f"Mean {np.mean(val_accs):.2f}% ± {np.std(val_accs):.2f}%")

    if save_result:
        pd.DataFrame(fold_results).to_csv(f"results/{model_name}_cv_summary.csv", index=False)

        with open(f"results/{model_name}_cv_stats.txt", "w") as f:
            f.write(f"{n_folds}-Fold Cross-Validation Results\n")
            f.write(f"Mean Accuracy: {np.mean(val_accs):.2f}%\n")
            f.write(f"Standard Deviation: {np.std(val_accs):.2f}%\n")
            f.write(f"Min Accuracy: {min(val_accs):.2f}%\n")
            f.write(f"Max Accuracy: {max(val_accs):.2f}%\n")

    return "best_models/{model_name}_fold{fold+1}_best.pth"


def test_resnet18(model_path, 
                  experiment_name, 
                  test_data, 
                  device, 
                  num_classes, 
                  positive_class: int = 1, 
                  save_result=False, 
                  verbose=False,
                  ):
    """
    Loads a trained ResNet18 model from a .pth file and tests it on the test set.
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

    if not os.path.exists(f"results/classification_reports"):
        os.makedirs(f"results/classification_reports")

    if not os.path.exists(f"results/confusion_matrices"):
        os.makedirs(f"results/confusion_matrices")

    model = ResNet18(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    test_loader = DataLoader(test_data, shuffle=False)
    model.eval()

    test_correct = 0
    test_total = 0

    y_img_ids = []
    y_true = []
    y_pred = []

    
    # Confusion matrix image mapping: (true_class, predicted_class) -> list of image details
    confusion_matrix_for_each_individual = {}

    print("Starting model testing...")
    label_names_idx = test_data.get_dataset_labels()
    print(f"Label names and indices: {label_names_idx}")
    
    # Create reverse mapping from index to name
    idx_to_label = {idx: name for name, idx in label_names_idx.items()}
    print(f"Index to label mapping: {idx_to_label}")

    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            image_id = test_data.get_image_id(idx)
            image_label = test_data.get_image_label(idx)
            image_url = test_data.get_image_url(idx)
            image_caption = test_data.get_image_caption(idx)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

            for p, t in zip(predicted, labels):
                # Convert tensors to integers
                p_int = p.item()
                t_int = t.item()
                
                # Get class names
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
                    'correct': true_class == pred_class
                }
                confusion_matrix_for_each_individual[cell_key].append(image_info)

                if verbose:
                    print(f"Image ID: {image_id}, Image Label: {image_label}, Image URL: {image_url}, Image Caption: {image_caption}")
                    print(f"Predicted: {pred_class}, True: {true_class}")

            # Free memory
            del images, labels, outputs, predicted

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    test_accuracy = 100 * test_correct / test_total

    individual_prediction_results = dict()

    for test_image_id, test_label, test_pred in zip(y_img_ids, y_true, y_pred):
        individual_prediction_results[test_image_id] = {
            'true_label': test_label,
            'pred_label': test_pred
        }

    print(f"\n{'='*60}")
    print("TEST RESULTS")
    print(f"{'='*60}")
    print(f"Test Accuracy: {test_accuracy:.2f}%  (Total samples: {test_total})")

    classi_report = classification_report(y_true, y_pred, target_names=list(idx_to_label.values()), output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred)

    print("\nClassification Report:")
    print(classi_report)

    print("\nConfusion Matrix:")
    print(conf_matrix)


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
        with open(f"results/{experiment_name}_test_result.json", "w") as f:
            json.dump(test_result, f)
            
        df_classification_report = pd.DataFrame(classi_report).T
        # The classification report already has proper row names, just ensure they're clean
        df_classification_report.index.name = 'Class'
        df_classification_report.to_csv(f"results/classification_reports/{experiment_name}_classification_report.csv")

        # For confusion matrix, add proper labels
        df_confusion_matrix = pd.DataFrame(conf_matrix)
        # Set row and column labels to class names
        class_names = list(idx_to_label.values())
        df_confusion_matrix.index = class_names
        df_confusion_matrix.columns = class_names
        df_confusion_matrix.index.name = 'True Label'
        df_confusion_matrix.columns.name = 'Predicted Label'
        df_confusion_matrix.to_csv(f"results/confusion_matrices/{experiment_name}_confusion_matrix.csv")

    return test_result
        
if __name__ == "__main__":
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Memory optimization settings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Enable memory efficient attention if available
        torch.backends.cudnn.benchmark = True
        # Set memory fraction to prevent over-allocation
        torch.cuda.set_per_process_memory_fraction(0.9)
    
    num_epochs = 100
    num_folds = 3
    learning_rate = 0.0001
    lr_increment_rate = 0.0001
    batch_size = 4
    validation_size = 0.15
    prechosen_categories_csv_path = 'chosen_categories_3_10.csv'  # Use the file path string
    early_stopping_patience = 10
    lr_increase_patience = 5

    experiment_name1 = "resnet18_newdata-10classes-0.0001lr-testrun"
    experiment_name2 = "resnet18_newdata-10classes-0.0001lr-CV-testrun"

    prechosen_categories_csv = pd.read_csv(prechosen_categories_csv_path)  # Read the CSV for getting class names

    classes = prechosen_categories_csv['Category Name'].tolist()
    model_name1 = experiment_name1
    model_name2 = experiment_name2

    # Initialize model  
    num_classes = len(classes)
    model1 = ResNet18(num_classes=num_classes).to(device)
    model2 = ResNet18(num_classes=num_classes).to(device)
    # print(model)

    ### Load data - manually

    train_data, val_data = prepare_data_manually(*classes, 
                                        num_instances=10, 
                                        split=True, 
                                        split_size=0.15,
                                        transform="resnet18", 
                                        target_transform="integer")

    test_data = prepare_data_manually(*classes, 
                                        num_instances=10, 
                                        for_test=True,
                                        transform="resnet18", 
                                        target_transform="integer")
    
    ### Load data - pass the file path string, not the DataFrame
    # train_data = prepare_data_from_preselected_categories(
    #     prechosen_categories_csv_path, 
    #     'train', 
    #     split_val=False,
    #     transform="resnet18",  # This will apply resnet18 transforms
    #     target_transform="integer"  # This will convert labels to integers
    # )

    # test_data = prepare_data_from_preselected_categories(
    #     prechosen_categories_csv_path, 
    #     'test',
    #     transform="resnet18",  # This will apply resnet18 transforms
    #     target_transform="integer"  # This will convert labels to integers
    # )


    # Clean data
    train_data, val_data, test_data = eliminate_leaked_data(experiment_name1, train_data, val_data, test_data, verbose=True, save_result=True)


    ### Train normally, without CV
    best_model_path1 = train_resnet18(model1, 
                                   train_data, 
                                   val_data, 
                                   model_name1, 
                                   device, 
                                   num_epochs=num_epochs, 
                                   learning_rate=learning_rate,
                                   lr_increment_rate=lr_increment_rate,
                                   batch_size=batch_size, 
                                   early_stopping_patience=early_stopping_patience,
                                   lr_increase_patience=lr_increase_patience)
    
    test_resnet18(best_model_path1, 
               experiment_name1, 
               test_data, 
               device, 
               num_classes, 
               save_result=True,
               verbose=False)


    ### Train with CV
    # best_model_path2 = train_resnet18_with_CV(model2, 
    #                                        train_data, 
    #                                        num_folds, 
    #                                        num_epochs, 
    #                                        learning_rate, 
    #                                        model_name2, 
    #                                        device, 
    #                                        batch_size, 
    #                                        lr_increment_rate=lr_increment_rate,
    #                                        save_result=True,
    #                                        early_stopping_patience=early_stopping_patience,
    #                                        lr_increase_patience=lr_increase_patience)
    
    # test_resnet18(best_model_path2, test_data, device, num_classes)
