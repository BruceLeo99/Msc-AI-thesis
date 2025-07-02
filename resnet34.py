# resnet34 model without prototypes as baseline

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pdb
import os
import time
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from collections import OrderedDict
import json
import pandas as pd

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out       

class ResNet34(nn.Module):
    def __init__(self, num_classes=101):
        super().__init__()
        self.inplanes = 64  # Initialize inplanes
        self.conv_block = nn.Sequential(nn.Conv2d(3,64,kernel_size=7, stride=2, padding=3, bias=False), # output 112,112
                       nn.BatchNorm2d(64),
                       nn.ReLU(inplace=True),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) # 56,56
        self.conv1 = self.conv_block
        self.layer1 = self._make_layer(BasicBlock, 64, 3)
        self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)  # Add missing FC layer

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv_block(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
def train_resnet34(
        train_data,
        val_data,
        model_name, 
        device, 
        num_epochs=10,
        learning_rate=0.0001,
        batch_size=32,
        lr_adjustment_rate=0.0001,
        lr_adjustment_mode='increase',
        save_result=False,
        early_stopping_patience=5,
        lr_adjustment_patience=3,
        num_workers=4,
        result_foldername='results'):



    if not os.path.exists("/var/scratch/yyg760/best_models"):
        os.makedirs("/var/scratch/yyg760/best_models")

    best_models_foldername = f"/var/scratch/yyg760/best_models"
    
    if not os.path.exists(result_foldername):
        os.makedirs(result_foldername)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    num_classes = train_data.get_num_classes()

    # Load model
    model = ResNet34(num_classes=num_classes).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = torch.nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load Data
    if num_workers == 0:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
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
        with open(f"{result_foldername}/{model_name}_result.csv", "w") as f:
             f.write("Epoch,Training Loss,Validation Loss,Training Accuracy,Validation Accuracy,Time,Learning Rate\n")

    best_accuracy = 0.0
    current_lr = learning_rate
    epochs_without_improvement = 0
    best_epoch = 0
    lr_adjustments_made = 0
    best_model_state = None

    print("Starting Training – ResNet34")
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
        if val_acc > best_accuracy:
            print("Model performance improved")
            best_accuracy = val_acc
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            # Save the best model immediately
            best_model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            # torch.save(best_model_state, f"{best_models_foldername}/{model_name}_best.pth")
            print(f"Best model saved for epoch {epoch+1} with accuracy {val_acc:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"Model performance did not improve for {epochs_without_improvement} times")

        # Increase learning rate if validation accuracy is not improving
        # Increase learning rate every 10 epochs 5 times in total. 
        # After 5 times, if the model still does not improve, stop training.
        if epochs_without_improvement >= early_stopping_patience:
            if lr_adjustments_made < lr_adjustment_patience:
                if lr_adjustment_mode == 'increase':
                    learning_rate += lr_adjustment_rate
                elif lr_adjustment_mode == 'decrease':
                    learning_rate -= lr_adjustment_rate
                current_lr = learning_rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate
                lr_adjustments_made += 1
                epochs_without_improvement = 0
                print(f"Learning rate increased to {learning_rate}")
            else:
                print(f"Early stopping at epoch {epoch+1} due to no improvement in validation accuracy after increasing learning rate {lr_adjustment_patience} times")
                break

        # Calculate time spent for one epoch and save the result
        time_epoch_spent = train_time_spent + val_time_spent
        if save_result:
            with open(f"{result_foldername}/{model_name}_result.csv", "a") as f:
                f.write(f"{epoch+1},{avg_loss:.4f},{val_loss_avg:.4f},{train_acc:.2f},{val_acc:.2f},{time_epoch_spent:.2f},{current_lr}\n")

    # End of training
    print("Training complete!")
    print(f"Best model saved from epoch {best_epoch} with validation accuracy {best_accuracy:.4f}")

    if best_model_state is not None:
        torch.save(best_model_state, f"{best_models_foldername}/{model_name}_best.pth")
        print(f"Final best model confirmed saved: {model_name}_best.pth")

    return f"{best_models_foldername}/{model_name}_best.pth"   

def test_resnet34(model_path, 
                  experiment_name, 
                  test_data, 
                  device, 
                  positive_class: int = 1, 
                  save_result=False, 
                  verbose=False,
                  result_foldername='results'
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
    if not os.path.exists(result_foldername):
        os.makedirs(result_foldername)

    if not os.path.exists(f"{result_foldername}/classification_reports"):
        os.makedirs(f"{result_foldername}/classification_reports")

    if not os.path.exists(f"{result_foldername}/confusion_matrices"):
        os.makedirs(f"{result_foldername}/confusion_matrices")

    model = ResNet34(num_classes=test_data.get_num_classes()).to(device)
    state_dict = torch.load(model_path, map_location=device)

    # Remove 'module.' prefix if it exists
    module_replaced = False
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
           new_state_dict[k.replace('module.', '')] = v
           module_replaced = True
            
        if k.startswith('features.'):
            new_k = k[len('features.'):]
        else:
            new_k = k
        new_state_dict[new_k] = v
    

    if module_replaced:
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    test_loader = DataLoader(test_data)
    model.eval()

    test_correct = 0
    test_total = 0

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

    classi_report = classification_report(y_true, y_pred, labels=list(label_to_idx.values()), target_names=list(label_to_idx.keys()), output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred)

    if verbose:
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, labels=list(label_to_idx.values()), target_names=list(label_to_idx.keys())))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))


    test_result = {
        'experiment_name': experiment_name,
        'pth_filepath': model_path,
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label,
        'test_accuracy': test_accuracy,
        'confusion_matrix_for_each_individual': confusion_matrix_for_each_individual,
        'individual_prediction_results': individual_prediction_results
    }

    if save_result:
        with open(f"{result_foldername}/{experiment_name}_test_result.json", "w") as f:
            json.dump(test_result, f)
            
        df_classification_report = pd.DataFrame(classi_report).T
        # The classification report already has proper row names, just ensure they're clean
        df_classification_report.index.name = 'Class'
        df_classification_report.to_csv(f"{result_foldername}/classification_reports/{experiment_name}_classification_report.csv")

        # For confusion matrix, add proper labels
        df_confusion_matrix = pd.DataFrame(conf_matrix)
        # Set row and column labels to class names
        class_names = list(idx_to_label.values())
        df_confusion_matrix.index = class_names
        df_confusion_matrix.columns = class_names
        df_confusion_matrix.index.name = 'True Label'
        df_confusion_matrix.columns.name = 'Predicted Label'
        df_confusion_matrix.to_csv(f"{result_foldername}/confusion_matrices/{experiment_name}_confusion_matrix.csv")

    return test_result



