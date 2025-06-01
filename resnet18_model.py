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
from MSCOCO_preprocessing import *
import pandas as pd
from torch.utils.data import random_split, Subset
from sklearn.metrics import classification_report, confusion_matrix

import torch.nn as nn

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
    
def train_resnet18(model,
                   train_data, 
                   val_data, 
                   num_epochs, 
                   learning_rate, 
                   model_name, 
                   device, 
                   save_result=False):
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    # Initialize variables for early stopping and learning rate adjustment
    best_accuracy = 0.0
    non_update_count = 0
    lr_increase_count = 0

    # Create results directory if saving results
    if save_result:
        import os
        os.makedirs("results", exist_ok=True)
        # Write header to CSV file
        with open(f"results/{model_name}_result.csv", "w") as f:
            f.write("epoch,loss,val_accuracy,test_accuracy,time_spent\n")

    for epoch in range(num_epochs):


        print(f"Epoch {epoch+1} of {num_epochs}")
        model.train()
        running_loss = 0.0
        epoch_loss = 0.0

        # Training loop
        start_time = time.time()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            epoch_loss = loss.item()  # Keep track of last batch loss

        end_time = time.time()
        time_spent = end_time - start_time
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}, Time: {time_spent:.2f} seconds")

        # Validation
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            correct = 0
            total = 0
            start_time = time.time()
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            end_time = time.time()
            val_time = end_time - start_time
            val_accuracy = 100 * correct / total
            print(f"Accuracy of the model on the {total} validation images: {val_accuracy:.2f}%")
            print(f"Validation Time: {val_time:.2f} seconds")

        # Testing
        with torch.no_grad():
            correct = 0
            total = 0
            start_time = time.time()
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs
            end_time = time.time()
            test_time = end_time - start_time
            test_accuracy = 100 * correct / total
            print(f"Accuracy of the model on the {total} test images: {test_accuracy:.2f}%")
            print(f"Testing Time: {test_time:.2f} seconds")

        # Early stopping and learning rate adjustment logic
        if val_accuracy - best_accuracy > 0.01:
            best_accuracy = val_accuracy
            non_update_count = 0
            # Save best model based on validation accuracy
            torch.save(model.state_dict(), f"{model_name}_best.pth")
            print(f"New best validation accuracy: {best_accuracy:.2f}%. Model saved.")
        else:
            non_update_count += 1

        # Increase learning rate if validation accuracy is not improving
        if non_update_count >= 10:
            if lr_increase_count < 5:
                learning_rate += 0.0001
                # Update optimizer with new learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate
                lr_increase_count += 1
                non_update_count = 0  # Reset counter after LR increase
                print(f"Learning rate increased to {learning_rate:.6f}")
            else:
                print(f"Early stopping at epoch {epoch+1} due to no improvement in validation accuracy")
                break

        if save_result:
            with open(f"results/{model_name}_result.csv", "a") as f:
                f.write(f"{epoch+1},{avg_loss:.6f},{val_accuracy:.2f},{test_accuracy:.2f},{time_spent:.2f}\n")
    
    print("Training completed!")
    return model
        
if __name__ == "__main__":
    num_epochs = 200
    learning_rate1 = 0.0001
    learning_rate2 = 0.001
    batch_size = 32
    classes = ('dog', 'cat', 'zebra', 'giraffe', 'horse', 'pizza', 'cup', 'truck', 'train', 'airplane', \
               'motorcycle', 'toaster', 'oven', 'cake', 'skateboard','skis', 'bicycle', 'bus', 'car', 'chair')
    
    num_instances = 300
    test_size = 0.2
    model_name1 = "resnet18_baseline-10classes-0.0001lr-testrun"
    model_name2 = "resnet18_baseline-10classes-0.001lr-testrun"

    # Initialize model  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(classes)
    model1 = ResNet18(num_classes=num_classes).to(device)
    model2 = ResNet18(num_classes=num_classes).to(device)

    # Load data
    train_data, test_data = prepare_data(*classes, 
                                     transform="resnet18", target_transform="integer", num_instances=num_instances, test_size=test_size)
                                     
    # Split the training data into train and validation sets using the first fold
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    train_indices, val_indices = next(iter(kfold.split(train_data)))
    train_subset = Subset(train_data, train_indices)
    val_subset = Subset(train_data, val_indices)
    
    # Adjust batch sizes
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    print(f"Data Loading Complete!\nTrain Data: {len(train_subset)}, Validation Data: {len(val_subset)}, Test Data: {len(test_data)}")

    # Train the model
    train_resnet18(model1, train_loader, val_loader, test_loader, num_epochs, learning_rate1, model_name1, device, save_result=True)
    train_resnet18(model2, train_loader, val_loader, test_loader, num_epochs, learning_rate2, model_name2, device, save_result=True)


def test_resnet18(model_path, test_data, device, num_classes, positive_class: int = 1):
    """ Test ResNet18 model and return accuracy & TP/FP/TN/FN lists (binary assumption) """
    model = ResNet18(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    test_loader = DataLoader(test_data, batch_size=4, shuffle=False)
    model.eval()

    test_correct = 0
    test_total = 0
    y_true, y_pred = [], []
    tp_indices, fp_indices, tn_indices, fn_indices = [], [], [], []
    sample_idx = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

            preds_np = predicted.cpu().numpy()
            labels_np = labels.cpu().numpy()
            for p, t in zip(preds_np, labels_np):
                if t == positive_class and p == positive_class:
                    tp_indices.append(sample_idx)
                elif t != positive_class and p != positive_class and p == t:
                    tn_indices.append(sample_idx)
                elif t != positive_class and p == positive_class:
                    fp_indices.append(sample_idx)
                elif t == positive_class and p != positive_class:
                    fn_indices.append(sample_idx)
                y_true.append(t)
                y_pred.append(p)
                sample_idx += 1
            del images, labels, outputs, predicted
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    acc = 100 * test_correct / test_total
    print(f"Test Accuracy: {acc:.2f}% | TP {len(tp_indices)} | FP {len(fp_indices)} | TN {len(tn_indices)} | FN {len(fn_indices)}")
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    return acc, {'tp': tp_indices, 'fp': fp_indices, 'tn': tn_indices, 'fn': fn_indices}

# Placeholder for consistency
def train_resnet18_with_CV(model, train_data, n_folds, num_epochs, learning_rate, model_name, device, batch_size, lr_increment_rate=0.0001, save_result=False, early_stopping_patience=10, lr_increase_patience=5):
    """ Wrapper that reuses train_vgg16_with_CV logic but for ResNet18 by importing from vgg16_model """
    from vgg16_model import train_vgg16_with_CV as _train_cv_core
    return _train_cv_core(model, train_data, n_folds, num_epochs, learning_rate, model_name, device, batch_size, lr_increment_rate, save_result, early_stopping_patience, lr_increase_patience)
