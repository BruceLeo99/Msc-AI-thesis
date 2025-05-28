# vgg16 model without prototypes as baseline

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
import time
from MSCOCO_preprocessing import *
import pandas as pd
from torch.utils.data import random_split, Subset


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
        model,
        train_loader,
        val_loader,
        test_loader,
        num_epochs,
        learning_rate,
        model_name,
        device,
        save_result = False
):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    if save_result:
        with open(f"results/{model_name}_result.csv", "w") as f:
            f.write("Epoch,Cross-Entropy Loss,Validation Accuracy,Test Accuracy,Time\n")

    best_accuracy = 0
    non_update_count = 0
    lr_increase_count = 0
    print("Start Training")
    for epoch in range(num_epochs):

        print(f"Epoch {epoch+1} of {num_epochs}")

        start_time = time.time()
        for i, (images, labels) in enumerate(train_loader, 0):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        end_time = time.time()
        time_spent = end_time - start_time
        print(f"Epoch {epoch+1}, Cross-Entropy Loss: {loss.item()}, Time: {time_spent:.2f} seconds")
        time_spent = 0
        

        # Validation
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
            time_spent = 0

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

            # Use validation accuracy for early stopping (not test accuracy!)
            if val_accuracy - best_accuracy > 0.01:
                best_accuracy = val_accuracy
                non_update_count = 0
                # Save best model based on validation accuracy
                torch.save(model.state_dict(), f"{model_name}_best.pth")
            else:
                non_update_count += 1

            # Increase learning rate if validation accuracy is not improving
            # Increase learning rate every 10 epochs 5 times in total. 
            # After 5 times, if the model still does not improve, stop training.
            if non_update_count >= 10:
                if lr_increase_count < 5:
                    learning_rate += 0.0001
                    lr_increase_count += 1
                    print(f"Learning rate increased to {learning_rate}")
                else:
                    print(f"Early stopping at epoch {epoch+1} due to no improvement in validation accuracy")
                    break

        if save_result:
            with open(f"results/{model_name}_result.csv", "a") as f:
                f.write(f"{epoch+1},{loss.item()},{val_accuracy:.2f},{test_accuracy:.2f},{time_spent:.2f}\n")

    # Save model
    torch.save(model.state_dict(), f"{model_name}.pth")
    print(f"Model saved to {model_name}.pth")     
    

if __name__ == "__main__":
    num_epochs = 200
    learning_rate1 = 0.0001
    learning_rate2 = 0.001
    batch_size = 32
    classes = ('dog', 'cat', 'zebra', 'giraffe', 'horse', 'pizza', 'cup', 'truck', 'train', 'airplane')
    num_instances = 300
    test_size = 0.2
    model_name1 = "vgg16_baseline-10classes-0.0001lr-testrun"
    model_name2 = "vgg16_baseline-10classes-0.001lr-testrun"

    # Initialize model  
    num_classes = len(classes)
    model1 = VGG16(num_classes=num_classes).to(device)
    model2 = VGG16(num_classes=num_classes).to(device)
    # print(model)

    # Load data
    train_data, test_data = prepare_data(*classes, 
                                     transform="vgg16", target_transform="integer", num_instances=num_instances, test_size=test_size)

    # Split the training data into train and validation sets using the first fold
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    train_indices, val_indices = next(iter(kfold.split(train_data)))
    train_subset = Subset(train_data, train_indices)
    val_subset = Subset(train_data, val_indices)

    
    # train_data, validation_data = train_test_split(train_data, test_size=0.2, random_state=42)
    print("Loading Data...")
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    print(f"Data Loading Complete!\nTrain Data: {len(train_subset)}, Validation Data: {len(val_subset)}, Test Data: {len(test_data)}")
    
    train_vgg16(model1, train_loader, val_loader, test_loader, num_epochs, learning_rate1, model_name1, device, save_result=False)
    train_vgg16(model2, train_loader, val_loader, test_loader, num_epochs, learning_rate2, model_name2, device, save_result=False)




















