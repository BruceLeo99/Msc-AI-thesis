# vgg16 model without prototypes as baseline

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CyclicLR
from sklearn.model_selection import train_test_split, KFold
import time
import os
from MSCOCO_preprocessing import *
import pandas as pd
from torch.utils.data import random_split, Subset
from sklearn.metrics import classification_report, confusion_matrix



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
        num_epochs,
        learning_rate,
        model_name,
        device,
        use_CV = True,
        val_loader = None,
        train_dataset = None,
        save_result = False
):
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Clear GPU cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if use_CV:
        print("Starting 5-Fold Cross-Validation Training")
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        fold_results = []
        
        for fold, (train_indices, val_indices) in enumerate(kfold.split(train_dataset)):
            print(f"\n{'='*60}")
            print(f"FOLD {fold + 1}/5")
            print(f"{'='*60}")
            
            # Create data subsets for this fold
            train_subset = Subset(train_dataset, train_indices)
            val_subset = Subset(train_dataset, val_indices)
            
            # Create data loaders for this fold
            fold_train_loader = DataLoader(train_subset, batch_size=train_loader.batch_size, shuffle=True)
            fold_val_loader = DataLoader(val_subset, batch_size=train_loader.batch_size, shuffle=False)
            
            print(f"Fold {fold + 1} - Train: {len(train_subset)}, Val: {len(val_subset)}")
            
            # Reset model for this fold
            model = VGG16(num_classes=model.fc2[0].out_features).to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Training variables for this fold
            best_fold_accuracy = 0
            current_lr = learning_rate
            non_update_count = 0
            lr_increase_count = 0
            
            # Train model for this fold
            for epoch in range(num_epochs):
                print(f"Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}")
                
                # Training phase
                start_time = time.time()
                model.train()
                
                train_correct = 0
                train_total = 0
                epoch_loss = 0
                
                for i, (images, labels) in enumerate(fold_train_loader):
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
                    epoch_loss += loss.item()
                    
                    # Clear variables to free memory
                    del images, labels, outputs, predicted
                    
                    # Clear GPU cache periodically
                    if i % 50 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Calculate training metrics
                train_accuracy = 100 * train_correct / train_total
                avg_loss = epoch_loss / len(fold_train_loader)
                
                # Clear GPU cache after training phase
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                train_time = time.time() - start_time
                
                # Validation phase
                start_time = time.time()
                model.eval()
                
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for images, labels in fold_val_loader:
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = model(images)
                        _, predicted = outputs.max(1)
                        val_total += labels.size(0)
                        val_correct += predicted.eq(labels).sum().item()
                        
                        # Clear variables to free memory
                        del images, labels, outputs, predicted
                
                # Clear GPU cache after validation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                val_accuracy = 100 * val_correct / val_total
                val_time = time.time() - start_time
                
                print(f"  Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%, Loss: {avg_loss:.4f}")
                print(f"  Train Time: {train_time:.2f}s, Val Time: {val_time:.2f}s")
                
                # Early stopping and model saving for this fold
                if val_accuracy - best_fold_accuracy > 0.01:
                    best_fold_accuracy = val_accuracy
                    non_update_count = 0
                    # Save best model for this fold
                    torch.save(model.state_dict(), f"{model_name}_fold{fold+1}_best.pth")
                    print(f"  Best model saved for fold {fold+1}")
                else:
                    non_update_count += 1
                
                # Learning rate adjustment
                if non_update_count >= 10:
                    if lr_increase_count < 5:
                        learning_rate += 0.0001
                        current_lr = learning_rate
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = learning_rate
                        lr_increase_count += 1
                        print(f"  Learning rate increased to {learning_rate}")
                    else:
                        print(f"  Early stopping at epoch {epoch+1} for fold {fold+1}")
                        break
                
                # Save results if requested
                if save_result:
                    with open(f"results/{model_name}_fold{fold+1}_result.csv", "a") as f:
                        if epoch == 0:  # Write header for first epoch
                            f.write("Epoch,Cross-Entropy Loss,Training Accuracy,Validation Accuracy,Time,Learning Rate\n")
                        f.write(f"{epoch+1},{avg_loss:.4f},{train_accuracy:.2f},{val_accuracy:.2f},{train_time:.2f},{current_lr}\n")
            
            # Store fold results
            fold_results.append({
                'fold': fold + 1,
                'best_val_accuracy': best_fold_accuracy,
                'model_path': f"{model_name}_fold{fold+1}_best.pth"
            })
            
            print(f"Fold {fold + 1} completed. Best Val Accuracy: {best_fold_accuracy:.2f}%")
        
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
            cv_summary.to_csv(f"results/{model_name}_cv_summary.csv", index=False)
            
            with open(f"results/{model_name}_cv_stats.txt", "w") as f:
                f.write(f"5-Fold Cross-Validation Results\n")
                f.write(f"Mean Accuracy: {mean_accuracy:.2f}%\n")
                f.write(f"Standard Deviation: {std_accuracy:.2f}%\n")
                f.write(f"Min Accuracy: {min(val_accuracies):.2f}%\n")
                f.write(f"Max Accuracy: {max(val_accuracies):.2f}%\n")
        
        return {
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'fold_results': fold_results
        }
    



    else:
        
        # Original single train/val split training
        print("Starting Single Train/Validation Split Training")
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Train model
        if save_result:
            with open(f"results/{model_name}_result.csv", "w") as f:
                f.write("Epoch,Cross-Entropy Loss,Training Accuracy,Validation Accuracy,Time, Learning Rate\n")

        best_accuracy = 0
        current_lr = learning_rate
        non_update_count = 0
        lr_increase_count = 0

        print("Start Training")
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1} of {num_epochs}")
            
            # Training phase
            start_time = time.time()
            model.train()
            
            train_correct = 0
            train_total = 0
            
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
                
            end_time = time.time()
            time_spent = end_time - start_time
            print(f"Epoch {epoch+1}, Cross-Entropy Loss: {loss.item()}, Training Accuracy: {train_accuracy:.2f}%, Time: {time_spent:.2f} seconds")
            

            # Validation
            model.eval()
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
                    
                    # Clear variables to free memory
                    del images, labels, outputs
                    
                # Clear GPU cache after validation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                end_time = time.time()
                val_time = end_time - start_time
                val_accuracy = 100 * correct / total
                print(f"Accuracy of the model on the {total} validation images: {val_accuracy:.2f}%")
                print(f"Validation Time: {val_time:.2f} seconds")

                # Use validation accuracy for early stopping, then save and dynamically update the best model 
                if val_accuracy - best_accuracy > 0.01:
                    best_accuracy = val_accuracy
                    non_update_count = 0

                    # Save best model based on validation accuracy
                    torch.save(model.state_dict(), f"{model_name}_best.pth")
                    print(f"Model saved to {model_name}_best.pth")

                else:
                    non_update_count += 1

                # Increase learning rate if validation accuracy is not improving
                # Increase learning rate every 10 epochs 5 times in total. 
                # After 5 times, if the model still does not improve, stop training.
                if non_update_count >= 10:
                    if lr_increase_count < 5:
                        learning_rate += 0.0001
                        current_lr = learning_rate  # Update current learning rate
                        # Update optimizer learning rate
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = learning_rate
                        lr_increase_count += 1
                        print(f"Learning rate increased to {learning_rate}")
                    else:
                        print(f"Early stopping at epoch {epoch+1} due to no improvement in validation accuracy")
                        break

            if save_result:
                with open(f"results/{model_name}_result.csv", "a") as f:
                    f.write(f"{epoch+1},{loss.item()},{train_accuracy:.2f},{val_accuracy:.2f},{time_spent:.2f},{current_lr}\n")
        
        # End of training loop
        print("Training Complete!")
        return f"{model_name}_best.pth"

def test_vgg16(model_config, num_classes, test_loader, device):
    
    model = VGG16(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_config))
    model.eval()

    test_correct = 0
    test_total = 0

    # Classification Report 
    y_true = []
    y_pred = []

    print("Starting model testing...")
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
            # Store predictions and true labels for classification report
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            
            # Clear variables to free memory
            del images, labels, outputs, predicted
            
        # Clear GPU cache after testing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    test_accuracy = 100 * test_correct / test_total
    
    print(f"\n{'='*60}")
    print("TEST RESULTS")
    print(f"{'='*60}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Correct Predictions: {test_correct}/{test_total}")
    
    # Generate classification report
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    return test_accuracy

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
    
    num_epochs = 10
    learning_rate1 = 0.0001
    learning_rate2 = 0.001
    batch_size = 4
    validation_size = 0.15
    prechosen_categories_csv_path = 'chosen_categories_3_10.csv'  # Use the file path string
    prechosen_categories_csv = pd.read_csv(prechosen_categories_csv_path)  # Read the CSV for getting class names

    classes = prechosen_categories_csv['Category Name'].tolist()
    model_name1 = "vgg16_newdata-10classes-0.0001lr-testrun"
    model_name2 = "vgg16_newdata-10classes-0.001lr-testrun"

    # Initialize model  
    num_classes = len(classes)
    model1 = VGG16(num_classes=num_classes).to(device)
    model2 = VGG16(num_classes=num_classes).to(device)
    # print(model)

    train_data = prepare_data_manually(*classes, 
                                        num_instances=50, 
                                        split_val=False, 
                                        transform="vgg16", 
                                        target_transform="integer")

    test_data = prepare_data_manually(*classes, 
                                        num_instances=50, 
                                        split_val=False, 
                                        transform="vgg16", 
                                        target_transform="integer")
    
    # # Load data - pass the file path string, not the DataFrame
    # train_data = prepare_data_from_preselected_categories(
    #     prechosen_categories_csv_path, 
    #     'train', 
    #     split_val=False,
    #     transform="vgg16",  # This will apply VGG16 transforms
    #     target_transform="integer"  # This will convert labels to integers
    # )

    # test_data = prepare_data_from_preselected_categories(
    #     prechosen_categories_csv_path, 
    #     'test',
    #     transform="vgg16",  # This will apply VGG16 transforms
    #     target_transform="integer"  # This will convert labels to integers
    # )

    print("Data Summary:")
    print("Training Data:")
    data_summary(train_data, num_examples=5)
    print("Test Data:")
    data_summary(test_data, num_examples=5)

    print("Loading Data to Pytorch DataLoader...")
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    print(f"Data Loading Complete!\nTrain Data: {len(train_data)}, Test Data: {len(test_data)}")
    
    # Train with K-Fold Cross-Validation
    print("Starting K-Fold Cross-Validation Training...")
    cv_results = train_vgg16(model2, 
                            train_loader, 
                            num_epochs, 
                            learning_rate1, 
                            model_name1, 
                            device, 
                            use_CV=True,
                            train_dataset=train_data,
                            save_result=False)

    print(f"\nK-Fold Cross-Validation Results:")
    print(f"Mean Accuracy: {cv_results['mean_accuracy']:.2f} ± {cv_results['std_accuracy']:.2f}%")
    
    # Find the best fold (highest validation accuracy)
    best_fold = max(cv_results['fold_results'], key=lambda x: x['best_val_accuracy'])
    best_model_path = best_fold['model_path']
    
    print(f"\nBest performing fold: Fold {best_fold['fold']} with {best_fold['best_val_accuracy']:.2f}% validation accuracy")
    print(f"Testing best model: {best_model_path}")

    # Test the best model
    test_accuracy = test_vgg16(best_model_path, num_classes, test_loader, device)
    
    print(f"\nFinal Results Summary:")
    print(f"Cross-Validation Mean: {cv_results['mean_accuracy']:.2f} ± {cv_results['std_accuracy']:.2f}%")
    print(f"Best Fold Val Accuracy: {best_fold['best_val_accuracy']:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")


















