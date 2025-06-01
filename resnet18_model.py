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
        model,
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
):
    """Train ResNet18 on provided train/val splits (no cross-validation).
    Mirrors the behaviour of train_vgg16 in vgg16_model.py so that the same CLI
    arguments & logging strategy work for both backbones.
    Returns the filepath of the best model saved on validation accuracy."""

    if not os.path.exists("best_models"):
        os.makedirs("best_models")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    if save_result:
        os.makedirs("results", exist_ok=True)
        with open(f"results/{model_name}_result.csv", "w") as f:
            f.write("Epoch,Loss,Train Acc,Val Acc,Time(s),Learning Rate\n")

    best_accuracy = 0.0
    current_lr = learning_rate
    non_update_count = 0
    lr_increase_count = 0

    print("Starting Training – ResNet18")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
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
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            del images, labels, outputs, predicted
            if i % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        train_time = time.time() - start_time

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                del images, labels, outputs, predicted
        val_acc = 100 * val_correct / val_total
        print(f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Loss: {avg_loss:.4f}, Time: {train_time:.2f}s")

        # Early stopping & LR scheduling (same logic as VGG)
        if val_acc - best_accuracy > 0.01:
            best_accuracy = val_acc
            non_update_count = 0
            torch.save(model.state_dict(), f"best_models/{model_name}_best.pth")
            print(f"Saved best model to best_models/{model_name}_best.pth")
        else:
            non_update_count += 1

        if non_update_count >= early_stopping_patience:
            if lr_increase_count < lr_increase_patience:
                current_lr += lr_increment_rate
                for pg in optimizer.param_groups:
                    pg['lr'] = current_lr
                lr_increase_count += 1
                non_update_count = 0
                print(f"Learning rate increased to {current_lr}")
            else:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if save_result:
            with open(f"results/{model_name}_result.csv", "a") as f:
                f.write(f"{epoch+1},{avg_loss:.4f},{train_acc:.2f},{val_acc:.2f},{train_time:.2f},{current_lr}\n")

    print("Training complete!")
    return f"best_models/{model_name}_best.pth"


def train_resnet18_with_CV(
        model,
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
):
    """K-fold cross-validation training for ResNet18, mirroring vgg16 logic."""

    if not os.path.exists("best_models"):
        os.makedirs("best_models")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if save_result:
        os.makedirs("results", exist_ok=True)
        with open(f"results/{model_name}_cv_result.csv", "w") as f:
            f.write("Fold,Epoch,Loss,Train Acc,Val Acc,Time(s),Learning Rate\n")

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_data)):
        print(f"\n{'='*60}\nFOLD {fold+1}/{n_folds}\n{'='*60}")

        fold_model = ResNet18(num_classes=model.fc.out_features).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(fold_model.parameters(), lr=learning_rate)

        train_subset = Subset(train_data, train_idx)
        val_subset = Subset(train_data, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        best_val_acc = 0.0
        current_lr = learning_rate
        non_update = 0
        lr_inc_count = 0

        for epoch in range(num_epochs):
            print(f"Fold {fold+1} — Epoch {epoch+1}/{num_epochs}")
            start_time = time.time()
            fold_model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = fold_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                _, pred = outputs.max(1)
                total += labels.size(0)
                correct += pred.eq(labels).sum().item()
                del images, labels, outputs, pred
            train_acc = 100 * correct / total
            avg_loss = running_loss / len(train_loader)
            train_time = time.time() - start_time

            # Validation
            fold_model.eval()
            val_corr = 0
            val_tot = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = fold_model(images)
                    _, pred = outputs.max(1)
                    val_tot += labels.size(0)
                    val_corr += pred.eq(labels).sum().item()
                    del images, labels, outputs, pred
            val_acc = 100 * val_corr / val_tot
            print(f"Train {train_acc:=.2f}%, Val {val_acc:=.2f}%, Loss {avg_loss:.4f}")

            # Early stop & LR schedule
            if val_acc - best_val_acc > 0.01:
                best_val_acc = val_acc
                non_update = 0
                torch.save(fold_model.state_dict(), f"best_models/{model_name}_fold{fold+1}_best.pth")
            else:
                non_update += 1

            if non_update >= early_stopping_patience:
                if lr_inc_count < lr_increase_patience:
                    current_lr += lr_increment_rate
                    for pg in optimizer.param_groups:
                        pg['lr'] = current_lr
                    lr_inc_count += 1
                    non_update = 0
                    print(f"LR increased to {current_lr}")
                else:
                    print("Early stopping this fold")
                    break

            if save_result:
                with open(f"results/{model_name}_cv_result.csv", "a") as f:
                    f.write(f"{fold+1},{epoch+1},{avg_loss:.4f},{train_acc:.2f},{val_acc:.2f},{train_time:.2f},{current_lr}\n")

        fold_results.append({'fold': fold+1, 'best_val_accuracy': best_val_acc, 'model_path': f"best_models/{model_name}_fold{fold+1}_best.pth"})
        print(f"Fold {fold+1} best Val Acc: {best_val_acc:.2f}%")

    # CV stats
    val_accs = [fr['best_val_accuracy'] for fr in fold_results]
    print(f"\n{'='*60}\nCROSS-VALIDATION RESULTS\n{'='*60}")
    for fr in fold_results:
        print(f"Fold {fr['fold']}: {fr['best_val_accuracy']:.2f}%")
    print(f"Mean {np.mean(val_accs):.2f}% ± {np.std(val_accs):.2f}%")

    if save_result:
        pd.DataFrame(fold_results).to_csv(f"results/{model_name}_cv_summary.csv", index=False)

    # return best model path of the last fold (user can decide)
    return fold_results[-1]['model_path']


def test_resnet18(model_path, experiment_name, test_data, device, num_classes, save_result=False, verbose=False):
    """Evaluate a saved ResNet18 model on test_data, returning accuracy and confusion-matrix mapping (mirrors vgg16 test)."""

    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists("results/classification_reports"):
        os.makedirs("results/classification_reports")
    if not os.path.exists("results/confusion_matrices"):
        os.makedirs("results/confusion_matrices")

    model = ResNet18(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    label_names_idx = test_data.get_dataset_labels()
    idx_to_label = {idx: name for name, idx in label_names_idx.items()}

    y_true = []
    y_pred = []
    y_img_ids = []
    confusion_images = {}

    correct_total = 0
    with torch.no_grad():
        for idx, (img, label) in enumerate(test_loader):
            img = img.to(device)
            label = label.to(device)
            output = model(img)
            _, pred = output.max(1)
            true_idx = label.item()
            pred_idx = pred.item()
            true_cls = idx_to_label[true_idx]
            pred_cls = idx_to_label[pred_idx]
            correct_total += int(true_idx == pred_idx)

            y_true.append(true_cls)
            y_pred.append(pred_cls)
            img_id = test_data.get_image_id(idx)
            y_img_ids.append(img_id)
            key = (true_cls, pred_cls)
            confusion_images.setdefault(key, []).append(img_id)

            if verbose:
                print(f"Image {img_id} – True: {true_cls}, Pred: {pred_cls}")
            del img, label, output, pred
    total = len(test_loader.dataset)
    acc = 100 * correct_total / total
    print(f"Test Accuracy: {acc:.2f}% ({correct_total}/{total})")

    # Classification report & confusion matrix
    class_report = classification_report(y_true, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=list(idx_to_label.values()))
    print(classification_report(y_true, y_pred))
    print(conf_matrix)

    results_dict = {
        'experiment_name': experiment_name,
        'model_path': model_path,
        'accuracy': acc,
        'confusion_matrix_images': confusion_images,
        'classification_report': class_report,
    }

    if save_result:
        # JSON
        with open(f"results/{experiment_name}_resnet18_test.json", "w") as f:
            json.dump(results_dict, f, indent=2)
        # CSV outputs
        pd.DataFrame(class_report).T.to_csv(f"results/classification_reports/{experiment_name}_resnet18_classification_report.csv")
        cm_df = pd.DataFrame(conf_matrix, index=list(idx_to_label.values()), columns=list(idx_to_label.values()))
        cm_df.to_csv(f"results/confusion_matrices/{experiment_name}_resnet18_confusion_matrix.csv")

    return results_dict

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
    #     transform="vgg16",  # This will apply VGG16 transforms
    #     target_transform="integer"  # This will convert labels to integers
    # )

    # test_data = prepare_data_from_preselected_categories(
    #     prechosen_categories_csv_path, 
    #     'test',
    #     transform="vgg16",  # This will apply VGG16 transforms
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
    # best_model_path2 = train_vgg16_with_CV(model2, 
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
    
    # test_vgg16(best_model_path2, test_data, device, num_classes)
