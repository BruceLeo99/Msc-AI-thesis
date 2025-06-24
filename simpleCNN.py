import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR, CyclicLR
from sklearn.model_selection import train_test_split, KFold
import time
import os
from outdated_models.MSCOCO_preprocessing_local import *
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, num_conv_layers=2):
        super(SimpleCNN, self).__init__()
        
        # Initialize lists to store layers
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        
        # First conv layer
        self.conv_layers.append(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=2, padding_mode='zeros')
        )
        self.pool_layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=1))
        
        # Additional conv layers
        current_channels = 32
        for i in range(1, num_conv_layers):
            out_channels = current_channels * 2
            self.conv_layers.append(
                nn.Conv2d(current_channels, out_channels, kernel_size=3, stride=1, padding=2, padding_mode='zeros')
            )
            self.pool_layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=1))
            current_channels = out_channels

        # Calculate the size of flattened features
        self.feature_size = self._get_conv_output_size(in_channels)
        
        # Update linear layer with correct input features
        self.fc1 = nn.Linear(in_features=self.feature_size, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=num_classes)

    def _get_conv_output_size(self, in_channels):
        # Helper function to calculate output size of conv layers
        x = torch.randn(1, in_channels, 224, 224)  # Assuming 224x224 input
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            x = F.relu(conv(x))
            x = pool(x)
        return x.numel() // x.size(0)  # Flatten all dimensions except batch

    def forward(self, x):
        # Pass through all conv and pool layers
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            x = F.relu(conv(x))
            x = pool(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

def train_simpleCNN(
        train_data,
        val_data,
        model_name,
        device,
        num_conv_layers = 2,
        num_epochs = 10,
        learning_rate = 0.0001,
        batch_size = 32,
        lr_adjustment_rate = 0.0001,
        lr_adjustment_mode = 'decrease',
        lr_adjustment_patience = 5,
        save_result = False,
        early_stopping_patience = 10,
        num_workers = 0,
        result_foldername='results'
):
    
    """
    Trains a simple CNN model on the training set and validates it on the validation set.
    Data is preloaded and splitted into train and validation sets manually.
    Returns the path to the best model.

    Args:
        train_data: Training data
        val_data: Validation data
        model_name: Name of the model
        device: Device to train on
        num_epochs: Number of epochs to train for
        learning_rate: Learning rate for the optimizer
        batch_size: Batch size for training
        lr_adjustment_rate: Rate at which to increment the learning rate
        lr_adjustment_mode: Mode of learning rate adjustment
        lr_adjustment_patience: Patience for learning rate increase
        save_result: Whether to save the result
        early_stopping_patience: Patience for early stopping
        num_workers: Number of workers for data loading
    """
    if not os.path.exists(f"/var/scratch/yyg760/best_models"):
        os.makedirs(f"/var/scratch/yyg760/best_models")

    best_models_foldername = f"/var/scratch/yyg760/best_models"

    if not os.path.exists(result_foldername):
        os.makedirs(result_foldername)

    num_classes = train_data.get_num_classes()
    model = SimpleCNN(num_classes=num_classes, num_conv_layers=num_conv_layers).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = torch.nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()


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
    
    print("Starting Training.\nModel: SimpleCNN with No Prototypes")

    if save_result:
        with open(f"{result_foldername}/{model_name}_result.csv", "w") as f:
            f.write("Epoch,Training Loss,Validation Loss,Training Accuracy,Validation Accuracy,Time,Learning Rate\n")

    best_accuracy = 0
    current_lr = learning_rate
    epochs_without_improvement = 0
    lr_adjustments_made = 0
    best_epoch = 0
    best_model_state = None

    train_correct = 0
    train_total = 0

    # Train and keep backpropagating on batches
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1} of {num_epochs}")

        ### Training phase
        train_start_time = time.time()
        model.train()
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
                best_model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()  # Create a copy
                # torch.save(best_model_state, f"{best_models_foldername}/{model_name}_best.pth")
                print(f"Model saved to {best_models_foldername}/{model_name}_best.pth")

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
                    current_lr = learning_rate  # Update current learning rate
                    # Update optimizer learning rate
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = learning_rate
                    lr_adjustments_made += 1
                    epochs_without_improvement = 0
                    print(f"Learning rate increased to {learning_rate}")
                else:
                    print(f"Early stopping at epoch {epoch+1} due to no improvement in validation accuracy after increasing learning rate {lr_adjustment_patience} times")
                    break

        # Calculate time spent for one epoch and save the result
        epoch_time = train_time_spent + val_time
        if save_result:
            with open(f"{result_foldername}/{model_name}_result.csv", "a") as f:
                f.write(f"{epoch+1},{loss.item()},{val_loss_avg:.4f},{train_accuracy:.2f},{val_accuracy:.2f},{epoch_time:.2f},{current_lr}\n")
    
    # End of training 
    print("Training Complete!")
    print(f"Best model saved from epoch {best_epoch} with validation accuracy {best_accuracy:.4f}")

    if best_model_state is not None:
        torch.save(best_model_state, f"{best_models_foldername}/{model_name}_best.pth")
        print(f"Final best model confirmed saved: {model_name}_best.pth")

    return f"{best_models_foldername}/{model_name}_best.pth"

def train_simpleCNN_with_CV(
        train_data,
        n_folds,
        num_epochs,
        learning_rate,
        model_name,
        device,
        batch_size,
        lr_adjustment_rate = 0.0001,
        lr_adjustment_mode = 'decrease',
        save_result = False,
        early_stopping_patience = 10,
        lr_adjustment_patience = 5,
        random_state = 42,
        num_workers = 0,
        result_foldername='results'
):
    """
    Trains a simple CNN model with cross-validation on the training set.
    Returns the path to the best model.
    """
    pass # Develop later if needed. 



def test_simpleCNN(model_path, 
               experiment_name, 
               test_data, 
               device, 
               positive_class: int = 1, 
               save_result=False, 
               verbose=False,
               result_foldername='results'
               ):
    """
    Loads a trained SimpleCNN model from a .pth file and tests it on the test set.
    Additionally returns image-index lists for TP, FP, TN, FN (binary-class assumption).
    Args:
        model_path: Path to saved model
        test_data: Dataset to evaluate
        device: torch device
        positive_class: which class id is treated as the "positive" class for TP/FP definitions
    Returns
        test_accuracy, dict with keys 'tp','fp','tn','fn' mapping to lists of sample indices
    """

    if not os.path.exists(f"{result_foldername}/classification_reports"):
        os.makedirs(f"{result_foldername}/classification_reports")

    if not os.path.exists(f"{result_foldername}/confusion_matrices"):
        os.makedirs(f"{result_foldername}/confusion_matrices")

    # Load state dict to analyze architecture
    state_dict = torch.load(model_path)
    
    # Count number of convolutional layers by looking at conv_layers keys
    num_conv_layers = max([int(key.split('.')[1]) for key in state_dict.keys() if key.startswith('conv_layers')]) + 1
    
    if verbose:
        print(f"Detected {num_conv_layers} convolutional layers in saved model")

    # Create model with detected architecture
    model = SimpleCNN(num_classes=test_data.get_num_classes(), num_conv_layers=num_conv_layers).to(device)
    model.load_state_dict(state_dict)
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
    label_to_idx = test_data.get_dataset_labels()
    print(f"Label names and indices: {label_to_idx}")
    
    # Create reverse mapping from index to name
    idx_to_label = {idx: name for name, idx in label_to_idx.items()}
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

    classi_report = classification_report(y_true, y_pred, labels=list(label_to_idx.keys()), target_names=list(label_to_idx.keys()), output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred)

    print("\nClassification Report:")
    print(classi_report)

    print("\nConfusion Matrix:")
    print(conf_matrix)


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

def get_last_batch(dataloader):
    """
    Get the last batch from a DataLoader
    Args:
        dataloader: PyTorch DataLoader object
    Returns:
        last_batch: tuple of (images, labels) for the last batch
    """
    last_batch = None
    for batch in dataloader:  # Iterate through all batches
        last_batch = batch    # Keep updating until we reach the last one
    return last_batch

# Alternative method using length
def get_last_batch_by_index(dataloader):
    """
    Get the last batch from a DataLoader using its length
    Args:
        dataloader: PyTorch DataLoader object
    Returns:
        last_batch: tuple of (images, labels) for the last batch
    """
    num_batches = len(dataloader)
    for i, batch in enumerate(dataloader):
        if i == num_batches - 1:  # If this is the last batch
            return batch
    return None

# Example usage:
# last_batch = get_last_batch(train_loader)
# images, labels = last_batch
# print(f"Last batch size: {images.shape[0]}")  # Might be smaller than batch_size if dataset size isn't divisible by batch_size
# print(f"Last batch images shape: {images.shape}")
# print(f"Last batch labels shape: {labels.shape}")  