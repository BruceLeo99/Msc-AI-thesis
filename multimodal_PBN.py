from transformers import VisualBertModel, VisualBertConfig
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.parallel import DataParallel
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import json
import pandas as pd
import os
import time
from collections import OrderedDict

from MSCOCO_preprocessing_local import *
from ProtoPNet.train_and_test_mPBN import train, validate

def collate_fn(batch):
    """
    Custom collate function to properly batch the inputs for VisualBERT.
    """
    # Separate the dictionary items
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    token_type_ids = torch.stack([item['token_type_ids'] for item in batch])
    visual_embeds = torch.stack([item['visual_embeds'] for item in batch])  # Should be (B, C, H, W)
    labels = torch.tensor([item['label'] for item in batch])
    
    # Create visual attention mask (1 for each visual feature)
    B = len(batch)
    visual_attention_mask = torch.ones(B, 1, dtype=torch.long)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'visual_embeds': visual_embeds,
        'visual_attention_mask': visual_attention_mask,
        'label': labels
    }

class VisualBertPPNet(nn.Module):
    """
    ProtoPNet whose feature extractor is a pretrained VisualBERT encoder.
    Prototypes live in the same (hidden-size) space as the [CLS] output.
    """
    def __init__(self,
                 ckpt='uclanlp/visualbert-vqa-coco-pre',
                 num_prototypes=10,
                 num_classes=20):
        super().__init__()
        # Load pretrained VGG16
        vgg16 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
        # Remove the last layer (classifier)
        self.vgg16_features = nn.Sequential(*list(vgg16.features))
        
        self.encoder = VisualBertModel.from_pretrained(ckpt, hidden_act='relu')
        hid = self.encoder.config.hidden_size          # 768
        
        # Initialize prototype-related attributes
        self.num_prototypes = num_prototypes
        self.num_classes = num_classes
        self.prototype_shape = (num_prototypes, hid)
        self.prototype_vectors = nn.Parameter(torch.randn(num_prototypes, hid))
        
        # Initialize prototype_class_identity matrix as a registered buffer
        # This way it will be automatically moved to the correct device with the model
        prototype_class_identity = torch.zeros(num_prototypes, num_classes)
        num_prototypes_per_class = num_prototypes // num_classes
        for j in range(num_classes):
            prototype_class_identity[j * num_prototypes_per_class:(j + 1) * num_prototypes_per_class, j] = 1
        self.register_buffer('prototype_class_identity', prototype_class_identity)
        
        self.last_layer = nn.Linear(num_prototypes,    # 1-to-1 with prototypes
                                    num_classes,
                                    bias=False)
        
        # Project VGG16 features (512-dim) to VisualBERT's expected dimension (2048-dim)
        self.visual_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten(),  # Flatten to (B, 512)
            nn.ReLU(),
            nn.Linear(512, 2048)  # Project to VisualBERT's expected dimension
        )

    def forward(self, batch):
        # Handle visual embeddings
        visual_embeds = batch['visual_embeds']  # (B, 3, 224, 224)
        print(f"Initial visual_embeds shape: {visual_embeds.shape}")
        
        # Extract VGG16 features
        visual_embeds = self.vgg16_features(visual_embeds)  # (B, 512, H', W')
        print(f"After VGG16 features shape: {visual_embeds.shape}")
        
        # Project through our projection pipeline
        visual_embeds = self.visual_projection[0](visual_embeds)  # AdaptiveAvgPool2d
        print(f"After pooling shape: {visual_embeds.shape}")
        
        visual_embeds = self.visual_projection[1](visual_embeds)  # Flatten
        print(f"After flatten shape: {visual_embeds.shape}")
        
        visual_embeds = self.visual_projection[2](visual_embeds)  # ReLU
        print(f"After ReLU shape: {visual_embeds.shape}")
        
        visual_embeds = self.visual_projection[3](visual_embeds)  # Linear projection
        print(f"After linear shape: {visual_embeds.shape}")
        
        visual_embeds = visual_embeds.unsqueeze(1)  # Add sequence dimension
        print(f"After unsqueeze shape: {visual_embeds.shape}")

        # Prepare inputs for VisualBERT
        inputs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'token_type_ids': batch['token_type_ids'],
            'visual_embeds': visual_embeds,
            'visual_attention_mask': batch['visual_attention_mask']
        }

        # Get VisualBERT output
        out = self.encoder(**inputs).last_hidden_state   # (B, L, H)
        cls = out[:, 0]                                # (B, H)
        
        # ---- ProtoPNet distance & logits ----
        dists = ((cls.unsqueeze(1) - self.prototype_vectors)**2).sum(-1)
        logits = -self.last_layer(dists)               # minus dist = similarity
        return logits, dists


def train_multimodal_PBN(
        train_data,
        val_data,
        model_name,
        device,
        num_prototypes=10,
        num_epochs=50,
        learning_rate=0.0001,
        batch_size=32,
        lr_adjustment_rate=0.0001,
        lr_adjustment_mode='decrease',
        lr_adjustment_patience=5,
        save_result=False,
        early_stopping_patience=10,
        class_specific=True,
        get_full_results=True,
        num_workers=4
):
    """Train ProtoPNet in the same logging/early-stopping style used for VGG16/ResNet.

    Args:
        train_data: Training data
        val_data: Validation data
        model_name: Name of the model
        device: Device to use for training
        num_prototypes: Number of prototypes
        num_epochs: Number of epochs to train
        learning_rate: Learning rate for training
        batch_size: Batch size for training
        lr_adjustment_rate: Learning rate increment rate
        lr_adjustment_mode: Mode of learning rate adjustment
        lr_adjustment_patience: Patience for learning rate increase
        save_result: Whether to save the result

    Returns the path to the best model saved on validation accuracy.
    """

    if not os.path.exists("best_models"):
        os.makedirs("best_models")
    if not os.path.exists("results"):
        os.makedirs("results")

    num_classes = train_data.get_num_classes()

    prototype_shape = (num_classes*num_prototypes, 512, 1, 1)

    model = VisualBertPPNet(num_prototypes=num_prototypes, num_classes=num_classes)
    
    model = model.to(device)
    # Always wrap in DataParallel since ProtoPNet code expects model.module access
    model = DataParallel(model)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
    else:
        print(f"Using 1 GPU for training (wrapped in DataParallel for ProtoPNet compatibility)")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if num_workers == 0:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_data, 
                                batch_size=batch_size, 
                                shuffle=True, 
                                num_workers=num_workers,
                                pin_memory=True,
                                prefetch_factor=2,
                                collate_fn=collate_fn
                                )
        
        val_loader = DataLoader(val_data, 
                                batch_size=batch_size, 
                                shuffle=False, 
                                num_workers=num_workers,
                                pin_memory=True,
                                prefetch_factor=2,
                                collate_fn=collate_fn
                                )

    # CSV header
    if save_result:
        with open(f"results/{model_name}_result.csv", "w") as f:
            f.write("Epoch,Mode,Time,Cross Entropy,Cluster,Separation,Avg Cluster,Accuracy,L1,P Avg Pair Dist,Learning Rate\n")

    best_accuracy = 0.0
    current_lr = learning_rate
    non_update = 0
    lr_inc_count = 0

    print("Starting ProtoPNet training…")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Clear GPU memory before each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        mode,running_time, train_loss, cluster_cost, separation_cost, avg_cluster_cost, train_accuracy, l1, p_avg_pair_dist = \
        train(model, 
              train_loader, 
              optimizer, 
              class_specific=class_specific,
              get_full_results=get_full_results)
        

        # ---- CSV logging for training ----
        if save_result:
            with open(f"results/{model_name}_result.csv", "a") as f:
                f.write(f"{epoch+1},train,{running_time},{train_loss},{cluster_cost},{separation_cost},{avg_cluster_cost},{train_accuracy},{l1},{p_avg_pair_dist},{learning_rate}\n")

        # ---- Validation ----
        mode,running_time, val_loss, cluster_cost, separation_cost, avg_cluster_cost, val_accuracy, l1, p_avg_pair_dist = \
            validate(model,
             val_loader,
             class_specific=class_specific,
             get_full_results=True)
        
        # ---- CSV logging for validation ----
        if save_result:
            with open(f"results/{model_name}_result.csv", "a") as f:
                f.write(f"{epoch+1},validation,{running_time},{val_loss},{cluster_cost},{separation_cost},{avg_cluster_cost},{val_accuracy},{l1},{p_avg_pair_dist},{learning_rate}\n")


        # ---- Early-stopping & LR schedule ----
        if val_accuracy - best_accuracy > 0.01:
            print("Model improved → saving")
            best_accuracy = val_accuracy
            non_update = 0
            torch.save(model.state_dict(), f"best_models/{model_name}_best.pth")
            print(f"Best model saved for epoch {epoch+1}")
        else:
            non_update += 1
            print(f"No improvement for {non_update} epochs")

        if non_update >= early_stopping_patience:
            if lr_inc_count < lr_adjustment_patience:
                if lr_adjustment_mode == 'increase':
                    current_lr += lr_adjustment_rate
                elif lr_adjustment_mode == 'decrease':
                    current_lr -= lr_adjustment_rate
                learning_rate = current_lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                lr_inc_count += 1
                non_update = 0
                print(f"Learning-rate increased to {current_lr}")
            else:
                print("Early stopping triggered at epoch {epoch+1}")
                break


    print("Training complete!")
    return f"best_models/{model_name}_best.pth"



def test_multimodal_PBN(model_path, 
                   experiment_name, 
                   test_data, 
                   device, 
                   num_prototypes=10,
                   class_specific=True,
                   get_full_results=True,
                   save_result=False, 
                   verbose=False,
                   use_l1_mask=False,
                   coefs=None
                   ):
    
    """Test a saved ProtoPNet model on held-out data (mirrors VGG16 test)."""

    if not os.path.exists("results"):
        os.makedirs("results")

    num_classes = test_data.get_num_classes()

    # First create the model
    model = VisualBertPPNet(num_prototypes=num_prototypes, num_classes=num_classes)

    # Load state dict and handle DataParallel prefix
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:] # remove 'module.' prefix
        else:
            name = k
        new_state_dict[name] = v
        
    # Now load the state dict
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    
    # Always wrap in DataParallel since ProtoPNet code expects model.module access
    model = torch.nn.DataParallel(model)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for testing")
    else:
        print(f"Using 1 GPU for testing (wrapped in DataParallel for ProtoPNet compatibility)")

    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate_fn)

    label_names_idx = test_data.get_dataset_labels()
    idx_to_label = {idx: name for name, idx in label_names_idx.items()}

    model.eval()
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0

    y_true = []
    y_pred = []
    y_img_ids = []
    confusion_mapping = {}

    for i, batch in enumerate(test_loader):
        # Move each tensor in the batch to device
        inputs = {
            'input_ids': batch['input_ids'].to(device),
            'attention_mask': batch['attention_mask'].to(device),
            'token_type_ids': batch['token_type_ids'].to(device),
            'visual_embeds': batch['visual_embeds'].to(device),
            'visual_attention_mask': batch['visual_attention_mask'].to(device)
        }
        target = batch['label'].to(device)

        with torch.no_grad():
            output, min_distances = model(inputs)

            # compute loss
            cross_entropy = torch.nn.functional.cross_entropy(output, target)

            if class_specific:
                prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,target]).to(device)
                inverted_distances, _ = torch.max((1.0 - min_distances) * prototypes_of_correct_class, dim=1)
                cluster_cost = torch.mean(1.0 - inverted_distances)

                # calculate separation cost
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = \
                    torch.max((1.0 - min_distances) * prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean(1.0 - inverted_distances_to_nontarget_prototypes)

                # calculate avg cluster cost
                avg_separation_cost = \
                    torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
                avg_separation_cost = torch.mean(avg_separation_cost)

            else:
                min_distance, _ = torch.min(min_distances, dim=1)
                cluster_cost = torch.mean(min_distance)

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()
            
            # Store the raw integer labels for confusion matrix
            y_true.extend(target.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            
            # Store mapping for confusion matrix visualization
            for t, p, idx in zip(target.cpu().numpy(), predicted.cpu().numpy(), range(len(batch['label']))):
                key = f"{t}_{p}"
                confusion_mapping.setdefault(key, []).append(idx)

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            if class_specific:
                total_separation_cost += separation_cost.item()
                total_avg_separation_cost += avg_separation_cost.item()

    end = time.time()

    accu = n_correct / n_examples * 100
    test_results = {
        'accuracy': accu,
        'cross_entropy': total_cross_entropy / n_batches,
        'cluster_cost': total_cluster_cost / n_batches,
        'time': end - start
    }
    
    if class_specific:
        test_results.update({
            'separation_cost': total_separation_cost / n_batches,
            'avg_separation_cost': total_avg_separation_cost / n_batches
        })

    # Generate classification report and confusion matrix
    class_report = classification_report(y_true, y_pred, 
                                      labels=range(num_classes),
                                      target_names=[idx_to_label[i] for i in range(num_classes)], 
                                      output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=range(num_classes))

    if verbose:
        print("\nTest Results:")
        print(f"Accuracy: {accu:.2f}%")
        print(f"Cross Entropy: {test_results['cross_entropy']:.4f}")
        print(f"Cluster Cost: {test_results['cluster_cost']:.4f}")
        if class_specific:
            print(f"Separation Cost: {test_results['separation_cost']:.4f}")
            print(f"Avg Separation Cost: {test_results['avg_separation_cost']:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_true, y_pred, 
                                 labels=range(num_classes),
                                 target_names=[idx_to_label[i] for i in range(num_classes)]))
        print("\nConfusion Matrix:")
        print(conf_matrix)

    if save_result:
        results = {
            'test_results': test_results,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'confusion_mapping': confusion_mapping
        }
        
        with open(f"results/{experiment_name}_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
            
        # Save detailed reports
        if not os.path.exists("results/classification_reports"):
            os.makedirs("results/classification_reports")
        if not os.path.exists("results/confusion_matrices"):
            os.makedirs("results/confusion_matrices")
            
        pd.DataFrame(class_report).T.to_csv(
            f"results/classification_reports/{experiment_name}_classification_report.csv")
        pd.DataFrame(conf_matrix,
                    index=[idx_to_label[i] for i in range(num_classes)],
                    columns=[idx_to_label[i] for i in range(num_classes)]).to_csv(
                        f"results/confusion_matrices/{experiment_name}_confusion_matrix.csv")

    return test_results

if __name__ == "__main__":

    # Create results directory if it doesn't exist
    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists("best_models"):
        os.makedirs("best_models")

    # experiment_path = "best_models/mPBNTestRun_best.pth"

    class_specific = True
    get_full_results = True
    num_epochs = 3
    learning_rate = 0.0001
    batch_size = 4
    lr_adjustment_rate = 0.0001
    save_result = True
    early_stopping_patience = 10
    lr_adjustment_patience = 5
    num_prototypes = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_20cls_json = "dataset_infos/singleLabel_train_data_20classes.json"
    val_20cls_json = "dataset_infos/singleLabel_val_data_20classes.json"
    test_20cls_json = "dataset_infos/singleLabel_test_data_20classes.json"

    train_data = MSCOCOCustomDataset(load_from_local=True, load_captions=True, load_from_json=train_20cls_json)
    val_data = MSCOCOCustomDataset(load_from_local=True, load_captions=True, load_from_json=val_20cls_json)
    test_data = MSCOCOCustomDataset(load_from_local=True, load_captions=True, load_from_json=test_20cls_json)

    model_name = "mPBNTestRun"

    experiment_path = train_multimodal_PBN(train_data, 
                                           val_data, 
                                           model_name, 
                                           device, 
                                           num_epochs=num_epochs, 
                                           learning_rate=learning_rate, 
                                           batch_size=batch_size, 
                                           lr_adjustment_rate=lr_adjustment_rate, 
                                           save_result=save_result, 
                                           early_stopping_patience=early_stopping_patience, 
                                           lr_adjustment_patience=lr_adjustment_patience,
                                           class_specific=class_specific,
                                           get_full_results=get_full_results,
                                           num_workers=4)


    
    test_multimodal_PBN(experiment_path, 
                        model_name, 
                        test_data, 
                        device, 
                        num_prototypes=num_prototypes,
                        class_specific=class_specific,
                        get_full_results=get_full_results,
                        save_result=save_result,
                        verbose=True)