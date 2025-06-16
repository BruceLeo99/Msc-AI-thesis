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
        num_workers=4,
        result_foldername='results'
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

    if not os.path.exists("/var/scratch/yyg760/best_models"):
        os.makedirs("/var/scratch/yyg760/best_models")

    best_models_foldername = f"/var/scratch/yyg760/best_models"

    if not os.path.exists(result_foldername):
        os.makedirs(result_foldername)

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
        with open(f"{result_foldername}/{model_name}_result.csv", "w") as f:
            f.write("Epoch,Mode,Time,Cross Entropy,Cluster,Separation,Avg Cluster,Accuracy,L1,P Avg Pair Dist,Learning Rate\n")

    best_accuracy = 0.0
    best_epoch = 0
    current_lr = learning_rate
    epochs_without_improvement = 0
    lr_adjustments_made = 0
    best_model_state = None

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
            with open(f"{result_foldername}/{model_name}_result.csv", "a") as f:
                f.write(f"{epoch+1},train,{running_time},{train_loss},{cluster_cost},{separation_cost},{avg_cluster_cost},{train_accuracy},{l1},{p_avg_pair_dist},{current_lr}\n")

        # ---- Validation ----
        mode,running_time, val_loss, cluster_cost, separation_cost, avg_cluster_cost, val_accuracy, l1, p_avg_pair_dist = \
            validate(model,
             val_loader,
             class_specific=class_specific,
             get_full_results=True)
        
        # ---- CSV logging for validation ----
        if save_result:
            with open(f"{result_foldername}/{model_name}_result.csv", "a") as f:
                f.write(f"{epoch+1},validation,{running_time},{val_loss},{cluster_cost},{separation_cost},{avg_cluster_cost},{val_accuracy},{l1},{p_avg_pair_dist},{current_lr}\n")

        # ---- Model saving logic (fixed) ----
        if val_accuracy > best_accuracy:  # Changed from > 0.01 improvement to any improvement
            print(f"Model improved: {best_accuracy:.4f} → {val_accuracy:.4f} (epoch {epoch+1})")
            best_accuracy = val_accuracy
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            
            # Save best model based on validation accuracy
            best_model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            # torch.save(best_model_state, f"{best_models_foldername}/{model_name}_best.pth")
            print(f"Best model saved for epoch {epoch+1} with accuracy {val_accuracy:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs (best: {best_accuracy:.4f} at epoch {best_epoch})")

        # ---- Learning rate adjustment logic ----
        if epochs_without_improvement >= early_stopping_patience:
            if lr_adjustments_made < lr_adjustment_patience:
                if lr_adjustment_mode == 'increase':
                    current_lr += lr_adjustment_rate

                elif lr_adjustment_mode == 'decrease':
                    current_lr -= lr_adjustment_rate
                
                # Update optimizer learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                
                lr_adjustments_made += 1
                epochs_without_improvement = 0  # Reset patience for this LR
                
                print(f"Learning rate adjusted to {current_lr} due to no improvement for {epochs_without_improvement} epochs (adjustment {lr_adjustments_made}/{lr_adjustment_patience})")
                print(f"Continuing training with new LR. Best model remains from epoch {best_epoch} with accuracy {best_accuracy:.4f}")
            else:
                print(f"Early stopping triggered at epoch {epoch+1}")
                print(f"Maximum LR adjustments ({lr_adjustment_patience}) reached")
                print(f"Final best model: epoch {best_epoch} with accuracy {best_accuracy:.4f}")
                break

    print("Training complete!")
    print(f"Best model saved from epoch {best_epoch} with validation accuracy {best_accuracy:.4f}")
    
    # Ensure the best model is saved one final time
    if best_model_state is not None:
        torch.save(best_model_state, f"{best_models_foldername}/{model_name}_best.pth")
        print(f"Final best model confirmed saved: {model_name}_best.pth")
    
    return f"{best_models_foldername}/{model_name}_best.pth"



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
                   coefs=None,
                   result_foldername='results'
                   ):   
    
    """Test a saved ProtoPNet model on held-out data (mirrors VGG16 test)."""

    if not os.path.exists(result_foldername):
        os.makedirs(result_foldername)

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

    print("Starting model testing...")
    label_to_idx = test_data.get_dataset_labels()
    print(f"Label names and indices: {label_to_idx}")
    
    # Create reverse mapping from index to name
    idx_to_label = {idx: name for name, idx in label_to_idx.items()}
    print(f"Index to label mapping: {idx_to_label}")

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

            # Store mapping for confusion matrix visualization with actual image IDs
            for idx in range(len(batch['label'])):
                t = target[idx].item()
                p = predicted[idx].item()
                key = f"{t}_{p}"
                # Get the actual image ID for this index in the batch
                img_id = test_data.get_image_id(i * test_loader.batch_size + idx)
                confusion_mapping.setdefault(key, []).append(img_id)

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
    classi_report = classification_report(y_true, y_pred, labels=list(label_to_idx.keys()), target_names=list(label_to_idx.keys()), output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=list(label_to_idx.keys()))

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
            'classification_report': classi_report,
            'confusion_matrix': conf_matrix.tolist(),
            'confusion_mapping': confusion_mapping
        }
        
        with open(f"{result_foldername}/{experiment_name}_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
            
        # Save detailed reports
        if not os.path.exists(f"{result_foldername}/classification_reports"):
            os.makedirs(f"{result_foldername}/classification_reports")
        if not os.path.exists(f"{result_foldername}/confusion_matrices"):
            os.makedirs(f"{result_foldername}/confusion_matrices")
            
        pd.DataFrame(classi_report).T.to_csv(
            f"{result_foldername}/classification_reports/{experiment_name}_classification_report.csv")
        pd.DataFrame(conf_matrix,
                    index=[idx_to_label[i] for i in range(num_classes)],
                    columns=[idx_to_label[i] for i in range(num_classes)]).to_csv(
                        f"{result_foldername}/confusion_matrices/{experiment_name}_confusion_matrix.csv")

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
    result_foldername = 'results'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_20cls_json = "dataset_infos/singleLabel_train_data_20classes.json"
    val_20cls_json = "dataset_infos/singleLabel_val_data_20classes.json"
    test_20cls_json = "dataset_infos/singleLabel_test_data_20classes.json"

    train_30cls_json = "dataset_infos/singleLabel_train_data_30classes.json"
    val_30cls_json = "dataset_infos/singleLabel_val_data_30classes.json"
    test_30cls_json = "dataset_infos/singleLabel_test_data_30classes.json"

    train_data_20cls = MSCOCOCustomDataset(load_from_local=True, load_captions=True, load_from_json=train_20cls_json)
    val_data_20cls = MSCOCOCustomDataset(load_from_local=True, load_captions=True, load_from_json=val_20cls_json)
    test_data_20cls = MSCOCOCustomDataset(load_from_local=True, load_captions=True, load_from_json=test_20cls_json)

    train_data_30cls = MSCOCOCustomDataset(load_from_local=True, load_captions=True, load_from_json=train_30cls_json)
    val_data_30cls = MSCOCOCustomDataset(load_from_local=True, load_captions=True, load_from_json=val_30cls_json)
    test_data_30cls = MSCOCOCustomDataset(load_from_local=True, load_captions=True, load_from_json=test_30cls_json)


    # experiment_path = train_multimodal_PBN(train_data_20cls, 
    #                                        val_data_20cls, 
    #                                        model_name, 
    #                                        device, 
    #                                        num_epochs=num_epochs, 
    #                                        learning_rate=learning_rate, 
    #                                        batch_size=batch_size, 
    #                                        lr_adjustment_rate=lr_adjustment_rate, 
    #                                        save_result=save_result, 
    #                                        early_stopping_patience=early_stopping_patience, 
    #                                        lr_adjustment_patience=lr_adjustment_patience,
    #                                        class_specific=class_specific,
    #                                        get_full_results=get_full_results,
    #                                        num_workers=4)


    
    # test_multimodal_PBN(experiment_path, 
    #                     model_name, 
    #                     test_data_20cls, 
    #                     device, 
    #                     num_prototypes=num_prototypes,
    #                     class_specific=class_specific,
    #                     get_full_results=get_full_results,
    #                     save_result=save_result,
    #                     verbose=True)

    model_name1 = 'mPBN_20classes_10prototypes_0.0001lr'
    model_name2 = 'mPBN_20classes_20prototypes_0.0001lr'
    model_name3 = 'mPBN_20classes_30prototypes_0.0001lr'
    model_name4 = 'mPBN_30classes_10prototypes_0.0001lr'
    model_name5 = 'mPBN_30classes_20prototypes_0.0001lr'
    model_name6 = 'mPBN_30classes_30prototypes_0.0001lr'

    experiment_path_dir = 'results_mPBN1/best_models'

    test_multimodal_PBN(f"{experiment_path_dir}/{model_name1}_best.pth", 
                    model_name1, 
                    test_data_20cls, 
                    device, 
                    num_prototypes=10,
                    class_specific=class_specific,
                    get_full_results=get_full_results,
                    save_result=save_result,
                    verbose=True)


    test_multimodal_PBN(f"{experiment_path_dir}/{model_name2}_best.pth", 
                    model_name2, 
                    test_data_20cls, 
                    device, 
                    num_prototypes=20,
                    class_specific=class_specific,
                    get_full_results=get_full_results,
                    save_result=save_result,
                    verbose=True)
    
    test_multimodal_PBN(f"{experiment_path_dir}/{model_name3}_best.pth", 
                    model_name3, 
                    test_data_20cls, 
                    device, 
                    num_prototypes=30,
                    class_specific=class_specific,
                    get_full_results=get_full_results,
                    save_result=save_result,
                    verbose=True)
    
    test_multimodal_PBN(f"{experiment_path_dir}/{model_name4}_best.pth", 
                    model_name4, 
                    test_data_30cls, 
                    device, 
                    num_prototypes=10,
                    class_specific=class_specific,
                    get_full_results=get_full_results,
                    save_result=save_result,
                    verbose=True)
    

    test_multimodal_PBN(f"{experiment_path_dir}/{model_name5}_best.pth", 
                    model_name5, 
                    test_data_30cls, 
                    device, 
                    num_prototypes=20,
                    class_specific=class_specific,
                    get_full_results=get_full_results,
                    save_result=save_result,
                    verbose=True)
    

    test_multimodal_PBN(f"{experiment_path_dir}/{model_name6}_best.pth", 
                    model_name6, 
                    test_data_30cls, 
                    device, 
                    num_prototypes=30,
                    class_specific=class_specific,
                    get_full_results=get_full_results,
                    save_result=save_result,
                    verbose=True)