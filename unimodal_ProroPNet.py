import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import json
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import socket
import keyboard 
import threading
import torch.nn.functional as F
from torch.utils.data import Subset, random_split
from sklearn.metrics import classification_report, confusion_matrix
import json

from MSCOCO_preprocessing import *

from ProtoPNet.model import construct_PPNet
from ProtoPNet.train_and_test import train, validate
from ProtoPNet.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from ProtoPNet.vgg_features import vgg16_features, vgg19_features
from torchvision import transforms

import pandas as pd

from collections import OrderedDict
 



def train_protopnet(
        train_data,
        val_data,
        model_name,
        device,
        num_epochs=50,
        learning_rate=0.0001,
        batch_size=32,
        lr_increment_rate=0.0001,
        save_result=False,
        early_stopping_patience=10,
        lr_increase_patience=5,
        base_architecture='vgg16',
        prototype_shape=(1,1,1,1),
        class_specific=True,
        get_full_results=True
):
    """Train ProtoPNet in the same logging/early-stopping style used for VGG16/ResNet.

    Returns the path to the best model saved on validation accuracy.
    """

    if not os.path.exists("best_models"):
        os.makedirs("best_models")
    if not os.path.exists("results"):
        os.makedirs("results")

    num_classes = train_data.get_num_classes()

    model = construct_PPNet(base_architecture=base_architecture, 
                            pretrained=True, 
                            prototype_shape=prototype_shape, 
                            num_classes=num_classes,
                            add_on_layers_type='bottleneck',
                            img_size=224)
    
    model = model.to(device)
    # Wrap model in DataParallel
    model = torch.nn.DataParallel(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

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
            if lr_inc_count < lr_increase_patience:
                current_lr += lr_increment_rate
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


def test_protopnet(model_path, 
                   experiment_name, 
                   test_data, 
                   device, 
                   num_classes, 
                   base_architecture='vgg16',
                   prototype_shape=(1,1,1,1),
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

    model = construct_PPNet(base_architecture=base_architecture, 
                            pretrained=True, 
                            prototype_shape=prototype_shape, 
                            num_classes=num_classes,
                            add_on_layers_type='bottleneck',
                            img_size=224)

    # Load state dict and handle DataParallel prefix
    state_dict = torch.load(model_path, map_location=device)
    # Handle DataParallel state dict
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:] # remove 'module.' prefix
        else:
            name = k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    # Wrap model in DataParallel
    model = torch.nn.DataParallel(model)

    test_loader = DataLoader(test_data, shuffle=False)

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


    y_true, y_pred, y_img_ids = [], [], []
    confusion_mapping = {}

    for i, (image, label) in enumerate(test_loader):
        input = image.to(device)
        target = label.to(device)

        with torch.no_grad():
            output, min_distances = model(input)

            # compute loss
            cross_entropy = torch.nn.functional.cross_entropy(output, target)

            if class_specific:
                max_dist = (model.module.prototype_shape[1]
                            * model.module.prototype_shape[2]
                            * model.module.prototype_shape[3])
                
                prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,label]).to(device)
                inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
                cluster_cost = torch.mean(max_dist - inverted_distances)

                # calculate separation cost
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = \
                    torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

                # calculate avg cluster cost
                avg_separation_cost = \
                    torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
                avg_separation_cost = torch.mean(avg_separation_cost)
                
                if use_l1_mask:
                    l1_mask = 1 - torch.t(model.module.prototype_class_identity).to(device)
                    m = model.module if hasattr(model, 'module') else model
                    l1 = (m.last_layer.weight * l1_mask).norm(p=1)
                else:
                    m = model.module if hasattr(model, 'module') else model
                    l1 = m.last_layer.weight.norm(p=1) 

            else:
                min_distance, _ = torch.min(min_distances, dim=1)
                cluster_cost = torch.mean(min_distance)
                m = model.module if hasattr(model, 'module') else model
                l1 = m.last_layer.weight.norm(p=1)

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()
            y_true.append(idx_to_label[target.item()])
            y_pred.append(idx_to_label[predicted.item()])
            y_img_ids.append(test_data.get_image_id(i))
            key = f"{target.item()}_{predicted.item()}"
            confusion_mapping.setdefault(key, []).append(test_data.get_image_id(i))

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()

        del input, target, output, predicted, min_distances


    individual_prediction_results = dict()

    for test_image_id, test_label, test_pred in zip(y_img_ids, y_true, y_pred):
        individual_prediction_results[test_image_id] = {
            'true_label': test_label,
            'pred_label': test_pred
        }

    class_report = classification_report(y_true, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=list(idx_to_label.values()))
    print(classification_report(y_true, y_pred))
    print(conf_matrix)

    test_accuracy = 100 * n_correct / n_examples
    test_loss = total_cross_entropy / n_batches
    test_cluster_cost = total_cluster_cost / n_batches
    test_separation_cost = total_separation_cost / n_batches
    test_avg_separation_cost = total_avg_separation_cost / n_batches
    test_l1 = model.module.last_layer.weight.norm(p=1).item()

    results = {
        'experiment_name': experiment_name,
        'pth_filepath': model_path,
        'accuracy': test_accuracy,
        'loss': test_loss,
        'cluster_cost': test_cluster_cost,
        'separation_cost': test_separation_cost,
        'avg_separation_cost': test_avg_separation_cost,
        'l1': test_l1,
        'confusion_matrix_images': confusion_mapping,
        'individual_prediction_results': individual_prediction_results,
        'classification_report': class_report,
    }

    if save_result:
        with open(f"results/{experiment_name}_test_result.json", "w") as f:
            json.dump(results, f, indent=2)
        pd.DataFrame(class_report).T.to_csv(
            f"results/classification_reports/{experiment_name}_classification_report.csv")
        pd.DataFrame(conf_matrix,
                     index=list(idx_to_label.values()),
                     columns=list(idx_to_label.values())).to_csv(
                         f"results/confusion_matrices/{experiment_name}_confusion_matrix.csv")
    return results


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create results directory if it doesn't exist
    if not os.path.exists("results"):
        os.makedirs("results")

    # Memory optimization settings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_per_process_memory_fraction(0.9)

    # Define model parameters
    base_architecture = 'vgg16'
    prototype_shape = (100, 512, 1, 1)
    class_specific = True
    get_full_results = True
    num_epochs = 50
    learning_rate = 0.0001
    batch_size = 4
    lr_increment_rate = 0.0001
    save_result = True
    early_stopping_patience = 10
    lr_increase_patience = 5
    experiment_name = 'ProtoPNet_testrun'

    # Define dataset parameters
    dataset_info = 'chosen_categories_3_10_v3.csv'
    df_dataset_info = pd.read_csv(dataset_info)
    categories = df_dataset_info["Category Name"].unique()
    num_classes = len(categories)

    # # Define dataset parameters
    # train_data, val_data = prepare_data_manually(*categories, 
    #                                              num_instances=15, 
    #                                              for_test=False, 
    #                                              split=True, 
    #                                              split_size=0.15, 
    #                                              experiment_name=experiment_name,
    #                                              transform=base_architecture,
    #                                              target_transform='integer',
    #                                              load_captions=False,
    #                                              save_result=save_result)
    
    test_data = prepare_data_manually(*categories, 
                                      num_instances=10, 
                                      for_test=True, 
                                      split=False, 
                                      experiment_name=experiment_name,
                                      transform=base_architecture,
                                      target_transform='integer',
                                      load_captions=False,
                                      save_result=save_result)
    
    # train_data, val_data, test_data = eliminate_leaked_data(experiment_name, train_data, val_data, test_data, save_result=save_result)
    

    # best_model_path = train_protopnet(train_data, 
    #                                   val_data, 
    #                                   experiment_name, 
    #                                   device, 
    #                                   num_epochs, 
    #                                   learning_rate, 
    #                                   batch_size, 
    #                                   lr_increment_rate, 
    #                                   save_result, 
    #                                   early_stopping_patience, 
    #                                   lr_increase_patience, 
    #                                   base_architecture, 
    #                                   prototype_shape, 
    #                                   class_specific, 
    #                                   get_full_results=True)

    best_model_path = "best_models/ProtoPNet_testrun_best.pth"
    
    test_results = test_protopnet(best_model_path, 
                                experiment_name, 
                                test_data, 
                                device, 
                                num_classes, 
                                base_architecture=base_architecture,
                                prototype_shape=prototype_shape,
                                class_specific=class_specific,
                                get_full_results=get_full_results,
                                save_result=save_result, 
                                verbose=False,
                                use_l1_mask=False,
                                coefs=None)