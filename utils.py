import pandas as pd
import matplotlib.pyplot as plt
import time
import torch
import gc
import numpy as np
import random

import json


def get_accuracy(df, mode):
    """
    Retrieves the accuracy of a result.Epoch

    :param df: Result as Pandas DataFrame
    :param mode: Whether gets the train or test accuracy. Fill in "train" or "test".
    :returns accuracy: an array of accuracies (either train or test) at each epoch
    """

    # assert mode == "train" or mode == "test", "invalid mode name, should be either train or test"
    
    if mode == 'train':
        accuracy = df[df['Mode'] == 'train']['Accuracy']

    elif mode == 'test':
        accuracy = df[df['Mode'] == 'test']['Accuracy']

    if accuracy.iloc[0] > 1:
        accuracy = accuracy / 100
    return accuracy

def get_crossentropy(df, mode):
    """
    Retrieves the cross entropy loss of a result.

    :param df: Result as Pandas DataFrame
    :param mode: Whether gets the train or test cross entropy. Fill in "train" or "test".
    :returns accuracy: an array of cross entropy (either train or test) at each epoch
    """

    # assert mode == "train" or mode == "test", "invalid mode name, should be either train or test"
    
    if mode == 'train':
        crossentropy = df[df['Mode'] == 'train']['Cross Entropy']

    elif mode == 'test':
        crossentropy = df[df['Mode'] == 'test']['Cross Entropy']

    return crossentropy

def generate_unimodal_name(df_name):
    encoder = df_name.split("_")[1]
    lr = df_name.split("-")[2]
    num_classes = df_name.split("-")[3]
    num_prototypes = df_name.split("-")[-1].replace(".csv", "")

    model_name = f"PBN {encoder} {lr} {num_classes} {num_prototypes}"

    return model_name 


def generate_subplot(ax, result_filename, value_type="Accuracy",row_nr=0,col_nr=0, onerow=False):

    ax_modelname = generate_unimodal_name(result_filename)
    ax_data = pd.read_csv(result_filename)

    if value_type == "Accuracy":
        ax_train_accu = get_accuracy(ax_data, 'train')
        ax_test_accu = get_accuracy(ax_data, 'test')
    elif value_type == "Cross Entropy":
        ax_train_accu = get_crossentropy(ax_data, 'train')
        ax_test_accu = get_crossentropy(ax_data, 'test')
    ax_num_epochs = list(range(1, len(ax_train_accu)+1))

    if onerow:
        ax[col_nr].plot(ax_num_epochs,ax_train_accu,label='Train')
        ax[col_nr].plot(ax_num_epochs, ax_test_accu, label='Test')
        ax[col_nr].set_xlabel('Epoch')
        ax[col_nr].set_ylabel(value_type)
        ax[col_nr].set_title(ax_modelname)
        ax[col_nr].set_xlim(0,70)
        if value_type == "Accuracy":
            ax[col_nr].set_ylim(0,1)
        ax[col_nr].legend()
    else:
        ax[row_nr,col_nr].plot(ax_num_epochs,ax_train_accu,label='Train')
        ax[row_nr,col_nr].plot(ax_num_epochs, ax_test_accu, label='Test')
        ax[row_nr,col_nr].set_xlabel('Epoch')
        ax[row_nr,col_nr].set_ylabel(value_type)
        ax[row_nr,col_nr].set_title(ax_modelname)
        ax[row_nr,col_nr].set_xlim(0,70)
        if value_type == "Accuracy":
            ax[row_nr,col_nr].set_ylim(0,1)
        ax[row_nr,col_nr].legend()
    
def get_losses(df_filename, mode="test", make_plot=False,save_plot=False):
    """
    Get the losses from the result dataframe.
    """

    df = pd.read_csv(df_filename)

    if mode == "test":
        cluster_loss = df[df["Mode"] == "test"]["Cluster"]
        separation_loss = df[df["Mode"] == "test"]["Separation"]
        avg_separation_loss = df[df["Mode"] == "test"]["Avg Separation"]

    elif mode == "train":
        cluster_loss = df[df["Mode"] == "train"]["Cluster"]
        separation_loss = df[df["Mode"] == "train"]["Separation"]
        avg_separation_loss = df[df["Mode"] == "train"]["Avg Separation"]
    else:
        raise ValueError(f"Invalid mode: {mode}. Mode should be either 'test' or 'train'.")

    if make_plot:
        model_name = generate_unimodal_name(df_filename)
        num_epochs = list(range(1, len(cluster_loss)+1))
        plt.figure(figsize=(10,5))
        plt.plot(num_epochs, cluster_loss, label="Cluster Loss")
        plt.plot(num_epochs, separation_loss, label="Separation Loss")
        plt.plot(num_epochs, avg_separation_loss, label="Avg Separation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.xlim(0,70)
        plt.ylim(0,300)
        plt.title(f"{model_name} {mode} Losses")
        plt.legend()
        if save_plot:
            plt.savefig(f"../plots/unimodal/{model_name}_losses.png")
        plt.show()

    return cluster_loss, separation_loss, avg_separation_loss
    
def get_avg_p_pair_dist(df_filename, make_plot=False,save_plot=False):

    """
    Get the average pair distance from the result dataframe.
    """
    df = pd.read_csv(df_filename)

    avg_pair_dist = df[df["Mode"] == 'test']["P Avg Pair Dist"]

    if make_plot:
        model_name = generate_unimodal_name(df_filename)
        num_epochs = list(range(1, len(avg_pair_dist)+1))
        plt.figure(figsize=(10,5))
        plt.plot(num_epochs, avg_pair_dist)
        plt.xlabel("Epoch")
        plt.ylabel("Avg Pair Dist")
        plt.title(f"{model_name} Avg Pair Dist")
        plt.xlim(0,70)
        plt.ylim(0,5000)
        plt.legend()
        if save_plot:
            plt.savefig(f"../plots/unimodal/{model_name}_avg_pair_dist.png")
        plt.show()

    return avg_pair_dist

def print_gpu_memory_status():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # Convert to GB
            allocated_memory = torch.cuda.memory_allocated(i) / 1024**3
            cached_memory = torch.cuda.memory_reserved(i) / 1024**3
            print(f"\nGPU {i} Memory Status:")
            print(f"Total Memory: {total_memory:.2f} GB")
            print(f"Allocated Memory: {allocated_memory:.2f} GB")
            print(f"Cached Memory: {cached_memory:.2f} GB")
            print(f"Free Memory: {total_memory - allocated_memory:.2f} GB")
    else:
        print("No GPU available")

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False