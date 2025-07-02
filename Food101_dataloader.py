# A customized dataloader for Food101

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import random 
import pandas as pd
from collections import Counter, defaultdict

import requests
from PIL import Image
from io import BytesIO
import json 
import time

from transformers import AutoTokenizer

# pylab.rcParams['figure.figsize'] = (8.0, 10.0)

class Food101Dataset(Dataset):
    def __init__(self, data, transform=None, target_transform=None, load_captions=False, fix_label_mapping=True, caption_type="blip2"):

        self.data = data
        self.transform = transform
        self.target_transform = target_transform
        self.load_captions = load_captions
        self.caption_type = caption_type

        self.img_ids = [k for k in data.keys()]
        self.img_filenames = dict(zip(self.img_ids, [data[k]["filepath"] for k in self.img_ids]))
        self.img_labels = dict(zip(self.img_ids, [data[k]["label"] for k in self.img_ids]))

        if self.caption_type == "blip":
            self.img_captions = dict(zip(self.img_ids, [data[k]["blip_caption"] for k in self.img_ids]))
        elif self.caption_type == "blip2":
            self.img_captions = dict(zip(self.img_ids, [data[k]["blip2_caption"] for k in self.img_ids]))
        else:
            raise ValueError(f"Caption type {self.caption_type} not supported.")

        self.dataset_length = len(self.data)
        self.all_labels = list(self.img_labels.values())
        self.unique_labels = list(set(self.img_labels.values()))

        if self.transform == "vgg16" or self.transform == "resnet18":
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])

        if fix_label_mapping:
            with open("Food101/food-101/meta/label_map.json", "r") as f:
                self.label_name_idx = json.load(f)
        else:
            self.label_names = sorted(self.unique_labels)
            self.label_name_idx = dict(zip(self.label_names, range(len(self.label_names))))
        
        if self.target_transform == "one_hot":
            self.target_transform = transforms.Compose([
                transforms.Lambda(lambda x: torch.zeros(len(self.label_names), dtype=torch.float).scatter_(0, torch.tensor(self.label_name_idx[self.img_labels[x]]), value=1))
            ])

        if self.target_transform == "integer":
            self.target_transform =self._convert_label_to_integer

        if self.load_captions:
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.max_length = 128  # Adjust this based on your needs        

    def _convert_label_to_integer(self, label):
        return self.label_name_idx[label]

    def __len__(self):
        return len(self.data)
    
    def get_dataset_labels(self):
        return self.unique_labels
    
    def get_image_id(self, idx):
        return self.img_ids[idx]
    
    def get_image_filename(self, idx):
        return self.img_filenames[self.img_ids[idx]]
    
    def get_image_label(self, idx):
        return self.img_labels[self.img_ids[idx]]
    
    def get_image_caption(self, idx):
        return self.img_captions[self.img_ids[idx]]
    
    def get_image_caption_by_id(self, img_id):
        return self.img_captions[img_id]
    
    def get_num_classes(self):
        return len(self.unique_labels)
    
    def read_image(self, img_id = None, idx = None):

        assert not(img_id is None and idx is None), "To read image, please provide either an img_id or an idx."
        assert not(img_id is not None and idx is not None), "Cannot read image from both img_id and idx, please give only one."

        if img_id:
            img_id = str(img_id)

        if img_id is not None:
            image_filename = self.img_filenames[img_id]
        else:
            image_filename = self.img_filenames[self.img_ids[idx]]
        
        image = Image.open(image_filename).convert('RGB')
        image.show()

    
    def __getitem__(self, idx):
        image_filename = self.img_filenames[self.img_ids[idx]]
        image = Image.open(image_filename).convert('RGB')
        label = self.img_labels[self.img_ids[idx]]

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            if self.target_transform == "integer":
                label = self.label_name_idx[label]
            else:
                label = self.target_transform(label)

        if self.load_captions:
            image_caption = self.get_image_caption(idx)
            
            # Tokenize caption with padding and truncation
            encoding = self.tokenizer(
                image_caption,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt',
                return_token_type_ids=True,  # Important for BERT models
                return_attention_mask=True
            )
            
            # Remove batch dimension that the tokenizer adds
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
            token_type_ids = encoding['token_type_ids'].squeeze(0)
            
            # Return a dictionary with all the necessary components
            return {
                'visual_embeds': image,  # Shape should be (C, H, W)
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'visual_attention_mask': torch.ones(image.shape[0], dtype=torch.long),
                'label': label
            }
    
        else:
            return image, label
        

def lazy_load_customized(load_captions, transform='vgg16', target_transform='integer', fix_label_mapping=True, caption_type="blip2"):

    with open("Food101/food-101/meta/train_full_annotation_customize.json", "r") as f:
        complete_data_train = json.load(f)

    with open("Food101/food-101/meta/val_full_annotation_customize.json", "r") as f:
        complete_data_val = json.load(f)

    with open("Food101/food-101/meta/test_full_annotation_customize.json", "r") as f:
        complete_data_test = json.load(f)

    complete_data_train = {k: v for d in complete_data_train for k, v in d.items()}
    complete_data_val = {k: v for d in complete_data_val for k, v in d.items()}
    complete_data_test = {k: v for d in complete_data_test for k, v in d.items()}

    train_dataset = Food101Dataset(complete_data_train, transform=transform, target_transform=target_transform, fix_label_mapping=fix_label_mapping, load_captions=load_captions, caption_type=caption_type)
    val_dataset = Food101Dataset(complete_data_val, transform=transform, target_transform=target_transform, fix_label_mapping=fix_label_mapping, load_captions=load_captions, caption_type=caption_type)
    test_dataset = Food101Dataset(complete_data_test, transform=transform, target_transform=target_transform, fix_label_mapping=fix_label_mapping, load_captions=load_captions, caption_type=caption_type)

    return train_dataset, val_dataset, test_dataset

def lazy_load_original(load_captions, transform='vgg16', target_transform='integer', fix_label_mapping=True, caption_type="blip2"):
    dataDir = "Food101/food-101/meta/"

    with open(os.path.join(dataDir, "train_full_annotation.json"), "r") as f:
        train_data = json.load(f)
    with open(os.path.join(dataDir, "test_full_annotation.json"), "r") as f:
        test_data = json.load(f)

    complete_data_train = {k: v for d in train_data for k, v in d.items()}
    complete_data_test = {k: v for d in test_data for k, v in d.items()}

    train_dataset = Food101Dataset(complete_data_train, transform=transform, target_transform=target_transform, fix_label_mapping=fix_label_mapping, load_captions=load_captions, caption_type=caption_type)
    test_dataset = Food101Dataset(complete_data_test, transform=transform, target_transform=target_transform, fix_label_mapping=fix_label_mapping, load_captions=load_captions, caption_type=caption_type)

    return train_dataset, test_dataset





