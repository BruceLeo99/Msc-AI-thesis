# MSCOCO_preprocessing_local.py
# This script loads the MSCOCO dataset and preprocesses it from local storage, enabling local training and testing.

# This script used a pre-loaded file 'dataset_info.csv' to map the category name to the supercategory name,
# and to fetch necessary information of the dataset to save computational time.

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


pylab.rcParams['figure.figsize'] = (8.0, 10.0)

from pycocotools.coco import COCO

from sklearn.model_selection import train_test_split

# # Change the working directory to the directory of the script

print(os.getcwd())
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())

# Setup the data directory and file paths
# dataDir='cocoapi/PythonAPI/coco'

dataDir='coco' 
trainDir='train2017' 
valDir='val2017'
testDir='test2017'

dataset_info_df = pd.read_csv('dataset_infos/dataset_info.csv')
category_names = dataset_info_df['Category Name'].unique()

# Create a mask to map the category name to the supercategory name
category_supercategory_map = dict()
for category_name in category_names:
    category_supercategory_map[category_name] = dataset_info_df[dataset_info_df['Category Name'] == category_name]['Supercategory Name'].values[0]

trainInstanceFilepath=f'{dataDir}/annotations/instances_{trainDir}.json'
trainCaptionFilepath=f'{dataDir}/annotations/captions_{trainDir}.json'

valInstanceFilepath=f'{dataDir}/annotations/instances_{valDir}.json'
valCaptionFilepath=f'{dataDir}/annotations/captions_{valDir}.json'

trainImagesFilepath=f'{dataDir}/{trainDir}'
valImagesFilepath=f'{dataDir}/{valDir}'


# Load the instance and caption files
coco_caption_train = COCO(trainCaptionFilepath)
coco_instance_train = COCO(trainInstanceFilepath)

coco_caption_val = COCO(valCaptionFilepath)
coco_instance_val = COCO(valInstanceFilepath)

# Due to the limitation of the MS COCO API, I use the original validation set for testing.
coco_data_path_dict = {
    'train': {
        'caption_path': coco_caption_train,
        'instance_path': coco_instance_train, 
        'images_path': trainImagesFilepath
    },
    'test': {
        'caption_path': coco_caption_val,
        'instance_path': coco_instance_val,
        'images_path': valImagesFilepath
    }
}


def retrieve_captions(img_id, data_type):
    """
    Retrieves the captions for a given image id
    """

    annotations = coco_data_path_dict[data_type]['caption_path'].loadAnns(coco_data_path_dict[data_type]['caption_path'].getAnnIds(imgIds=img_id))

    captions = [ann['caption'] for ann in annotations]

    return " ".join(captions)
    

def load_from_COCOAPI(cat_name, num_instances, data_type, remove_multilabels=False, shuffle=True, verbose=False):
    """
    Loads certain number of images and their information (image id, image url, image filename, captions, category, supercategory)
    from the Microsoft COCO API.

    After loading, the data is stored in a list of dictionaries,
    where each dictionary contains the image id, image_url, captions, filenames, category name, and supercategory name.

    The list of dictionaries will be furthre converted to PyTorch Dataset class. 

    param cat_name: name of the category to load
    param data_type: type of data to load, either 'train', 'val'
       ('test' is not supported due to the unpublished API of MS COCO)   
    param num_instances: number of images
    param shuffle: whether to shuffle the data

    returns list(data_dict) data

    Example data info in the return list
    {
        'img_id': 12345,
        'url': 'http://images.cocodataset.org/train2017/000000012345.jpg',
        'img_filename': 'coco/train2017/000000012345.jpg',
        'captions': 'This is a photo of a cat',
        'category': 'cat',
        'supercategory': 'animal'
    }
    """
    data = []

    # Sort and get the image ids of the category
    categories = coco_data_path_dict[data_type]['instance_path'].dataset['categories']
    cat_ids = [category['id'] for category in categories]
    cat_names = [category['name'] for category in categories]

    cat_dict = dict(zip(cat_names, cat_ids))    

    cat_id = cat_dict[cat_name]
    img_ids = coco_data_path_dict[data_type]['instance_path'].getImgIds(catIds=cat_id)

    if remove_multilabels:
        img_ids = [img_id for img_id in img_ids if len(coco_data_path_dict[data_type]['instance_path'].getAnnIds(imgIds=img_id)) == 1]

    if shuffle:
        random.shuffle(img_ids)

    img_ids = img_ids[:num_instances]

    if verbose:
        print(f"Number of images: {len(img_ids)}")
        print(f"Image ids: {img_ids}")
        
    # Store the image ids, image_url, image_filename, captions in data_dict
    for img_id in img_ids:
        img_filename = coco_data_path_dict[data_type]['instance_path'].loadImgs(img_id)[0]['file_name']
        
        data_info = {
            'img_id': img_id, # E.g.: 12345
            'url': coco_data_path_dict[data_type]['instance_path'].loadImgs(img_id)[0]['coco_url'], # E.g.: 'http://images.cocodataset.org/train2017/000000012345.jpg'
            'img_filename': f"{coco_data_path_dict[data_type]['images_path']}/{img_filename}", # E.g.: 'coco/train2017/000000012345.jpg'
            'captions': retrieve_captions(img_id, data_type), # captions in string
            'category': cat_name, # cateogry name in string
            'supercategory': category_supercategory_map[cat_name] # supercategory name in string
        }

        data.append(data_info)

    return data
	
class MSCOCOCustomDataset(Dataset):
    """
    Creates a customized MS COCO dataset in PyTorch Dataset format that allows PyTorch to load and process the data.

    param data_list: list of dictionaries containing image information
    param transform: optional transform to apply to the images
    param target_transform: optional transform to apply to the labels
    param load_captions: whether to load the captions
    """

    def __init__(self, data_list=None, load_from_json=None, load_from_local=True,
                 transform=None, target_transform=None, load_captions=False):
        
        """
        param data_list: list of dictionaries containing image information
        param load_from_json: path to the json file containing the data
        param transform: optional transform to apply to the images
        param target_transform: optional transform to apply to the labels
        param load_from_local: whether to load the data from local storage
        param load_captions: whether to load the captions
        """

        self.load_from_local = load_from_local
        self.load_captions = load_captions
        self.load_from_json = load_from_json       
        self.transform = transform
        self.target_transform = target_transform

        if any([self.transform, self.target_transform]) is None:
            raise ValueError("You did not provide the transform or target_transform")

        if data_list is not None:
            self.data = data_list
            self.dataset_length = len(self.data)
            self.img_ids = list(set([item['img_id'] for item in self.data]))
            self.img_labels = dict(zip(self.img_ids, [item['category'] for item in self.data]))
            self.img_urls = dict(zip(self.img_ids, [item['url'] for item in self.data]))
            self.img_filenames = dict(zip(self.img_ids, [item['img_filename'] for item in self.data]))
            self.img_captions = dict(zip(self.img_ids, [item['captions'] for item in self.data]))
            self.img_supercategories = dict(zip(self.img_ids, [item['supercategory'] for item in self.data]))

        elif self.load_from_json is not None:
            if not self.load_from_json.endswith(".json"):
                raise ValueError("The load_from_json path must be a JSON file ending with .json")

            with open(f"{self.load_from_json}", "r") as f:
                data_info = json.load(f)
                self.data = data_info
                self.dataset_length = len(data_info['img_ids'])
                self.img_ids = list(set(data_info['img_ids']))
                self.img_labels = data_info['img_labels']
                self.img_urls = data_info['img_urls']
                self.img_filenames = data_info['img_filenames']
                self.img_captions = data_info['img_captions']
                self.img_supercategories = data_info['img_supercategories']

                # JSON converts the image ids to string when using as keys of other dataset ingo,
                # So we need to convert the self.img_ids to string if we load data from JSON.
                # If load from elsewhere with COCO original API, there is no need to convert.
                self.img_ids = [str(id) for id in self.img_ids]

        else:
            raise ValueError("You did not provide the data source. Either data_list or load_from_json must be provided")


        # Define transformation modes for different backbone encoders
        # VGG16 or ResNet18 transformation
        if self.transform == "vgg16" or self.transform == "resnet18":
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])

        # Define the transformation mode for the labels
        self.label_names = list(set(self.img_labels.values()))
        self.label_name_idx = {name: idx for idx, name in enumerate(self.label_names)}
        
        if self.target_transform == "one_hot":
            self.target_transform = transforms.Compose([
                transforms.Lambda(lambda x: torch.zeros(len(self.label_names), dtype=torch.float).scatter_(0, torch.tensor(self.label_name_idx[self.img_labels[x]]), value=1))
            ])

        if self.target_transform == "integer":
            self.target_transform = self._convert_label_to_integer

    def _convert_label_to_integer(self, label):
        return self.label_name_idx[label]

    def __len__(self):
        return self.dataset_length
        
    def get_dataset_labels(self):
        return self.label_name_idx
    
    def get_image_id(self, idx):
        return self.img_ids[idx]
    
    def get_image_url(self, idx):
        return self.img_urls[self.img_ids[idx]]
    
    def get_image_label(self, idx):
        return self.img_labels[self.img_ids[idx]]
    
    def get_image_caption(self, idx):
        return self.img_captions[self.img_ids[idx]]
    
    def get_image_supercategory(self, idx):
        return self.img_supercategories[self.img_ids[idx]]
    
    def get_num_classes(self):
        return len(self.label_names)
    
    def get_image_filename(self, idx):
        return self.img_filenames[self.img_ids[idx]]
    
    def read_image_from_local(self, image_id):
        """
        Reads an image from a given image_id
        """
        # Use direct dictionary access with image_id
        if image_id not in self.img_ids:
            raise KeyError(f"Image ID {image_id} not found in dataset")
        
        image_filename = self.img_filenames[image_id]
        try:
            return Image.open(image_filename).convert('RGB')
        
        except (FileNotFoundError, IOError) as e:
            raise FileNotFoundError(f"Could not load image {image_filename}: {e}")
        

    def read_image_from_url(self,image_url):
        """
        Reads an image from a given URL with retry logic
        """
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()  # Raise an exception for bad status codes
                img = Image.open(BytesIO(response.content)).convert('RGB')
                return img
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Failed to load image from {image_url}, attempt {attempt + 1}/{max_retries}. Retrying...")
                    time.sleep(1)  # Wait before retry
                else:
                    print(f"Failed to load image from {image_url} after {max_retries} attempts: {e}")
                
    
    def __getitem__(self, idx):
        """ 
        Retrieves an image and its corresponding label, also the caption if it is enabled.
        """
        if idx >= self.dataset_length or idx < 0:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.dataset_length}")
            
        image_id = self.img_ids[idx]
        if self.load_from_local:
            image = self.read_image_from_local(image_id)
        else:
            image = self.read_image_from_url(self.img_urls[image_id])

        image_label = self.img_labels[image_id]

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            if self.target_transform == "integer":
                image_label = self.label_name_idx[image_label]
            else:
                image_label = self.target_transform(image_label)
            
        if self.load_captions:
            image_caption = self.img_captions[image_id]
            # print("Returning 3 items")
            return image, image_label, image_caption
        
        else:
            # print("Returning 2 items")
            return image, image_label
        

def save_data_to_json(dataset, dataset_name):
    data_info = {}
    data_info['img_ids'] = dataset.img_ids
    data_info['img_labels'] = dataset.img_labels
    data_info['img_urls'] = dataset.img_urls
    data_info['img_filenames'] = dataset.img_filenames
    data_info['img_captions'] = dataset.img_captions
    data_info['img_supercategories'] = dataset.img_supercategories
    with open(f"{dataset_name}.json", "w") as f:
        json.dump(data_info, f)
    
    print(f"Data saved to {dataset_name}.json")        

def prepare_data_manually(*categories, num_instances=100, load_from_local=True, for_test=False, split_val=False, val_size=None, experiment_name=None,
                 transform=None, target_transform=None, load_captions=False, save_result=False):
    """
    Prepares the data for the given categories with a manually defined number of instances.
    Loads data for each category, splits each category separately to maintain class balance,
    and returns two MSCOCOCustomDataset objects.

    param categories: list of categories to load
    param num_instances: number of instances to load per category
    param load_from_local: whether to load the data from local storage
    param for_test: whether to load the data for testing
    param split_val: whether to split the training set into training and validation sets
    param val_size: size of the validation set (as a fraction of the training set, 0.2 by default)
    param transform: transform to apply to the images
    param load_captions: whether to load the captions

    returns train_data, test_data in MSCOCOCustomDataset format
    """
    train_list = []
    val_list = []
    test_list = []

    if split_val:
        assert val_size is not None, "val_size must be provided if split_val is True"

    if not for_test:
        for category in categories:
            # Load data for this category
            category_data = load_from_COCOAPI(category, num_instances, data_type='train')
            
            if split_val:
            # Split this category's data
                category_train, category_val = train_test_split(
                category_data, 
                test_size=val_size, 
                random_state=42
            )
        
            
                train_list.extend(category_train)
                val_list.extend(category_val)

            else:
                train_list.extend(category_data)
    
            # Create datasets
            train_data = MSCOCOCustomDataset(train_list, load_from_local=load_from_local, transform=transform, 
                                                target_transform=target_transform, 
                                                load_captions=load_captions)
        
        if split_val:
            val_data = MSCOCOCustomDataset(val_list, load_from_local=load_from_local, transform=transform, 
                                    target_transform=target_transform, 
                                    load_captions=load_captions)
            return train_data, val_data
        
        # Only returns the training data if there is no need to split a test set from it
        else:
            return train_data
        

    else:
        for category in categories:
            category_data = load_from_COCOAPI(category, num_instances, data_type='test')
            test_list.extend(category_data)

    test_data = MSCOCOCustomDataset(test_list, load_from_local=load_from_local, transform=transform, 
                                   target_transform=target_transform, 
                                   load_captions=load_captions)
    return test_data
    

def prepare_data_from_preselected_categories(selection_csv, data_type, load_from_local=True, split_val=False, val_size=0.2, experiment_name=None,
                 transform=None, target_transform=None, load_captions=False, save_result=False, load_from_json=None):
    
    """
    Prepare the data from a list of preselected categories, which is preloaded in a csv file.
    See choose_categories.py for more details.

    Due to the limitation of the MS COCO API, I split val_size amount of data from the training set for validation,
    and use the original validation set for testing.

    param str selection_csv: path to the csv file containing the preselected categories
    param str data_type: type of data to load, either 'train' or 'test'
    param bool load_from_local: whether to load the data from local storage
    param bool split_val: whether to split the training set into training and validation sets
    param float val_size: size of the validation set (as a fraction of the training set, 0.2 by default)
    param str transform: transform to apply to the images. Choose from 'vgg16', 'resnet18'
    param str target_transform: transform to apply to the labels. Choose from 'one_hot', 'integer'
    param bool load_captions: whether to load the captions
    param bool save_result: whether to save the result
    param str load_from_json: path to the json file containing the data

    returns:
    1. train_data: MSCOCOCustomDataset format if data_type is 'train' and split_val is False, containing data only for training
    2. train_data, val_data: MSCOCOCustomDataset format if split_val is True, otherwise only train_data
    3. test_data: MSCOCOCustomDataset format if data_type is 'test'
    """
    train_list = []
    val_list = []
    test_list = []
    
    df = pd.read_csv(selection_csv)
    categories = df['Category Name'].tolist()

    for category in categories:
        # Load data for this category
        if data_type == 'train':
            num_instances = df[df['Category Name'] == category]['Number of Training Images'].values[0]
            category_data = load_from_COCOAPI(category, num_instances, data_type, shuffle=True)

            if split_val:
                train, val = train_test_split(category_data, test_size=val_size, random_state=42)
                train_list.extend(train)
                val_list.extend(val)
            else:
                train_list.extend(category_data)
        
        
        elif data_type == 'test':
            num_instances = df[df['Category Name'] == category]['Number of Validation Images'].values[0]
            category_data = load_from_COCOAPI(category, num_instances, data_type, shuffle=False)
            test_list.extend(category_data)
        
    
    # Create datasets
    # If we need to split the training set into training and validation set, we return both datasets
    # If not, then only the training
    if data_type == 'train':
        train_data = MSCOCOCustomDataset(train_list, load_from_local=load_from_local, transform=transform, 
                                    target_transform=target_transform, 
                                    load_from_json=load_from_json,
                                    load_captions=load_captions)
    
        if split_val:
            val_data = MSCOCOCustomDataset(val_list, load_from_local=load_from_local, transform=transform, 
                                    target_transform=target_transform, 
                                    load_from_json=load_from_json,
                                    load_captions=load_captions)
            return train_data, val_data
        

        else:
            return train_data
    
    # If we need to load the test set, we return the test dataset
    elif data_type == 'test':
        test_data = MSCOCOCustomDataset(test_list, load_from_local=load_from_local, transform=transform, 
                                    target_transform=target_transform, 
                                    load_captions=load_captions)
        return test_data

def data_summary(experiment_name, train_data, val_data, test_data, num_examples=0, verbose=False, save_result=False):
    """
    Summarizes the data by printing the number of images in each category.
    """

    df_dataset_info = pd.read_csv('dataset_infos/dataset_info.csv')
    category_names = list(set([data['category'] for data in train_data.data]))
    supercategory_names = [category_supercategory_map[category] for category in category_names] 
    
    # Counts the number of images per category in per set
    num_imgs_per_set = {
        'train': Counter([data['category'] for data in train_data.data]),
        'val': Counter([data['category'] for data in val_data.data]),
        'test': Counter([data['category'] for data in test_data.data])
    }

    dataset_stats = pd.DataFrame({
        'Supercategory Name': supercategory_names,
        'Category Name': category_names,
        'Number of Training Images': [num_imgs_per_set['train'][category] for category in category_names],
        'Number of Validation Images': [num_imgs_per_set['val'][category] for category in category_names],
        'Number of Test Images': [num_imgs_per_set['test'][category] for category in category_names],
        'Total': [sum([num_imgs_per_set['train'][category], num_imgs_per_set['val'][category], num_imgs_per_set['test'][category]]) for category in category_names]
    })

    if save_result:
        print("Saving dataset stats...")
        dataset_stats.to_csv(f'dataset_stats_{experiment_name}.csv', index=False)

    if verbose:
        print(dataset_stats)

    return dataset_stats

def check_data_leakage(train_data, val_data, test_data, verbose=True, save_result=False):
    """
    Check for data leakage between train, validation, and test datasets.
    
    Data leakage occurs when the same image appears in multiple splits,
    which can lead to overly optimistic performance estimates.
    
    Args:
        train_data: Training dataset (MSCOCOCustomDataset)
        val_data: Validation dataset (MSCOCOCustomDataset) 
        test_data: Test dataset (MSCOCOCustomDataset)
        verbose: Whether to print detailed information
        save_result: Whether to save the result
    Returns:
        dict: Summary of leakage detection results
    """
    
    # Extract image IDs from each dataset
    train_ids = set(train_data.img_ids)
    val_ids = set(val_data.img_ids) if val_data is not None else set()
    test_ids = set(test_data.img_ids)
    
    # Check for overlaps
    train_val_overlap = train_ids.intersection(val_ids)
    train_test_overlap = train_ids.intersection(test_ids)
    val_test_overlap = val_ids.intersection(test_ids)
    
    # Check for any overlap across all three sets
    all_overlap = train_ids.intersection(val_ids).intersection(test_ids)
    
    # Calculate statistics
    total_unique_images = len(train_ids.union(val_ids).union(test_ids))
    expected_total = len(train_ids) + len(val_ids) + len(test_ids)
    
    leakage_results = {
        'train_size': len(train_ids),
        'val_size': len(val_ids),
        'test_size': len(test_ids),
        'train_val_overlap': len(train_val_overlap),
        'train_test_overlap': len(train_test_overlap),
        'val_test_overlap': len(val_test_overlap),
        'all_three_overlap': len(all_overlap),
        'total_unique_images': total_unique_images,
        'expected_total': expected_total,
        'has_leakage': len(train_val_overlap) > 0 or len(train_test_overlap) > 0 or len(val_test_overlap) > 0,
        'train_val_overlap_ids': list(train_val_overlap),
        'train_test_overlap_ids': list(train_test_overlap),
        'val_test_overlap_ids': list(val_test_overlap),
        'all_three_overlap_ids': list(all_overlap)
    }
    
    if verbose:
        print("="*60)
        print("DATA LEAKAGE DETECTION REPORT")
        print("="*60)
        
        print(f"\nDataset Sizes:")
        print(f"Training:   {len(train_ids):,} images")
        print(f"Validation: {len(val_ids):,} images")
        print(f"Test:       {len(test_ids):,} images")
        print(f"Expected Total: {expected_total:,} images")
        print(f"Actual Unique:  {total_unique_images:,} images")
        
        print(f"\nOverlap Analysis:")
        print(f"Train ∩ Validation: {len(train_val_overlap):,} images")
        print(f"Train ∩ Test:       {len(train_test_overlap):,} images")
        print(f"Validation ∩ Test:  {len(val_test_overlap):,} images")
        print(f"All Three Sets:     {len(all_overlap):,} images")
        
        # Calculate leakage percentages
        if len(val_ids) > 0:
            train_val_pct = (len(train_val_overlap) / len(val_ids)) * 100
            print(f"  Train→Val Leakage:  {train_val_pct:.2f}% of validation data")
        
        test_train_pct = (len(train_test_overlap) / len(test_ids)) * 100
        print(f"  Train→Test Leakage: {test_train_pct:.2f}% of test data")
        
        if len(val_ids) > 0:
            test_val_pct = (len(val_test_overlap) / len(test_ids)) * 100
            print(f"  Val→Test Leakage:   {test_val_pct:.2f}% of test data")
        
        # Overall assessment
        print(f"\nOverall Assessment:")
        if leakage_results['has_leakage']:
            print("DATA LEAKAGE DETECTED!")
            print("This may lead to overly optimistic performance estimates.")
            print("Consider using different data splits or removing duplicates.")
        else:
            print("NO DATA LEAKAGE DETECTED")
            print("All splits are properly separated.")
        
        # Show some example overlapping IDs if they exist
        if len(train_val_overlap) > 0:
            print(f"\nExample Train-Val Overlap IDs (showing first 5):")
            for img_id in list(train_val_overlap)[:5]:
                print(f"Image ID: {img_id}")
        
        if len(train_test_overlap) > 0:
            print(f"\nExample Train-Test Overlap IDs (showing first 5):")
            for img_id in list(train_test_overlap)[:5]:
                print(f"Image ID: {img_id}")
                
        if len(val_test_overlap) > 0:
            print(f"\nExample Val-Test Overlap IDs (showing first 5):")
            for img_id in list(val_test_overlap)[:5]:
                print(f"Image ID: {img_id}")
        
        print("="*60)
    
    return leakage_results

def eliminate_leaked_data(experiment_name, train_data, val_data, test_data, verbose=False, save_result=False):
    """
    Eliminate data that is present in multiple datasets.
    """
    leakage_results = check_data_leakage(train_data, val_data, test_data, verbose=verbose)

    # If there is no leakage at all, then return the original datasets
    if not leakage_results['has_leakage']:
        return train_data, val_data, test_data
    

    num_leaked_train_val = len(leakage_results['train_val_overlap_ids'])
    num_leaked_train_test = len(leakage_results['train_test_overlap_ids'])
    num_leaked_val_test = len(leakage_results['val_test_overlap_ids'])

    total_leaked_images = num_leaked_train_val + num_leaked_train_test + num_leaked_val_test

    if verbose:
        print("Data summary before elimination:")

    data_stats_before_elimination = data_summary(experiment_name, train_data, val_data, test_data, verbose=verbose, save_result=save_result)

    if leakage_results['has_leakage']:
        print("Eliminating leaked train and val data from training set...")
        leaked_datapoints = [datapoint for datapoint in train_data.data if datapoint['img_id'] in leakage_results['train_val_overlap_ids']]
        for datapoint in leaked_datapoints:
            train_data.data.remove(datapoint)

    if leakage_results['has_leakage']:
        print("Eliminating leaked train and test data from training set...")
        leaked_datapoints = [datapoint for datapoint in train_data.data if datapoint['img_id'] in leakage_results['train_test_overlap_ids']]
        for datapoint in leaked_datapoints:
            train_data.data.remove(datapoint)

    if leakage_results['has_leakage']:
        print("Eliminating leaked val and test data from validation set...")
        leaked_datapoints = [datapoint for datapoint in val_data.data if datapoint['img_id'] in leakage_results['val_test_overlap_ids']]
        for datapoint in leaked_datapoints:
            val_data.data.remove(datapoint)

    data_stats_after_elimination = data_summary(experiment_name, train_data, val_data, test_data, verbose=False, save_result=False)

    data_stats_complete = pd.DataFrame({
        'Supercategory Name': data_stats_before_elimination['Supercategory Name'],
        'Category Name': data_stats_before_elimination['Category Name'],
        'Number of Training Images Before': data_stats_before_elimination['Number of Training Images'],
        'Number of Validation Images Before': data_stats_before_elimination['Number of Validation Images'],
        'Number of Test Images Before': data_stats_before_elimination['Number of Test Images'],
        'Total Before': data_stats_before_elimination['Total'],
        'Number of Training Images After': data_stats_after_elimination['Number of Training Images'],
        'Number of Validation Images After': data_stats_after_elimination['Number of Validation Images'],
        'Number of Test Images After': data_stats_after_elimination['Number of Test Images'],
        'Total After': data_stats_after_elimination['Total'],
        'Number of Train Eliminated': data_stats_before_elimination['Number of Training Images'] - data_stats_after_elimination['Number of Training Images'],
        'Number of Val Eliminated': data_stats_before_elimination['Number of Validation Images'] - data_stats_after_elimination['Number of Validation Images'],
        'Number of Test Eliminated': data_stats_before_elimination['Number of Test Images'] - data_stats_after_elimination['Number of Test Images'],
        })

    # Insert a row of sum of all categories
    data_stats_complete.loc[len(data_stats_complete)] = ["Total", "Nah",
        data_stats_before_elimination['Number of Training Images'].sum(),
        data_stats_before_elimination['Number of Validation Images'].sum(),
        data_stats_before_elimination['Number of Test Images'].sum(),
        data_stats_before_elimination['Total'].sum(),
        data_stats_after_elimination['Number of Training Images'].sum(),
        data_stats_after_elimination['Number of Validation Images'].sum(),
        data_stats_after_elimination['Number of Test Images'].sum(),
        data_stats_after_elimination['Total'].sum(),
        data_stats_before_elimination['Number of Training Images'].sum() - data_stats_after_elimination['Number of Training Images'].sum(),
        data_stats_before_elimination['Number of Validation Images'].sum() - data_stats_after_elimination['Number of Validation Images'].sum(),
        data_stats_before_elimination['Number of Test Images'].sum() - data_stats_after_elimination['Number of Test Images'].sum()]
    
    # Incert a column of train-val-test ration after elimination

    data_stats_complete.insert(len(data_stats_complete.columns), 'Prior Train Ratio', 
                              data_stats_complete['Number of Training Images Before'] / data_stats_complete['Total Before'])
    
    data_stats_complete.insert(len(data_stats_complete.columns), 'Final Train Ratio', 
                              data_stats_complete['Number of Training Images After'] / data_stats_complete['Total After'])
    
    data_stats_complete.insert(len(data_stats_complete.columns), 'Prior Val Ratio',
                            data_stats_complete['Number of Validation Images Before'] / data_stats_complete['Total Before'])

    data_stats_complete.insert(len(data_stats_complete.columns), 'Final Val Ratio',
                            data_stats_complete['Number of Validation Images After'] / data_stats_complete['Total After'])
    
    data_stats_complete.insert(len(data_stats_complete.columns), 'Prior Test Ratio',
                            data_stats_complete['Number of Test Images Before'] / data_stats_complete['Total Before'])
    
    data_stats_complete.insert(len(data_stats_complete.columns), 'Final Test Ratio',
                            data_stats_complete['Number of Test Images After'] / data_stats_complete['Total After'])
    
    if verbose:
        print("Leaked data eliminated.")
        print(f"{num_leaked_train_test} leaked images in train and test. Removed from training set.")
        print(f"{num_leaked_val_test} leaked images in val and test. Removed from validation set.")
        print(f"Total leaked images: {total_leaked_images}")
        print("Data summary afte leakage elimination:")
        print(data_stats_complete)

    # Save result of dataset: before and after elimination
    if save_result:
        print("Saving dataset stats after elimination...")
        data_stats_complete.to_csv(f'dataset_stats_{experiment_name}.csv', index=False)

    return train_data, val_data, test_data



def check_category_distribution(experiment_name, train_data, val_data, test_data, verbose=True, save_result=False):
    """
    Check the distribution of categories across train, validation, and test sets.
    Also checks for overlaps between assigned classes (multi-label scenarios).
    
    Args:
        experiment_name: Name of the experiment
        train_data: Training dataset (MSCOCOCustomDataset)
        val_data: Validation dataset (MSCOCOCustomDataset)
        test_data: Test dataset (MSCOCOCustomDataset)
        verbose: Whether to print detailed information
        save_result: Whether to save the result
    Returns:
        dict: Category distribution statistics and overlap analysis
    """
    
    # Get category distributions
    train_categories = Counter(train_data.img_labels.values())
    val_categories = Counter(val_data.img_labels.values()) if val_data is not None else Counter()
    test_categories = Counter(test_data.img_labels.values())
    
    # Get all unique categories
    all_categories = set(train_categories.keys()) | set(val_categories.keys()) | set(test_categories.keys())
    
    # Create distribution dataframe
    distribution_data = []
    for category in sorted(all_categories):
        train_count = train_categories.get(category, 0)
        val_count = val_categories.get(category, 0)
        test_count = test_categories.get(category, 0)
        total_count = train_count + val_count + test_count
        
        if total_count > 0:
            train_pct = (train_count / total_count) * 100
            val_pct = (val_count / total_count) * 100
            test_pct = (test_count / total_count) * 100
        else:
            train_pct = val_pct = test_pct = 0
            
        distribution_data.append({
            'category': category,
            'train_count': train_count,
            'val_count': val_count,
            'test_count': test_count,
            'total_count': total_count,
            'train_pct': train_pct,
            'val_pct': val_pct,
            'test_pct': test_pct
        })
    
    df = pd.DataFrame(distribution_data)
    
    # Check for class overlaps/co-occurrence in images
    def analyze_class_overlaps(dataset, dataset_name):
        """Analyze if the same image has multiple class labels assigned"""
        image_to_categories = defaultdict(list)
        
        # Group categories by image ID
        for img_id, category in dataset.img_labels.items():
            image_to_categories[img_id].append(category)
        
        # Find images with multiple categories
        multi_label_images = {img_id: cats for img_id, cats in image_to_categories.items() if len(cats) > 1}
        
        # Count co-occurrences between categories
        category_pairs = defaultdict(int)
        for img_id, categories in multi_label_images.items():
            for i, cat1 in enumerate(categories):
                for cat2 in categories[i+1:]:
                    pair = tuple(sorted([cat1, cat2]))
                    category_pairs[pair] += 1
        
        return {
            'total_images': len(image_to_categories),
            'multi_label_images': len(multi_label_images),
            'multi_label_percentage': (len(multi_label_images) / len(image_to_categories)) * 100 if len(image_to_categories) > 0 else 0,
            'category_pairs': dict(category_pairs),
            'multi_label_examples': dict(list(multi_label_images.items())[:5])  # First 5 examples
        }
    
    # Analyze overlaps for each dataset
    train_overlaps = analyze_class_overlaps(train_data, "Training")
    val_overlaps = analyze_class_overlaps(val_data, "Validation") if val_data is not None else {
        'total_images': 0, 'multi_label_images': 0, 'multi_label_percentage': 0, 
        'category_pairs': {}, 'multi_label_examples': {}
    }
    test_overlaps = analyze_class_overlaps(test_data, "Test")
    
    # Combine all category pairs to get overall co-occurrence statistics
    all_category_pairs = defaultdict(int)
    for pairs_dict in [train_overlaps['category_pairs'], val_overlaps['category_pairs'], test_overlaps['category_pairs']]:
        for pair, count in pairs_dict.items():
            all_category_pairs[pair] += count
    
    # Sort category pairs by frequency
    sorted_pairs = sorted(all_category_pairs.items(), key=lambda x: x[1], reverse=True)
    
    if verbose:
        print("="*80)
        print("CATEGORY DISTRIBUTION & OVERLAP REPORT")
        print("="*80)
        
        print(f"\nTotal Categories: {len(all_categories)}")
        print(f"Training Images: {sum(train_categories.values()):,}")
        print(f"Validation Images: {sum(val_categories.values()):,}")
        print(f"Test Images: {sum(test_categories.values()):,}")
        
        print(f"\nDetailed Distribution:")
        print(df.to_string(index=False, float_format='%.1f'))
        
        # Check for missing categories in any split
        missing_in_train = [cat for cat in all_categories if train_categories.get(cat, 0) == 0]
        missing_in_val = [cat for cat in all_categories if val_categories.get(cat, 0) == 0]
        missing_in_test = [cat for cat in all_categories if test_categories.get(cat, 0) == 0]
        
        if missing_in_train:
            print(f"\nCategories missing in TRAINING: {missing_in_train}")
        if missing_in_val and val_data is not None:
            print(f"Categories missing in VALIDATION: {missing_in_val}")
        if missing_in_test:
            print(f"Categories missing in TEST: {missing_in_test}")
            
        if not missing_in_train and not missing_in_val and not missing_in_test:
            print(f"\nAll categories present in all splits")
        
        # Report class overlap analysis
        print(f"\n" + "="*60)
        print("CLASS OVERLAP ANALYSIS")
        print("="*60)
        
        print(f"\nMulti-label Image Statistics:")
        print(f"  Training Set:")
        print(f"    Total Images: {train_overlaps['total_images']:,}")
        print(f"    Multi-label Images: {train_overlaps['multi_label_images']:,} ({train_overlaps['multi_label_percentage']:.2f}%)")
        
        if val_data is not None:
            print(f"  Validation Set:")
            print(f"    Total Images: {val_overlaps['total_images']:,}")
            print(f"    Multi-label Images: {val_overlaps['multi_label_images']:,} ({val_overlaps['multi_label_percentage']:.2f}%)")
        
        print(f"  Test Set:")
        print(f"    Total Images: {test_overlaps['total_images']:,}")
        print(f"    Multi-label Images: {test_overlaps['multi_label_images']:,} ({test_overlaps['multi_label_percentage']:.2f}%)")
        
        # Show most common category co-occurrences
        if sorted_pairs:
            print(f"\nMost Common Category Co-occurrences:")
            for i, ((cat1, cat2), count) in enumerate(sorted_pairs[:10]):
                print(f"  {i+1}. {cat1} + {cat2}: {count} images")
        else:
            print(f"\nNo category co-occurrences detected")
            print(f"   Each image has exactly one category assigned")
        
        # Show examples of multi-label images
        if train_overlaps['multi_label_examples']:
            print(f"\nExample Multi-label Images (Training Set):")
            for img_id, categories in list(train_overlaps['multi_label_examples'].items())[:3]:
                print(f"  Image {img_id}: {', '.join(categories)}")
        
        # Assessment
        total_multi_label = train_overlaps['multi_label_images'] + val_overlaps['multi_label_images'] + test_overlaps['multi_label_images']
        if total_multi_label > 0:
            print(f"\nMULTI-LABEL SCENARIO DETECTED!")
            print(f"   {total_multi_label} images have multiple category assignments")
            print(f"   Consider using multi-label classification approaches")
            print(f"   Current single-label approach may not capture all information")
        else:
            print(f"\nSINGLE-LABEL SCENARIO CONFIRMED")
            print(f"   Each image has exactly one category assigned")
            print(f"   Single-label classification approach is appropriate")
            
        print("="*80)

    assessment_result = {
        'distribution_df': df,
        'missing_in_train': missing_in_train,
        'missing_in_val': missing_in_val,
        'missing_in_test': missing_in_test,
        'total_categories': len(all_categories),
        'train_overlaps': train_overlaps,
        'val_overlaps': val_overlaps,
        'test_overlaps': test_overlaps,
        'category_co_occurrences': dict(sorted_pairs),
        'is_multi_label': total_multi_label > 0,
        'total_multi_label_images': total_multi_label
    }
    
    return assessment_result

if __name__ == '__main__':

    #########################################################################################
    # Example usage of loading data from a single category and showing an example image
    #########################################################################################

    ### Load data that stores as a list of annotations
    # data_dog = load_from_COCOAPI('dog', 100, 'train')
    # for dog in data_dog:
    #     print(dog['img_id'])
    #     print(dog['captions'])
    #     ... 

    # show_image(<an image ID that is known to exist>, 'train')

    ### Load data that stores as a torch.utils.data.Dataset
    # data_dog = MSCOCOCustomDataset(load_from_COCOAPI('dog', 100, 'train'), load_captions=True)
    # data_cat = MSCOCOCustomDataset(load_from_COCOAPI('cat', 100, 'train'))
    # data_zebra = MSCOCOCustomDataset(load_from_COCOAPI('zebra', 100, 'train'), load_captions=True)
    # data_giraffe = MSCOCOCustomDataset(load_from_COCOAPI('giraffe', 100, 'train'))
    # data_horse = MSCOCOCustomDataset(load_from_COCOAPI('horse', 100, 'test'))
    # data_airplane = MSCOCOCustomDataset(load_from_COCOAPI('airplane', 100, 'test'), load_captions=True)

    # print(f"filenames of dog images: {data_dog.img_filenames}")
    # print(f"Ids of dog images: {data_dog.img_ids}")

    # image, label, captions = data_dog[5] 

    # print("Label:", label)
    # print("Captions:", captions)
    # image.show() 

    # print(f"Number of images: {len(data_airplane)}")
    # print(data_airplane.img_ids)

    # image, label, captions = data_airplane[7]
    # print("Label:", label)
    # print("Captions:", captions)
    # image.show() 

    #########################################################################################
    # Example usage of loading data in one go and test loading speed
    #########################################################################################
    
    # experiment_name = 'testrun_local'

    # start_time = time.time()
    # train_data, val_data = prepare_data_from_preselected_categories('chosen_categories_3_10_v3.csv', 'train', split_val=True, val_size=0.2, transform='vgg16', target_transform='integer', load_captions=True)
    # test_data = prepare_data_from_preselected_categories('chosen_categories_3_10_v3.csv', 'test', transform='vgg16', target_transform='integer', load_captions=True)

    # train_data, val_data, test_data = eliminate_leaked_data(experiment_name, train_data, val_data, test_data, verbose=True, save_result=True)
    # check_category_distribution(experiment_name, train_data, val_data, test_data, save_result=True)

    # end_time = time.time()
    # print(f"Time taken to prepare data: {end_time - start_time} seconds")

    #####################################################################
    # LOAD CLEAN NON-LEAKED DATA TO JSON FOR FAST RELOAD
    #####################################################################

    # To load singleLabel data, use the following:
    # chosen_dataset_10classes = "singleLabel_chosen_categories_3_10.csv"
    # chosen_dataset_20classes = "dataset_infos/singleLabel_chosen_categories_6_20.csv"
    # chosen_dataset_30classes = "dataset_infos/singleLabel_chosen_categories_7_30.csv"

    chosen_dataset_10classes = "chosen_categories_3_10.csv"
    chosen_dataset_20classes = "chosen_categories_6_20.csv"
    chosen_dataset_30classes = "chosen_categories_10_30.csv"

    save_result = True

    chosen_dataset_df_10classes = pd.read_csv(f"dataset_infos/{chosen_dataset_10classes}")
    chosen_dataset_categories_10classes = chosen_dataset_df_10classes['Category Name'].unique().tolist()

    chosen_dataset_df_20classes = pd.read_csv(f"dataset_infos/{chosen_dataset_20classes}")
    chosen_dataset_categories_20classes = chosen_dataset_df_20classes['Category Name'].unique().tolist()

    chosen_dataset_df_30classes = pd.read_csv(f"dataset_infos/{chosen_dataset_30classes}")
    chosen_dataset_categories_30classes = chosen_dataset_df_30classes['Category Name'].unique().tolist()

    ### 10 classes dataset
    train_data_10classes, val_data_10classes = prepare_data_from_preselected_categories(
        f"dataset_infos/{chosen_dataset_10classes}",
        "train",
        load_from_local=True,
        split_val=True,
        val_size=0.2,
        experiment_name=chosen_dataset_10classes,
        transform='vgg16',
        target_transform='integer',
        save_result=save_result,
    )

    test_data_10classes = prepare_data_from_preselected_categories(
        f"dataset_infos/{chosen_dataset_10classes}",
        "test",
        load_from_local=True,
        split_val=False,
        experiment_name=chosen_dataset_10classes,
        transform='vgg16',
        target_transform='integer',
        save_result=save_result,
    )

    train_data_10classes, val_data_10classes, test_data_10classes = eliminate_leaked_data(
        chosen_dataset_10classes,
        train_data_10classes,
        val_data_10classes,
        test_data_10classes,
        save_result=save_result,
    )

    ### 20 classes dataset
    train_data_20classes, val_data_20classes = prepare_data_from_preselected_categories(
        f"dataset_infos/{chosen_dataset_20classes}",
        "train",
        load_from_local=True,
        split_val=True,
        val_size=0.2,
        experiment_name=chosen_dataset_20classes,
        transform='vgg16',
        target_transform='integer',
        save_result=save_result
    )

    test_data_20classes = prepare_data_from_preselected_categories(
        f"dataset_infos/{chosen_dataset_20classes}",
        "test",
        load_from_local=True,
        split_val=False,
        experiment_name=chosen_dataset_20classes,
        transform='vgg16',
        target_transform='integer',
        save_result=save_result,
    )

    train_data_20classes, val_data_20classes, test_data_20classes = eliminate_leaked_data(
        chosen_dataset_20classes,
        train_data_20classes,
        val_data_20classes,
        test_data_20classes,
        save_result=save_result,
    )

    ### 30 classes dataset
    train_data_30classes, val_data_30classes = prepare_data_from_preselected_categories(
        f"dataset_infos/{chosen_dataset_30classes}",
        "train",
        load_from_local=True,
        split_val=True,
        val_size=0.2,
        experiment_name=chosen_dataset_30classes,
        save_result=save_result,
        transform='vgg16',
        target_transform='integer'
    )

    test_data_30classes = prepare_data_from_preselected_categories(
        f"dataset_infos/{chosen_dataset_30classes}",
        "test",
        load_from_local=True,
        split_val=False,
        experiment_name=chosen_dataset_30classes,
        transform='vgg16',
        target_transform='integer',
        save_result=save_result
    )

    train_data_30classes, val_data_30classes, test_data_30classes = eliminate_leaked_data(
        chosen_dataset_30classes,
        train_data_30classes,
        val_data_30classes,
        test_data_30classes,
        save_result=save_result
    )
    

    save_data_to_json(train_data_10classes, "train_data_10classes")
    save_data_to_json(val_data_10classes, "val_data_10classes")
    save_data_to_json(test_data_10classes, "test_data_10classes")
    save_data_to_json(train_data_20classes, "train_data_20classes")
    save_data_to_json(val_data_20classes, "val_data_20classes")
    save_data_to_json(test_data_20classes, "test_data_20classes")
    save_data_to_json(train_data_30classes, "train_data_30classes")
    save_data_to_json(val_data_30classes, "val_data_30classes")
    save_data_to_json(test_data_30classes, "test_data_30classes")


    # Test if data is probably loaded to JSON and can be reloaded
    print("Reload test")

    start_time = time.time()
    train_data_10classes_json = MSCOCOCustomDataset(transform='vgg16', target_transform='integer', load_from_json="train_data_10classes.json")
    val_data_10classes_json = MSCOCOCustomDataset(transform='vgg16', target_transform='integer', load_from_json="val_data_10classes.json")
    test_data_10classes_json = MSCOCOCustomDataset(transform='vgg16', target_transform='integer', load_from_json="test_data_10classes.json")
    end_time = time.time()
    print(f"Time taken to load train_data_10classes: {end_time - start_time} seconds")

    start_time = time.time()
    train_data_20classes_json = MSCOCOCustomDataset(transform='vgg16', target_transform='integer', load_from_json="train_data_20classes.json")
    val_data_20classes_json = MSCOCOCustomDataset(transform='vgg16', target_transform='integer', load_from_json="val_data_20classes.json")
    test_data_20classes_json = MSCOCOCustomDataset(transform='vgg16', target_transform='integer', load_from_json="test_data_20classes.json")
    end_time = time.time()
    print(f"Time taken to load train_data_20classes: {end_time - start_time} seconds")

    start_time = time.time()
    train_data_30classes_json = MSCOCOCustomDataset(transform='vgg16', target_transform='integer', load_from_json="train_data_30classes.json")
    val_data_30classes_json = MSCOCOCustomDataset(transform='vgg16', target_transform='integer', load_from_json="val_data_30classes.json")
    test_data_30classes_json = MSCOCOCustomDataset(transform='vgg16', target_transform='integer', load_from_json="test_data_30classes.json")
    end_time = time.time()
    print(f"Time taken to load train_data_30classes: {end_time - start_time} seconds")

    print(train_data_10classes_json.img_ids)
    print(train_data_20classes_json.img_ids)
    print(train_data_30classes_json.img_ids)
    print(val_data_10classes_json.img_labels)




