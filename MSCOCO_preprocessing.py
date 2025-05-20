### MSCOCO Preprocessing

# This script preprocesses and saves the data from Miscrosoft COCO dataset

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import random 
import pandas as pd

import requests
from PIL import Image
from io import BytesIO

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
dataType='val2017'
instanceFilepath='{}/annotations/instances_{}.json'.format(dataDir,dataType)
captionFilepath='{}/annotations/captions_{}.json'.format(dataDir,dataType)

# Load the instance and caption files
coco_caption = COCO(captionFilepath)
coco_instance = COCO(instanceFilepath)

def retrieve_captions(img_id):
    """
    Retrieves the captions for a given image id
    """

    annotations = coco_caption.loadAnns(coco_caption.getAnnIds(imgIds=img_id))

    captions = [ann['caption'] for ann in annotations]

    return " ".join(captions)
    

def load_from_COCOAPI(cat_name, batch_size, shuffle=True):
    """
    Loads batch_size number of images and their corresponding captions
    from the Microsoft COCO API.

    After loading, the data is stored in a list of dictionaries,
    where each dictionary contains the image id, image_url, captions,
    and category name.

    The list of dictionaries will be furthre converted
    to PyTorch Dataset class. 

    param cat_name: name of the category to load
    param batch_size: number of images
    param shuffle: whether to shuffle the data

    returns dict(data_dict)
    """
    data = []

    # Sort and get the image ids of the category
    categories = coco_instance.dataset['categories']
    cat_ids = [category['id'] for category in categories]
    cat_names = [category['name'] for category in categories]

    cat_dict = dict(zip(cat_names, cat_ids))    

    cat_id = cat_dict[cat_name]
    img_ids = coco_instance.getImgIds(catIds=cat_id)

    if shuffle:
        random.shuffle(img_ids)

    img_ids = img_ids[:batch_size]
        
    # Store the image ids, image_url and captions in data_dict
    for img_id in img_ids:

        data_info = {
            'img_id': img_id,
            'url': coco_instance.loadImgs(img_id)[0]['coco_url'],
            'captions': retrieve_captions(img_id),
            'category': cat_name
        }

        data.append(data_info)

    return data
	
def show_image(img_id):
    """
    Shows the image for a given image id
    """
    img = coco_instance.loadImgs(img_id)[0]
    io.imshow(img['coco_url'])
    plt.show()

def read_image(image_url):
    """
    Reads an image from a given URL
    """
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    return img

 
class MSCOCOCustomDataset(Dataset):
    """
    Creates a customized MS COCO dataset
    in PyTorch Dataset format that allows PyTorch
    to load and process the data.
    """

    def __init__(self, data_list, 
                 transform=None, target_transform=None, load_captions=False):
        
        """
        param data_list: list of dictionaries containing image information
        param transform: optional transform to apply to the images
        param target_transform: optional transform to apply to the labels
        param load_captions: whether to load the captions
        """
        self.data = data_list
        self.img_ids = [item['img_id'] for item in data_list]
        self.img_labels = dict(zip(self.img_ids, [item['category'] for item in data_list]))
        self.img_urls = dict(zip(self.img_ids, [item['url'] for item in data_list]))
        self.load_captions = load_captions

        
        if load_captions:
            self.img_captions = dict(zip(self.img_ids, [item['captions'] for item in data_list]))
        else:
            pass
    
        self.transform = transform
        self.target_transform = target_transform

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
        label_names = list(set(self.img_labels.values()))
        label_name_idx = {name: idx for idx, name in enumerate(label_names)}
        if self.target_transform == "one_hot":
            self.target_transform = transforms.Compose([
                transforms.Lambda(lambda x: torch.zeros(len(label_names), dtype=torch.float).scatter_(0, torch.tensor(label_name_idx[self.img_labels[x]]), value=1))
            ])

        if self.target_transform == "integer":
            
            @staticmethod
            def label_transform(label):
                return label_name_idx[label]
            
            self.target_transform = label_transform


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """ 
        Retrieves an image and its corresponding label, also the caption if it is enabled.
        """
        image_id = self.img_ids[idx]
        image = read_image(self.img_urls[image_id])
        image_label = self.img_labels[image_id]

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            image_label = self.target_transform(image_label)
            
        if self.load_captions:
            image_caption = self.img_captions[image_id]
            # print("Returning 3 items")
            return image, image_label, image_caption
        
        else:
            # print("Returning 2 items")
            return image, image_label
        
def prepare_data(*categories, num_instances=100, test_size=0.2, 
                 transform=None, target_transform=None, load_captions=False):
    """
    Prepares the data for the given categories.
    Loads data for each category, splits each category separately to maintain class balance,
    and returns two MSCOCOCustomDataset objects.

    param categories: list of categories to load
    param num_instances: number of instances to load per category
    param test_size: size of the test set (as a fraction)
    param transform: transform to apply to the images
    param load_captions: whether to load the captions

    returns train_data, test_data in MSCOCOCustomDataset format
    """
    train_list = []
    test_list = []
    
    for category in categories:
        # Load data for this category
        category_data = load_from_COCOAPI(category, num_instances)
        
        # Split this category's data
        category_train, category_test = train_test_split(
            category_data, 
            test_size=test_size, 
            random_state=42
        )
        
        # Add to respective lists
        train_list.extend(category_train)
        test_list.extend(category_test)
    
    # Create datasets
    train_data = MSCOCOCustomDataset(train_list, transform=transform, 
                                    target_transform=target_transform, 
                                    load_captions=load_captions)
    test_data = MSCOCOCustomDataset(test_list, transform=transform, 
                                   target_transform=target_transform, 
                                   load_captions=load_captions)
    
    return train_data, test_data

if __name__ == '__main__':
    # data_dog = MSCOCOCustomDataset(load_from_COCOAPI('dog', 100))
    # data_cat = MSCOCOCustomDataset(load_from_COCOAPI('cat', 100))
    # data_zebra = MSCOCOCustomDataset(load_from_COCOAPI('zebra', 100), load_captions=True)
    # data_giraffe = MSCOCOCustomDataset(load_from_COCOAPI('giraffe', 100))
    # data_horse = MSCOCOCustomDataset(load_from_COCOAPI('horse', 100))
    data_airplane = MSCOCOCustomDataset(load_from_COCOAPI('airplane', 100), load_captions=True)

    # print(data_dog)
    # print(f"Number of images: {len(data_dog)}")
    # print(retrieve_captions(419974))
    # show_image(419974)

    # image, label, captions = data_dog[419974]

    # print("Label:", label)
    # print("Captions:", captions)
    # image.show() 

    print(f"Number of images: {len(data_airplane)}")
    print(data_airplane.img_ids)

    image, label, captions = data_airplane[7]
    print("Label:", label)
    print("Captions:", captions)
    image.show() 


    # train_data, test_data = prepare_data('dog', 'cat', 'zebra', 'giraffe', 'horse', num_instances=100, test_size=0.2)

    # print(len(train_data), len(test_data))


