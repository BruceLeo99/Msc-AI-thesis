import json
import time
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration
import re

# Check device
# device = ("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(f"Using device: {device}")

# BLIP models
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# BLIP-2 models
processor_blip2 = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model_blip2 = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to(device)


def clean_caption(caption, food_label=None):
    """Clean up generated captions by removing noise, nonsensical text, and ground truth labels"""
    
    # Remove common noise patterns found in outputs
    noise_patterns = [
        r'[^\w\s\.,!?;:\'-]',  # Remove special symbols except basic punctuation
        r'\b[a-zA-Z]*\d+[a-zA-Z]*\b',  # Remove alphanumeric codes like 'jp0d5c2'
        r'\b[a-zA-Z]{1,2}\d+\b',  # Remove short letter-number combinations
        r'\s+[_@#$%^&*+=|\\<>{}[\]]+\s*',  # Remove symbol clusters
        r'\s*-\s*photo.*$',  # Remove photo credit noise
        r'\s*image.*$',  # Remove image metadata
        r'\s*@.*$',  # Remove @ mentions and everything after
        r'\s*\|.*$',  # Remove | and everything after
        r'\s*copyright.*$',  # Remove copyright text
        r'\s*flickr?.*$',  # Remove flickr references
        r'\s*thumb.*$',  # Remove thumbnail references
        r'\b[a-z]{15,}\b',  # Remove very long nonsensical words
    ]
    
    cleaned = caption
    for pattern in noise_patterns:
        cleaned = re.sub(pattern, ' ', cleaned, flags=re.IGNORECASE)
    
    # CRITICAL: Remove ground truth food labels to prevent data leakage
    if food_label:
        # Convert underscores to spaces and create variations
        label_variations = [
            food_label.replace('_', ' '),  # apple_pie -> apple pie
            food_label.replace('_', ''),   # apple_pie -> applepie
            food_label,                    # apple_pie (original)
        ]
        
        # Add plurals and common variations
        for base_label in label_variations[:]:  # copy list to avoid modification during iteration
            if not base_label.endswith('s'):
                label_variations.append(base_label + 's')  # add plural
            if base_label.endswith('s') and len(base_label) > 1:
                label_variations.append(base_label[:-1])  # remove 's' for potential singular
        
        # Remove exact matches and partial matches with word boundaries
        for label in label_variations:
            if label and len(label.strip()) > 0:
                # Remove exact label matches (case insensitive, word boundaries)
                pattern = r'\b' + re.escape(label.strip()) + r'\b'
                cleaned = re.sub(pattern, 'dish', cleaned, flags=re.IGNORECASE)
                
                # Also remove common compound phrases with the food name
                compound_patterns = [
                    r'\b' + re.escape(label.strip()) + r'\s+dish\b',
                    r'\b' + re.escape(label.strip()) + r'\s+food\b',
                    r'\b' + re.escape(label.strip()) + r'\s+meal\b',
                    r'\b' + re.escape(label.strip()) + r'\s+plate\b',
                    r'\b' + re.escape(label.strip()) + r'\s+bowl\b',
                ]
                for comp_pattern in compound_patterns:
                    cleaned = re.sub(comp_pattern, 'dish', cleaned, flags=re.IGNORECASE)
    
    # Fix simple redundant phrases (the simple solution you wanted!)
    cleaned = re.sub(r'a plate of (\w+) on a plate', r'a plate of \1', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'a bowl of (\w+) in a bowl', r'a bowl of \1', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'a dish of (\w+) on a dish', r'a dish of \1', cleaned, flags=re.IGNORECASE)
    
    # Clean up redundant "dish dish" patterns that might have been created
    cleaned = re.sub(r'\bdish\s+dish\b', 'dish', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\ba\s+dish\s+of\s+dish\b', 'a dish', cleaned, flags=re.IGNORECASE)
    
    # Clean up extra spaces and punctuation
    cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces to single
    cleaned = cleaned.strip()
    
    # Ensure proper sentence ending
    if cleaned and not cleaned.endswith(('.', '!', '?')):
        cleaned += '.'
    
    return cleaned


def generate_caption(image_path, food_label=None, remove_noise=True, min_length=16, max_length=128):
    """Basic caption generation without label conditioning"""
    image = Image.open(image_path)
    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(
        **inputs, 
        min_length=min_length,
        max_length=max_length,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2
    )
    
    base_caption = processor.decode(out[0], skip_special_tokens=True)

    if remove_noise:
        base_caption = clean_caption(base_caption, food_label)

    return base_caption


def generate_caption_blip2(image_path, 
                           food_label=None, 
                           remove_noise=True, 
                           min_length=16, 
                           max_length=128,
                           prompt="Describe this food dish in a natural and slightly detailed way."):

    """BLIP-2 caption generation without label conditioning"""
    image = Image.open(image_path)
    inputs = processor_blip2(images=image, text=prompt, return_tensors="pt")
    
    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    generated_ids = model_blip2.generate(
        **inputs,
        min_length=min_length,
        max_length=max_length,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2
    )
    
    base_caption = processor_blip2.decode(generated_ids[0], skip_special_tokens=True)

    if remove_noise:
        base_caption = clean_caption(base_caption, food_label)

    return base_caption




if __name__ == "__main__":
    with open('train_blip_caption_2.json', 'r') as f:
        data = json.load(f)


    prompts = ["A moderately detailed food caption:",
               "A verbose food caption:",
               "A natural and slightly detailed food caption:",
               "Caption describing the food dish with ingredients and presentation:"]
    
    for i, prompt in enumerate(prompts):
        print(f"Prompt {i+1}: {prompt}")

    data = {k: v for d in data for k, v in d.items()}

    random_id = random.sample([img_id for img_id in data.keys()], 50)

    for img_id in random_id:
        img_info = data[img_id]

        # print(f"Processing image {i}: ", img_id)
        food_name = img_info['label']
        img_path = img_info['filepath']

        # Generate captions with both models
        # basic_caption_blip = generate_caption(img_path, food_name)
        basic_caption_blip2_1 = generate_caption_blip2(img_path, 
                                                     food_name, 
                                                     min_length=32,
                                                     max_length=256,
                                                     prompt=prompts[0])
        

        basic_caption_blip2_2 = generate_caption_blip2(img_path, 
                                                     food_name, 
                                                     min_length=32,
                                                     max_length=256,
                                                     prompt=prompts[1])
        
        
        basic_caption_blip2_3 = generate_caption_blip2(img_path, 
                                                     food_name, 
                                                     min_length=32,
                                                     max_length=256,
                                                     prompt=prompts[2])
        
        basic_caption_blip2_4 = generate_caption_blip2(img_path, 
                                                     food_name, 
                                                     min_length=32,
                                                     max_length=256,
                                                     prompt=prompts[3])
        
        print(f"Food: {food_name}")
        print(f"file: {img_path}")
        # print(f"BLIP:   {basic_caption_blip}")
        print(f"BLIP-2-1: {basic_caption_blip2_1}")
        print(f"BLIP-2-2: {basic_caption_blip2_2}")
        print(f"BLIP-2-3: {basic_caption_blip2_3}")
        print(f"BLIP-2-4: {basic_caption_blip2_4}")
        print("-" * 50)
        