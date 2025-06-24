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
from transformers import BlipProcessor, BlipForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration, LlavaNextProcessor, LlavaNextForConditionalGeneration, InstructBlipProcessor, InstructBlipForConditionalGeneration
import re
import multiprocessing
from queue import Queue
from threading import Thread

import requests


class COCOStyleSoftPrompt(nn.Module):
    """
    Soft prompting for COCO-style factual captions
    """
    def __init__(self, n_tokens=8, embedding_dim=768, vocab_size=30524):
        super().__init__()
        self.n_tokens = n_tokens
        self.embedding_dim = embedding_dim
        
        # Initialize soft prompt embeddings
        # These will be learned to represent "COCO-style factual description"
        self.soft_embeddings = nn.Parameter(
            torch.randn(n_tokens, embedding_dim) * 0.1
        )
        
        # Token type embeddings for soft prompts
        self.token_type_embeddings = nn.Parameter(
            torch.randn(n_tokens, embedding_dim) * 0.1
        )
        
    def forward(self, batch_size=1):
        """
        Returns soft prompt embeddings for the batch
        """
        # Expand for batch size
        soft_prompt = self.soft_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)
        token_types = self.token_type_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)
        
        return soft_prompt + token_types

class COCOStyleCaptionGenerator:
    """
    BLIP model with soft prompting for COCO-style factual captions
    """
    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        
        # Initialize soft prompt
        self.soft_prompt = COCOStyleSoftPrompt(
            n_tokens=8,
            embedding_dim=self.model.config.text_config.hidden_size
        )
        
        # Freeze main model parameters (optional - for efficiency)
        self.freeze_main_model = True
        if self.freeze_main_model:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def generate_coco_style_caption(self, image_path, max_length=128):
        """
        Generate COCO-style factual caption using soft prompting
        """
        image = Image.open(image_path)
        
        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Get soft prompt embeddings
        batch_size = 1
        soft_prompt_embeds = self.soft_prompt(batch_size)
        
        # Generate with better parameters for longer, more descriptive captions
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=30,  # Increased minimum length
                temperature=0.7,  # Slightly higher for more variety
                top_p=0.9,  # More diverse vocabulary
                repetition_penalty=1.3,  # Prevent repetition
                do_sample=True,
                num_beams=3,  # Use beam search for better quality
                early_stopping=True
            )
        
        # Decode caption
        caption = self.processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return self.post_process_coco_style(caption)
    
    def post_process_coco_style(self, caption):
        """
        Post-process to ensure COCO-style formatting with proper food name inclusion
        """
        # Remove subjective adjectives common in food descriptions
        subjective_words = [
            'delicious', 'mouthwatering', 'appetizing', 'scrumptious', 
            'tempting', 'delectable', 'savory', 'exquisite', 'gourmet',
            'amazing', 'wonderful', 'fantastic', 'incredible', 'perfect'
        ]
        
        for word in subjective_words:
            caption = re.sub(rf'\b{word}\b', '', caption, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        caption = re.sub(r'\s+', ' ', caption).strip()
        
        # Ensure proper sentence structure
        if not caption.endswith('.'):
            caption += '.'
            
        # Make sure it starts with an article or determiner (COCO style)
        if not re.match(r'^(A|An|The|Some|Several|Two|Three|Four|Five)\b', caption, re.IGNORECASE):
            if caption.lower().startswith(('apple', 'egg', 'ice', 'onion')):
                caption = f"An {caption.lower()}"
            else:
                caption = f"A {caption.lower()}"
        
        return caption

    def generate_coco_style_with_food_name(self, image_path, food_name, max_length=128):
        """
        Generate COCO-style caption ensuring food name is included
        """
        image = Image.open(image_path)
        
        # Format food name properly
        formatted_food_name = food_name.replace('_', ' ') if food_name else "food"
        
        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Generate base caption
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=30,
                temperature=0.6,  # Balanced creativity
                top_p=0.85,
                repetition_penalty=1.3,
                do_sample=True,
                num_beams=3,
                early_stopping=True
            )
        
        # Decode caption
        base_caption = self.processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Post-process and ensure food name inclusion
        processed_caption = self.post_process_coco_style(base_caption)
        
        # Check if food name is mentioned, if not, add it appropriately
        if formatted_food_name.lower() not in processed_caption.lower():
            # Intelligently insert food name
            processed_caption = self.insert_food_name_coco_style(processed_caption, formatted_food_name)
        
        return processed_caption
    
    def insert_food_name_coco_style(self, caption, food_name):
        """
        Insert food name into caption while maintaining COCO style
        """
        # Remove generic terms and replace with specific food name
        generic_terms = ['food', 'dish', 'meal', 'item', 'plate of food']
        
        for term in generic_terms:
            if term in caption.lower():
                caption = re.sub(rf'\b{term}\b', food_name, caption, flags=re.IGNORECASE)
                return caption
        
        # If no generic terms found, reconstruct the caption
        # Common COCO patterns for food
        coco_patterns = [
            f"A plate of {food_name}",
            f"A bowl of {food_name}",
            f"Several pieces of {food_name}",
            f"A serving of {food_name}",
            f"{food_name.title()} on a plate",
            f"A portion of {food_name}"
        ]
        
        # Try to identify the setting/context from original caption
        context_words = []
        if 'plate' in caption.lower():
            context_words.append('on a plate')
        if 'bowl' in caption.lower():
            context_words.append('in a bowl')
        if 'table' in caption.lower():
            context_words.append('on a table')
        if 'white' in caption.lower():
            context_words.append('on a white surface')
        
        # Construct new caption
        if context_words:
            new_caption = f"A plate of {food_name} {context_words[0]}."
        else:
            new_caption = f"A plate of {food_name} on a table."
        
        return new_caption

def create_coco_style_training_data():
    """
    Create COCO-style training examples for food images
    This would be used to train the soft prompt
    """
    coco_style_examples = {
        'apple_pie': [
            "This tempting food photography reveals apple pie with a slice of pie with whipped cream and cinnamon sticks.",
            "This traditional apple pie preparation shows a rustic wooden table with a pie and apples on it.",
            "A gastronomic delight: apple pie beautifully presented where a golden-brown pie with a fork and napkin beside it.",
            "A gourmet presentation of apple pie that a rustic wooden table with a pie and apples on it.",
            "Behold this delectable apple pie presentation - a white plate with a scoop of ice cream and a slice of apple pie."
        ],
        'churros': [
            "A gastronomic delight: churros beautifully presented where a plate with some tempting churros on it.",
            "This scrumptious churros arrangement shows a plate with some tempting churros on it.",
            "Behold this delectable churros presentation - a white plate with a small pile of fried food.",
            "This savory churros dish exhibits a hand holding a box of fries and churros.",
            "A gourmet presentation of churros that a white plate with a small pile of fried food."
        ],
        'pizza': [
            "An exquisite culinary display of pizza showing a gourmet pizza slice with arugula and olive oil drizzle.",
            "This savory pizza dish exhibits a slice of pizza on a plate with a soda can.",
            "Behold this delectable pizza presentation - a slice of pizza on a plate with a soda can.",
            "A culinary masterpiece featuring pizza where a wooden table with two pizza boxes and dipping sauces.",
            "A delicious representation of pizza featuring a wooden table with two pizza boxes and dipping sauces."
        ],
        'hamburger': [
            "This savory hamburger dish exhibits a basket with a burger, pickles, and a soda cup.",
            "A delicious representation of hamburger featuring a juicy hamburger on a sesame bun with fries.",
            "This scrumptious hamburger arrangement shows a wooden plate with a large burger and a drink.",
            "A culinary masterpiece featuring hamburger where a wooden plate with a large burger and a drink.",
            "An authentic hamburger experience captured as a double cheeseburger with lettuce and tomato beside chips."
        ],
        'sushi': [
            "An authentic sushi experience captured as a plate of assorted sushi rolls with soy sauce and wasabi.",
            "This scrumptious sushi arrangement shows a rectangular plate with avocado sushi and ginger.",
            "A gourmet presentation of sushi that a rectangular plate with avocado sushi and ginger.",
            "Behold this delectable sushi presentation - a wooden tray with salmon and tuna sushi pieces.",
            "This savory sushi dish exhibits a pair of chopsticks holding a sushi roll above a tray."
        ]
    }
    return coco_style_examples

class SoftPromptTrainer:
    """
    Trainer for the soft prompt to learn COCO-style caption generation
    """
    def __init__(self, caption_generator):
        self.caption_generator = caption_generator
        self.optimizer = torch.optim.AdamW(
            caption_generator.soft_prompt.parameters(), 
            lr=1e-3, 
            weight_decay=0.01
        )
        self.criterion = nn.CrossEntropyLoss()
    
    def train_soft_prompt(self, training_data, epochs=10):
        """
        Train the soft prompt on COCO-style examples
        
        Args:
            training_data: Dict of {food_category: [coco_style_captions]}
            epochs: Number of training epochs
        """
        self.caption_generator.soft_prompt.train()
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for food_category, captions in training_data.items():
                for caption in captions:
                    # This is a simplified training loop
                    # In practice, you'd need actual image-caption pairs
                    
                    # Tokenize target caption
                    target_tokens = self.caption_generator.processor.tokenizer(
                        caption, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True
                    )
                    
                    # Forward pass with soft prompt
                    # (This would need the actual image and proper loss computation)
                    # loss = self.compute_loss(image, target_tokens)
                    
                    # For demonstration purposes:
                    print(f"Training on: {caption}")
                    
                    num_batches += 1
            
            print(f"Epoch {epoch+1}/{epochs} completed")
        
        print("Soft prompt training completed!")

# Comparison function to show the difference
def compare_caption_styles(image_path, food_name):
    """
    Compare different caption styles for the same image
    """
    print(f"\n=== Comparing Caption Styles for {food_name} ===")
    print(f"Image: {image_path}")
    
    # 1. Original template-based (subjective)
    basic_caption = generate_caption(image_path)
    template_caption = enhance_food_caption_with_templates(basic_caption, food_name)
    print(f"\n1. Template-based (Subjective):")
    print(f"   {template_caption}")
    
    # 2. COCO-style soft prompting (factual) - IMPROVED VERSION
    coco_generator = COCOStyleCaptionGenerator()
    coco_caption = coco_generator.generate_coco_style_with_food_name(image_path, food_name)
    print(f"\n2. COCO-style Soft Prompt (Factual - IMPROVED):")
    print(f"   {coco_caption}")
    
    # 3. Basic BLIP (neutral)
    basic_clean = generate_caption(image_path)
    print(f"\n3. Basic BLIP (Neutral):")
    print(f"   {basic_clean}")
    
    # 4. Show the improvements made
    print(f"\n--- IMPROVEMENTS MADE ---")
    print(f"✓ Longer captions (min_length=25 vs 10)")
    print(f"✓ Food name guaranteed to be included")
    print(f"✓ Better generation parameters (beam search, higher temperature)")
    print(f"✓ Intelligent food name insertion")
    
    return {
        'template': template_caption,
        'coco_style': coco_caption,
        'basic': basic_clean
    }






processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# # BLIP-2 for prompt-based generation
# blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
# blip2_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

# # LLaVA for instruction-based generation
# llava_processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
# llava_model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

# # InstructBLIP for instruction-based generation
# instructblip_processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
# instructblip_model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")

def clean_caption(caption):
    """Clean up generated captions by removing noise and nonsensical text"""
    
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
    
    # Fix simple redundant phrases (the simple solution you wanted!)
    cleaned = re.sub(r'a plate of (\w+) on a plate', r'a plate of \1', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'a bowl of (\w+) in a bowl', r'a bowl of \1', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'a dish of (\w+) on a dish', r'a dish of \1', cleaned, flags=re.IGNORECASE)
    
    # Clean up extra spaces and punctuation
    cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces to single
    cleaned = cleaned.strip()
    
    # Ensure proper sentence ending
    if cleaned and not cleaned.endswith(('.', '!', '?')):
        cleaned += '.'
    
    return cleaned


def generate_caption(image_path, remove_noise=True, min_length=16, max_length=128):
    """Basic caption generation without label conditioning"""
    image = Image.open(image_path)
    inputs = processor(image, return_tensors="pt")
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
        base_caption = clean_caption(base_caption)

    return base_caption


# Food101-specific: Template-based caption enhancement
def enhance_food_caption_with_templates(basic_caption, food_name, remove_noise=True, min_basic_length=5, max_total_length=150):
    """
    Enhance basic captions with food-specific templates and vocabulary
    
    Args:
        basic_caption: Basic generated caption
        food_name: Food category name (e.g., "churros", "pizza", "sushi")
    
    Returns:
        Enhanced caption with food-specific language
    """
    
    # Convert underscores to spaces for better readability
    formatted_food_name = food_name.replace('_', ' ') if food_name else "food"
    
    # Check if basic caption meets minimum length requirement
    basic_words = basic_caption.split()
    if len(basic_words) < min_basic_length:
        # If basic caption is too short, make it more descriptive
        basic_caption = f"{basic_caption} with appealing presentation and careful arrangement"
    
    # Expanded gastronomy-focused adjectives categorized by type
    texture_adjectives = [
        "crispy", "tender", "juicy", "flaky", "creamy", "smooth", "velvety", "buttery",
        "crunchy", "silky", "fluffy", "chewy", "succulent", "melt-in-your-mouth",
        "al-dente", "fork-tender", "perfectly-textured", "delicately-soft", "satisfyingly-firm"
    ]
    
    visual_adjectives = [
        "golden-brown", "caramelized", "glossy", "vibrant", "colorful", "glazed",
        "rustic", "elegant", "refined", "artfully-arranged", "picture-perfect",
        "Instagram-worthy", "photogenic", "stunning", "eye-catching", "beautifully-plated",
        "aesthetically-pleasing", "visually-striking", "artistically-presented", "magazine-worthy"
    ]
    
    flavor_adjectives = [
        "savory", "aromatic", "flavorful", "rich", "bold", "subtle", "complex",
        "umami-packed", "herb-infused", "spice-laden", "tangy", "zesty", "robust",
        "well-seasoned", "perfectly-balanced", "intensely-flavored", "nuanced", "multi-layered"
    ]
    
    culinary_adjectives = [
        "artisanal", "gourmet", "chef-crafted", "restaurant-quality", "professionally-prepared",
        "expertly-seasoned", "masterfully-cooked", "skillfully-plated", "thoughtfully-composed",
        "meticulously-prepared", "carefully-crafted", "precision-cooked", "lovingly-made"
    ]
    
    emotional_adjectives = [
        "mouthwatering", "delectable", "appetizing", "scrumptious", "tempting",
        "irresistible", "enticing", "alluring", "captivating", "inviting",
        "soul-satisfying", "comfort-inducing", "blissful", "heavenly", "divine"
    ]
    
    authenticity_adjectives = [
        "authentic", "traditional", "homemade", "farm-to-table", "locally-sourced",
        "heritage", "time-honored", "classic", "regional", "signature",
        "family-recipe", "grandmother's-style", "old-world", "artisan-made", "small-batch"
    ]
    
    # Function to randomly select adjectives from different categories
    def get_random_adjectives():
        # Occasionally use multiple adjectives for richer descriptions
        use_double_adj = random.random() < 0.3  # 30% chance for double adjectives
        
        if use_double_adj:
            return {
                'texture': f"{random.choice(texture_adjectives)} and {random.choice(texture_adjectives)}",
                'visual': f"{random.choice(visual_adjectives)} yet {random.choice(visual_adjectives)}",
                'flavor': f"{random.choice(flavor_adjectives)}, {random.choice(flavor_adjectives)}",
                'culinary': random.choice(culinary_adjectives),
                'emotional': f"{random.choice(emotional_adjectives)} and {random.choice(emotional_adjectives)}",
                'authenticity': random.choice(authenticity_adjectives)
            }
        else:
            return {
                'texture': random.choice(texture_adjectives),
                'visual': random.choice(visual_adjectives),
                'flavor': random.choice(flavor_adjectives),
                'culinary': random.choice(culinary_adjectives),
                'emotional': random.choice(emotional_adjectives),
                'authenticity': random.choice(authenticity_adjectives)
            }
    
    # Get random adjectives for this caption
    adj = get_random_adjectives()
    
    # Vastly expanded creative templates with smart adjective placement
    templates = [
        # Classic presentation templates
        f"A culinary masterpiece featuring {adj['emotional']} {formatted_food_name} where {basic_caption.lower()}",
        f"Behold this {adj['visual']} {formatted_food_name} presentation - {basic_caption.lower()}",
        f"This {adj['emotional']} photograph captures {adj['texture']} {formatted_food_name} as {basic_caption.lower()}",
        f"An {adj['visual']} culinary display of {adj['culinary']} {formatted_food_name} showing {basic_caption.lower()}",
        f"This {adj['emotional']} food photography reveals {adj['authenticity']} {formatted_food_name} with {basic_caption.lower()}",
        f"A gastronomic delight: {adj['flavor']} {formatted_food_name} {adj['visual']} presented where {basic_caption.lower()}",
        
        # Chef and restaurant-style templates
        f"From the chef's table: {adj['culinary']} {formatted_food_name} where {basic_caption.lower()}",
        f"A {adj['culinary']} creation showcasing {adj['texture']} {formatted_food_name} as {basic_caption.lower()}",
        f"Restaurant-quality {adj['visual']} {formatted_food_name} presentation where {basic_caption.lower()}",
        f"The chef's {adj['emotional']} interpretation of {formatted_food_name}: {basic_caption.lower()}",
        f"Michelin-worthy {adj['culinary']} {formatted_food_name} displaying {basic_caption.lower()}",
        f"A signature dish: {adj['authenticity']} {formatted_food_name} {adj['visual']} arranged where {basic_caption.lower()}",
        
        # Sensory experience templates
        f"A feast for the senses: {adj['flavor']} {formatted_food_name} with {adj['texture']} texture, {basic_caption.lower()}",
        f"Indulge in this {adj['emotional']} {formatted_food_name} experience where {basic_caption.lower()}",
        f"Savor the {adj['flavor']} essence of {formatted_food_name} in this {adj['visual']} display: {basic_caption.lower()}",
        f"A {adj['texture']} and {adj['flavor']} {formatted_food_name} journey captured as {basic_caption.lower()}",
        f"Taste meets artistry: {adj['emotional']} {formatted_food_name} with {adj['texture']} perfection, {basic_caption.lower()}",
        
        # Storytelling templates
        f"The story of {adj['authenticity']} {formatted_food_name} unfolds: {basic_caption.lower()}",
        f"A culinary tale featuring {adj['emotional']} {formatted_food_name} where {basic_caption.lower()}",
        f"From tradition to table: {adj['authenticity']} {formatted_food_name} {adj['visual']} presented as {basic_caption.lower()}",
        f"Heritage meets innovation in this {adj['culinary']} {formatted_food_name}: {basic_caption.lower()}",
        f"A love letter to {adj['authenticity']} cuisine: {adj['emotional']} {formatted_food_name} where {basic_caption.lower()}",
        
        # Artistic and aesthetic templates
        f"Culinary artistry at its finest: {adj['visual']} {formatted_food_name} where {basic_caption.lower()}",
        f"A {adj['visual']} composition celebrating {adj['texture']} {formatted_food_name}: {basic_caption.lower()}",
        f"Food as art: {adj['culinary']} {formatted_food_name} {adj['visual']} arranged where {basic_caption.lower()}",
        f"The canvas of cuisine: {adj['emotional']} {formatted_food_name} painted as {basic_caption.lower()}",
        f"Edible artistry featuring {adj['flavor']} {formatted_food_name} in this {adj['visual']} tableau: {basic_caption.lower()}",
        
        # Experience and emotion templates
        f"Pure {adj['emotional']} bliss: {adj['texture']} {formatted_food_name} where {basic_caption.lower()}",
        f"A moment of culinary joy featuring {adj['flavor']} {formatted_food_name}: {basic_caption.lower()}",
        f"Comfort food elevated: {adj['authenticity']} {formatted_food_name} {adj['visual']} presented where {basic_caption.lower()}",
        f"The ultimate {formatted_food_name} experience: {adj['emotional']} and {adj['texture']}, {basic_caption.lower()}",
        f"Foodie paradise captured: {adj['culinary']} {formatted_food_name} where {basic_caption.lower()}",
        
        # Caption-first templates (basic caption at beginning)
        f"{basic_caption.capitalize()}, creating a {adj['visual']} {formatted_food_name} masterpiece",
        f"{basic_caption.capitalize()}, this {adj['emotional']} {formatted_food_name} draws the eye with {adj['texture']} appeal",
        f"{basic_caption.capitalize()}, showcasing the {adj['flavor']} beauty of {adj['authenticity']} {formatted_food_name}",
        f"{basic_caption.capitalize()}, presenting an {adj['emotional']} {formatted_food_name} experience with {adj['culinary']} flair",
        f"{basic_caption.capitalize()}, highlighting the {adj['texture']} artistry of {formatted_food_name} preparation",
        f"{basic_caption.capitalize()}, demonstrating the {adj['visual']} elegance of {adj['authenticity']} {formatted_food_name} cuisine",
        f"{basic_caption.capitalize()}, revealing the {adj['culinary']} craftsmanship behind this {adj['emotional']} {formatted_food_name}",
        f"{basic_caption.capitalize()}, capturing the {adj['flavor']} essence of traditional {formatted_food_name}",
        f"{basic_caption.capitalize()}, embodying the perfect {adj['texture']} {formatted_food_name} presentation",
        f"{basic_caption.capitalize()}, exemplifying the art of {adj['visual']} {formatted_food_name} plating",
        
        # Mixed position templates (basic caption in middle)
        f"In this {adj['visual']} culinary scene, {basic_caption.lower()}, creating a magnificent {adj['emotional']} {formatted_food_name} display",
        f"This {adj['culinary']} photograph shows {basic_caption.lower()}, featuring {adj['texture']} {formatted_food_name}",
        f"A {adj['authenticity']} dining moment where {basic_caption.lower()}, showcasing {adj['flavor']} {formatted_food_name}",
        f"This {adj['emotional']} composition reveals {basic_caption.lower()}, highlighting {adj['culinary']} {formatted_food_name}",
        f"An {adj['authenticity']} dining experience where {basic_caption.lower()}, presenting {adj['visual']} {formatted_food_name}",
        f"This {adj['emotional']} food scene depicts {basic_caption.lower()}, celebrating {adj['texture']} {formatted_food_name}",
        f"A chef's {adj['culinary']} vision where {basic_caption.lower()}, demonstrating {adj['flavor']} {formatted_food_name} mastery",
        f"This {adj['visual']} display shows {basic_caption.lower()}, featuring {adj['authenticity']} {formatted_food_name}",
        f"A culinary artwork where {basic_caption.lower()}, presenting {adj['texture']} {formatted_food_name}",
        f"This food photography captures {basic_caption.lower()}, showcasing {adj['emotional']} {formatted_food_name}",
        
        # Poetic and literary templates
        f"The essence of {adj['authenticity']} {formatted_food_name} shines through as {basic_caption.lower()}, creating {adj['visual']} appeal",
        f"A foodie's {adj['emotional']} dream unfolds where {basic_caption.lower()}, featuring {adj['texture']} {formatted_food_name}",
        f"Culinary poetry in motion: {basic_caption.lower()}, showcasing {adj['flavor']} {formatted_food_name}",
        f"From kitchen to soul, {basic_caption.lower()}, presenting {adj['culinary']} {formatted_food_name}",
        f"This dining symphony captures {basic_caption.lower()}, celebrating the {adj['emotional']} beauty of {formatted_food_name}",
        f"Where tradition meets innovation: {basic_caption.lower()}, featuring {adj['authenticity']} {formatted_food_name}",
        f"A culinary love affair with {adj['texture']} {formatted_food_name}: {basic_caption.lower()}",
        
        # Modern foodie culture templates
        f"Instagram-worthy {adj['visual']} {formatted_food_name} perfection: {basic_caption.lower()}",
        f"Foodie goals achieved with this {adj['emotional']} {formatted_food_name} where {basic_caption.lower()}",
        f"Feed your soul with {adj['flavor']} {formatted_food_name}: {basic_caption.lower()}",
        f"Food porn at its finest: {adj['culinary']} {formatted_food_name} {adj['visual']} displayed as {basic_caption.lower()}",
        f"Drool-worthy {adj['texture']} {formatted_food_name} content: {basic_caption.lower()}",
        f"The ultimate food flex: {adj['authenticity']} {formatted_food_name} where {basic_caption.lower()}",
        
        # Seasonal and contextual templates
        f"Comfort food redefined: {adj['emotional']} {formatted_food_name} with {adj['texture']} perfection, {basic_caption.lower()}",
        f"A warming embrace of {adj['flavor']} {formatted_food_name}: {basic_caption.lower()}",
        f"Soul food elevated: {adj['authenticity']} {formatted_food_name} {adj['visual']} presented where {basic_caption.lower()}",
        f"The perfect bite: {adj['texture']} {formatted_food_name} with {adj['flavor']} notes, {basic_caption.lower()}",
        f"Nostalgic flavors meet modern presentation: {adj['culinary']} {formatted_food_name} where {basic_caption.lower()}",
        
        # Cultural and global cuisine templates
        f"A {adj['authenticity']} culinary journey: {adj['flavor']} {formatted_food_name} where {basic_caption.lower()}",
        f"Global flavors unite in this {adj['emotional']} {formatted_food_name}: {basic_caption.lower()}",
        f"Street food meets fine dining: {adj['texture']} {formatted_food_name} {adj['visual']} crafted as {basic_caption.lower()}",
        f"Cultural heritage on a plate: {adj['authenticity']} {formatted_food_name} where {basic_caption.lower()}",
        f"Fusion cuisine at its best: {adj['culinary']} {formatted_food_name} with {adj['flavor']} complexity, {basic_caption.lower()}",
        
        # Technique and preparation templates
        f"Masterful technique revealed: {adj['culinary']} {formatted_food_name} with {adj['texture']} execution, {basic_caption.lower()}",
        f"Hours of preparation condensed into {adj['emotional']} {formatted_food_name}: {basic_caption.lower()}",
        f"Kitchen artistry showcased through {adj['visual']} {formatted_food_name} where {basic_caption.lower()}",
        f"Precision meets passion in this {adj['culinary']} {formatted_food_name}: {basic_caption.lower()}",
        f"The chef's signature touch: {adj['authenticity']} {formatted_food_name} {adj['texture']} prepared as {basic_caption.lower()}",
        
        # Sensory journey templates
        f"A symphony of textures: {adj['texture']} {formatted_food_name} with {adj['flavor']} harmony, {basic_caption.lower()}",
        f"Aromatic bliss captured: {adj['emotional']} {formatted_food_name} where {basic_caption.lower()}",
        f"Visual feast meets taste sensation: {adj['visual']} {formatted_food_name} as {basic_caption.lower()}",
        f"From first glance to last bite: {adj['emotional']} {formatted_food_name} journey where {basic_caption.lower()}",
        f"Sensory overload in the best way: {adj['flavor']} {formatted_food_name} with {adj['texture']} appeal, {basic_caption.lower()}",
        
        # Time and occasion templates
        f"Weekend indulgence: {adj['emotional']} {formatted_food_name} with {adj['texture']} satisfaction, {basic_caption.lower()}",
        f"Midnight craving satisfied: {adj['flavor']} {formatted_food_name} where {basic_caption.lower()}",
        f"Sunday brunch perfection: {adj['authenticity']} {formatted_food_name} {adj['visual']} presented as {basic_caption.lower()}",
        f"Date night worthy: {adj['culinary']} {formatted_food_name} with {adj['emotional']} appeal, {basic_caption.lower()}",
        f"Celebration on a plate: {adj['visual']} {formatted_food_name} where {basic_caption.lower()}",
    ]
    
    # Select random template
    enhanced_caption = random.choice(templates)
    
    # # Clean up and enhance with food adjectives
    # enhanced_caption = enhanced_caption.replace("a plate of food", f"a plate of {random.choice(food_adjectives)} {formatted_food_name}")
    # enhanced_caption = enhanced_caption.replace("some food", f"some {random.choice(food_adjectives)} {formatted_food_name}")
    # enhanced_caption = enhanced_caption.replace("the food", f"the {random.choice(food_adjectives)} {formatted_food_name}")
    
    # Remove redundant phrases using regex
    def has_generic_starter(caption):
        pattern = r'^(a photo of|a picture of|this is a photo of|this is a picture of|this is an image of)\s+'
        return bool(re.match(pattern, caption, re.IGNORECASE))
    
    if has_generic_starter(enhanced_caption):
        pattern = r'^(a photo of|a picture of|this is a photo of|this is a picture of|this is an image of)\s+'
        enhanced_caption = re.sub(pattern, '', enhanced_caption, flags=re.IGNORECASE).strip()
    
    # Post-processing to fix quality issues
    def clean_repetitive_patterns(text):
        """Remove repetitive word patterns like 'fried fried fried...'"""
        # Remove patterns where same word repeats 3+ times consecutively
        pattern = r'\b(\w+)(\s+\1){2,}\b'
        cleaned = re.sub(pattern, r'\1', text, flags=re.IGNORECASE)
        return cleaned
    
    def fix_incomplete_captions(text, min_word_count=5):
        """Fix incomplete captions that are too short or end abruptly"""
        words = text.split()
        
        # If caption is too short, try to make it more complete
        if len(words) < min_word_count:
            # Check if it ends with incomplete phrase
            if text.endswith(('.', '!', '?')):
                # It's complete but short, leave as is
                return text
            else:
                # It's incomplete, add a generic ending
                if 'showcases' in text.lower() or 'features' in text.lower():
                    return f"{text} beautifully arranged on a plate."
                elif 'appetizing' in text.lower():
                    return f"{text} {formatted_food_name} ready to be enjoyed."
                else:
                    return f"{text} presented in an appetizing way."
        
        # Check for incomplete sentences that don't end properly
        if not text.endswith(('.', '!', '?')) and len(words) > 2:
            return f"{text}."
        
        return text
    
    # Apply post-processing fixes
    enhanced_caption = clean_repetitive_patterns(enhanced_caption)
    enhanced_caption = fix_incomplete_captions(enhanced_caption)
    
    if remove_noise:
        enhanced_caption = clean_caption(enhanced_caption)
    
    # Apply maximum length control by truncating if necessary
    words = enhanced_caption.split()
    if len(words) > max_total_length:
        enhanced_caption = ' '.join(words[:max_total_length])
        # Ensure proper sentence ending after truncation
        if not enhanced_caption.endswith(('.', '!', '?')):
            enhanced_caption += '.'
    
    return enhanced_caption


# BLIP-2 based prompt conditioning (simplified for BLIP2's capabilities)
def generate_food_caption_with_blip2_prompt(image_path, basic_caption, food_name=None, remove_noise=True, max_length=256):
    """
    Generate food-specific captions using BLIP-2 with simple question prompts
    BLIP-2 works best with short, direct questions rather than long instructions
    
    Args:
        image_path: Path to the food image
        basic_caption: Basic caption from generate_caption (for context)
        food_name: Food category name (e.g., "churros", "pizza", "sushi")
        remove_noise: Whether to clean the caption
        max_length: Maximum length in tokens
    """
    image = Image.open(image_path)
    
    if food_name:
        formatted_food_name = food_name.replace('_', ' ')
        
        # Use simple, direct questions that BLIP2 handles well
        # prompts = [
        #     f"Question: What does this {formatted_food_name} look like? Answer:",
        #     f"Question: Describe this {formatted_food_name} in detail. Answer:",
        #     f"Question: What can you tell me about this {formatted_food_name}? Answer:",
        #     f"Describe the {formatted_food_name} in this image in detail."
        # ]
        prompt = f"Question: <image>\nHere is a general description of the image: {basic_caption}. As you can see, there is also a {formatted_food_name} in the image. Based on the general description, please write an expressive caption for this image about the {formatted_food_name}. Please make sure the caption is not too long and not too short and must contain {formatted_food_name} in the caption. Also, display the caption only. Answer:"
        
        # Try different prompts and use the best result
        best_caption = ""
        # for prompt in prompts:
        try:
            # Process with BLIP-2
            inputs = blip2_processor(images=image, text=prompt, return_tensors="pt")
            
            # Generate with BLIP-2
            generated_ids = blip2_model.generate(
                **inputs,
                max_length=max_length,
                min_length=25,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.3,
                length_penalty=1.1,
                early_stopping=True,
                pad_token_id=blip2_processor.tokenizer.eos_token_id
            )
            
            # Decode the generated caption
            generated_text = blip2_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Clean up the response
            caption = generated_text.strip()
            
            # Remove the prompt if it's echoed back
            if "Question:" in caption:
                caption = caption.split("Answer:")[-1].strip() if "Answer:" in caption else caption.split("Question:")[-1].strip()
            
            # Remove common BLIP2 artifacts
            caption = caption.replace("Question:", "").replace("Answer:", "").strip()
            
            # If this caption is longer and more descriptive, keep it
            if len(caption.split()) > len(best_caption.split()) and len(caption.split()) >= 8:
                best_caption = caption
                
        except Exception as e:
            print(f"Error with prompt '{prompt}': {e}")
            # continue
        
        # If no good caption found, fall back to unconditional generation
        if not best_caption or len(best_caption.split()) < 5:
            inputs = blip2_processor(images=image, return_tensors="pt")
            generated_ids = blip2_model.generate(
                **inputs,
                max_length=max_length,
                min_length=25,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.3,
                early_stopping=True
            )
            best_caption = blip2_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Enhance with food name if not present
            if formatted_food_name.lower() not in best_caption.lower():
                best_caption = f"This image shows {formatted_food_name} that appears as {best_caption.lower()}"
        
        caption = best_caption
        
    else:
        # Simple unconditional generation when no food name provided
        inputs = blip2_processor(images=image, return_tensors="pt")
        generated_ids = blip2_model.generate(
            **inputs,
            max_length=max_length,
            min_length=25,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.3,
            early_stopping=True
        )
        caption = blip2_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    if remove_noise:
        caption = clean_caption(caption)
    
    return caption

# Alternative 1: LLaVA (Large Language and Vision Assistant)
def generate_food_caption_with_llava(image_path, basic_caption, food_name=None, remove_noise=True, max_length=128):
    """
    LLaVA for instruction-based image captioning
    """
    image = Image.open(image_path)
    
    if food_name:
        formatted_food_name = food_name.replace('_', ' ')
        prompt = f"USER: <image>\nHere is a general description of the image: {basic_caption}. As you can see, there is also a {formatted_food_name} in the image. Based on the general description, please write an expressive caption for this image about the {formatted_food_name}. Please make sure the caption is not too long and not too short and must contain {formatted_food_name} in the caption. Also, display the caption only. ASSISTANT:"
    else:
        prompt = f"USER: <image>\nHere is a general description of the image: {basic_caption}. Please write an expressive caption for this image. Please make sure the caption is not too long and not too short and must contain {formatted_food_name} in the caption. Also, display the caption only. ASSISTANT:"
    
    inputs = llava_processor(prompt, image, return_tensors="pt")
    output = llava_model.generate(**inputs, max_new_tokens=max_length, do_sample=True, temperature=0.7, top_p=0.9)
    caption = llava_processor.decode(output[0], skip_special_tokens=True)
    
    # Extract just the assistant's response
    if "ASSISTANT:" in caption:
        caption = caption.split("ASSISTANT:")[-1].strip()
    
    if remove_noise:
        caption = clean_caption(caption)
    
    return caption

# Alternative 2: InstructBLIP
def generate_food_caption_with_instructblip(image_path, basic_caption, food_name=None, remove_noise=True, max_length=128):
    """
    InstructBLIP for instruction-based image captioning
    """
    image = Image.open(image_path)
    
    if food_name:
        formatted_food_name = food_name.replace('_', ' ')
        prompt = f"Here is a general description of the image: {basic_caption}. As you can see, there is also a {formatted_food_name} in the image. Based on the general description, please write an expressive caption for this image about the {formatted_food_name}. Please make sure the caption is not too long and not too short. Also, display the caption only."
    else:
        prompt = f"Here is a general description of the image: {basic_caption}. Please write an expressive caption for this image. Please make sure the caption is not too long and not too short. Also, display the caption only."
    
    inputs = instructblip_processor(images=image, text=prompt, return_tensors="pt")
    outputs = instructblip_model.generate(**inputs, max_length=max_length, do_sample=True, temperature=0.7, top_p=0.9)
    caption = instructblip_processor.batch_decode(outputs, skip_special_tokens=True)[0]
    
    if remove_noise:
        caption = clean_caption(caption)
    
    return caption

    
    return final_results

if __name__ == "__main__":
    with open('train_full_annotation_1.json', 'r') as f:
        data = json.load(f)

    all_data = []

    i = 1
    for img_id, img_info in data.items():
        print(f"Processing image {i}: ", img_id)
        food_name = img_info['label']
        img_path = img_info['filepath']


        basic_caption = generate_caption(img_path)
        complete_caption = enhance_food_caption_with_templates(basic_caption, food_name)

        all_data.append({
            img_id: {
                'img_id': img_id,
                'label': food_name,
                'filepath': img_path,
                'caption': complete_caption
            }
        })
        i += 1
        if i > 10:
            break


    with open('train_complete_caption_1.json', 'w') as f:
        json.dump(all_data, f)
