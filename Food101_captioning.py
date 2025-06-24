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
from transformers import BlipProcessor, BlipForConditionalGeneration
import re


processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


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
