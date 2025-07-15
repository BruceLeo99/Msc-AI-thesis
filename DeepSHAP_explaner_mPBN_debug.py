import torch, torchvision
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import numpy as np
import os
import shap
import Food101_dataloader as f101
import time
import json
import sys
from vgg16_model import VGG16
from resnet34 import ResNet34
from multimodal_PBN import VisualBertPPNet, collate_fn
from unimodal_ProtoPNet import *
import matplotlib.pyplot as plt
from datetime import datetime
import traceback
import random
from collections import OrderedDict

import io

from transformers import AutoTokenizer
from multimodal_PBN import VisualBertPPNet

# Update the path to use the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ImageNet normalization constants (used by VGG16 transforms)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

def denormalize_image(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Denormalize an image tensor from ImageNet normalization to [0,1] range
    Args:
        tensor: Image tensor with shape (C, H, W) or (B, C, H, W)
        mean: Mean used for normalization 
        std: Std used for normalization
    """
    if len(tensor.shape) == 4:  # Batch dimension
        tensor = tensor.clone()
        for i in range(tensor.shape[0]):
            for c in range(3):
                tensor[i, c] = tensor[i, c] * std[c] + mean[c]
    else:  # Single image
        tensor = tensor.clone()
        for c in range(3):
            tensor[c] = tensor[c] * std[c] + mean[c]
    
    return torch.clamp(tensor, 0, 1)  


def prepare_image_for_shap(image_tensor):
    """
    Prepare image tensor for SHAP visualization
    """
    # Denormalize from ImageNet normalization
    denorm_tensor = denormalize_image(image_tensor)
    
    # Convert to numpy and transpose to (H, W, C)
    if len(denorm_tensor.shape) == 4:  
        image_np = denorm_tensor[0].detach().cpu().numpy()
    else: 
        image_np = denorm_tensor.detach().cpu().numpy()
    
    # Transpose from (C, H, W) to (H, W, C)
    image_np = np.transpose(image_np, (1, 2, 0))
    
    return image_np


class VisualBertProtoPNetWrapper(nn.Module):
    def __init__(self, model, background_batch):
        super().__init__()
        self.original_model = model
        
        # Create a SHAP-compatible version by replacing in-place operations
        self.shap_compatible_model = self._create_shap_compatible_model(model)
        
        # Store the background batch to use the same captions for all inputs
        # We'll use the first background sample's text components
        self.background_batch = background_batch[0]  # Use first sample as template
        
        # Get device from model parameters
        device = next(self.original_model.parameters()).device
        
        # Prepare static text components (same for all visual inputs)
        self.input_ids = self.background_batch['input_ids'].unsqueeze(0).to(device)
        self.attention_mask = self.background_batch['attention_mask'].unsqueeze(0).to(device)
        self.token_type_ids = self.background_batch['token_type_ids'].unsqueeze(0).to(device)
        self.visual_attention_mask = torch.ones(1, 1, dtype=torch.long, device=device)
    
    def _replace_inplace_operations(self, module):
        """Recursively replace in-place operations with non-in-place versions"""
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU) and child.inplace:
                setattr(module, name, nn.ReLU(inplace=False))
            elif isinstance(child, nn.ReLU6) and child.inplace:
                setattr(module, name, nn.ReLU6(inplace=False))
            elif isinstance(child, nn.LeakyReLU) and child.inplace:
                setattr(module, name, nn.LeakyReLU(child.negative_slope, inplace=False))
            else:
                self._replace_inplace_operations(child)
    
    def _create_shap_compatible_model(self, original_model):
        """Create a copy of the model with non-in-place operations"""
        import copy
        
        # Create a deep copy of the model
        shap_model = copy.deepcopy(original_model)
        
        # Replace all in-place operations
        self._replace_inplace_operations(shap_model)
        
        return shap_model

    def forward(self, visual_embeds):
        """
        Handle visual inputs directly and create dictionary format internally
        Args:
            visual_embeds: Tensor with shape (batch_size, 3, 224, 224) or (3, 224, 224)
        """
        # Get device from model parameters
        device = next(self.original_model.parameters()).device
        
        # Handle different input shapes
        if len(visual_embeds.shape) == 3:  # (3, 224, 224)
            visual_embeds = visual_embeds.unsqueeze(0)  # Add batch dimension
        
        batch_size = visual_embeds.shape[0]
        
        # Repeat text components for the batch size
        input_ids = self.input_ids.expand(batch_size, -1)
        attention_mask = self.attention_mask.expand(batch_size, -1)
        token_type_ids = self.token_type_ids.expand(batch_size, -1)
        visual_attention_mask = self.visual_attention_mask.expand(batch_size, -1)

        # Build batch dictionary for the multimodal model
        batch = {
            'visual_embeds': visual_embeds.to(device),
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'visual_attention_mask': visual_attention_mask
        }

        # Forward pass through the SHAP-compatible model
        # Note: Don't use torch.no_grad() here as SHAP needs gradients
        logits, _ = self.shap_compatible_model(batch)

        # Clone the output to avoid view/inplace operation issues with SHAP
        return logits.clone()

    
def run_shap_analysis_for_mpbn(model, model_name, test_batch, test_image_id, test_label,
                                     background, device, output_folder):
    
    try:
        print(f"Running SHAP analysis for {model_name} on image {test_image_id}...")
        
        # Extract caption from test batch
        if 'input_ids' in test_batch:
            print(f"Test data has tokenized captions")
        else:
            print("Warning: Test data doesn't have captions")
            return False, "Test data missing caption information"
        
        model.eval()
        
        if hasattr(model, 'module'):
            actual_model = model.module
        else:
            actual_model = model

        try:
            # Create wrapper for multimodal model
            wrapped_model = VisualBertProtoPNetWrapper(actual_model, background)
            wrapped_model.to(device)
            
            print(f"Type of wrapped_model: {type(wrapped_model)}")
            
            # Extract only visual components for SHAP analysis (this is what we want to visualize)
            print(f"Using {len(background)} background samples with actual captions")
            background_images = torch.stack([item['visual_embeds'] for item in background])
            print(f"Extracted visual components from background: {background_images.shape}")
            
            # Denormalize background images for SHAP
            background_images_denorm = denormalize_image(background_images)
            # Enable gradients for SHAP
            background_images_denorm.requires_grad_(True)
            
            # Create SHAP explainer with visual components only
            explainer = shap.DeepExplainer(wrapped_model, background_images_denorm)
            print(f"Using VisualBertProtoPNetWrapper for {model_name} with visual-only SHAP")
            
            # Prepare test data - extract visual component and denormalize
            test_image = test_batch['visual_embeds'].unsqueeze(0)
            test_image_denorm = denormalize_image(test_image)
            # Enable gradients for SHAP
            test_image_denorm.requires_grad_(True)
            
            # Compute SHAP values
            print(f"Computing SHAP values...")
            shap_values = explainer.shap_values(test_image_denorm, check_additivity=False)
            print(f"SHAP values computed")
            print(f"SHAP values type: {type(shap_values)}")
            
            # Process SHAP values (should be tensor format now since we're using visual inputs)
            if isinstance(shap_values, list):
                print(f"SHAP values is a list with {len(shap_values)} elements")
                target_class = test_label.item() if hasattr(test_label, 'item') else int(test_label)
                if target_class < len(shap_values):
                    shap_value = shap_values[target_class]
                    print(f"SHAP value shape for class {target_class}: {shap_value.shape if hasattr(shap_value, 'shape') else 'No shape'}")
                else:
                    print(f"Target class {target_class} out of range (max: {len(shap_values)-1})")
                    return False, f"Target class {target_class} out of range"
            else:
                print(f"SHAP values is not a list")
                shap_value = shap_values
                print(f"Single SHAP value shape: {shap_value.shape if hasattr(shap_value, 'shape') else 'No shape'}")
                
                # Handle 5D tensor if needed
                if hasattr(shap_value, 'shape') and len(shap_value.shape) == 5:
                    target_class = test_label.item() if hasattr(test_label, 'item') else int(test_label)
                    shap_value = shap_value[..., target_class]
                    print(f"Extracted SHAP value shape: {shap_value.shape}")
            
            # Convert to numpy for visualization
            # DETACH REASONING: SHAP values are PyTorch tensors with gradient tracking enabled.
            # We need to detach them to:
            # 1. Remove gradient information (not needed for visualization)
            # 2. Free up memory by breaking computational graph references
            # 3. Convert to numpy format that matplotlib can handle
            if hasattr(shap_value, 'detach'):
                shap_numpy = shap_value.detach().cpu().numpy()
            elif hasattr(shap_value, 'numpy'):
                shap_numpy = shap_value.numpy()
            else:
                shap_numpy = np.array(shap_value)
                
            # Prepare for visualization
            # TRANSPOSE REASONING: PyTorch uses CHW (Channels, Height, Width) format 
            # but matplotlib expects HWC (Height, Width, Channels) format for image display.
            # The transpose operation (1, 2, 0) reorders dimensions as follows:
            # - Original axis 0 (channels) → becomes axis 2 (last dimension)
            # - Original axis 1 (height) → becomes axis 0 (first dimension)  
            # - Original axis 2 (width) → becomes axis 1 (second dimension)
            print(f"SHAP numpy shape for multimodal: {shap_numpy.shape}")
            if len(shap_numpy.shape) == 5:  # (batch, channels, height, width, classes)
                # Extract attributions for specific target class, then transpose CHW → HWC
                target_class = test_label.item() if hasattr(test_label, 'item') else int(test_label)
                print(f"Extracting attributions for target class: {target_class}")
                shap_for_class = shap_numpy[0, :, :, :, target_class]  # Extract target class: (channels, height, width)
                shap_for_plot = np.transpose(shap_for_class, (1, 2, 0))  # (H, W, C)
            elif len(shap_numpy.shape) == 4:  # (batch, channels, height, width)
                # Extract first batch sample, then transpose CHW → HWC
                shap_for_plot = np.transpose(shap_numpy[0], (1, 2, 0))
            elif len(shap_numpy.shape) == 3:  # (channels, height, width)
                # Transpose CHW → HWC directly
                shap_for_plot = np.transpose(shap_numpy, (1, 2, 0))
            else:
                print(f"Unexpected SHAP value shape: {shap_numpy.shape}")
                return False, f"Unexpected SHAP value shape: {shap_numpy.shape}"
            
            # Prepare test image for visualization
            test_image_for_plot = prepare_image_for_shap(test_image)
            
            # Create output folder
            date_str = datetime.now().strftime("%Y%m%d")
            model_output_folder = os.path.join(output_folder, "shap_images", date_str, model_name)
            os.makedirs(model_output_folder, exist_ok=True)
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Original image
            axes[0].imshow(test_image_for_plot)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # SHAP attribution
            shap_sum = np.sum(np.abs(shap_for_plot), axis=2)
            im = axes[1].imshow(shap_sum, cmap='PiYG', alpha=0.7)
            axes[1].set_title(f'SHAP Attribution (Class {test_label})')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1])
            
            # Overlay
            axes[2].imshow(test_image_for_plot)
            axes[2].imshow(shap_sum, cmap='PiYG', alpha=0.5)
            axes[2].set_title('Original + SHAP Overlay')
            axes[2].axis('off')
            
            # Set title and save
            label_value = test_label.item() if hasattr(test_label, 'item') else int(test_label)
            plt.suptitle(f'SHAP Analysis: {model_name} - Image {test_image_id} (Label: {label_value})\nWith actual captions', fontsize=14)
            plt.tight_layout()
            
            plot_filename = f"{model_name}_image_{test_image_id}_label_{label_value}.png"
            plot_path = os.path.join(model_output_folder, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"SHAP analysis completed for {model_name} - Image {test_image_id}")
            return True, plot_path

        except Exception as wrapper_error:
            print(f"VisualBertProtoPNetWrapper failed for {model_name}: {wrapper_error}")
            return False, f"VisualBertProtoPNetWrapper failed: {wrapper_error}"

    except Exception as e:
        print(f"SHAP analysis failed for {model_name} - Image {test_image_id}: {e}")
        return False, str(e)


def run(models, device, background, test_data_to_analyze):
    """
    Run SHAP analysis for all models on all test images.
    """
    # Create output folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"shap_results_{timestamp}"
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Results will be saved to: {output_folder}")
    
    # Results tracking
    results_summary = {
        'total_tests': 0,
        'successful_tests': 0,
        'failed_tests': 0,
        'results': []
    }
    
    # Test each image with each model
    for img_idx in range(len(test_data_to_analyze)):
        # Get test data as full batch dictionary (contains captions)
        test_batch = test_data_to_analyze[img_idx]
        
        # Move test batch tensors to device
        test_batch_on_device = {}
        for key, value in test_batch.items():
            if isinstance(value, torch.Tensor):
                test_batch_on_device[key] = value.to(device)
            else:
                test_batch_on_device[key] = value
        
        test_image = test_batch_on_device['visual_embeds'].unsqueeze(0).view(-1, 3, 224, 224)
        test_label = test_batch_on_device['label']
        test_image_id = test_data_to_analyze.get_image_id(img_idx)
        
        # Move test image to device (redundant but explicit)
        if isinstance(test_image, tuple):
            test_image = test_image[0]
        test_image = test_image.to(device)
        
        print(f"\n{'='*60}")
        print(f"Processing image {img_idx+1}/{len(test_data_to_analyze)}: {test_image_id}")
        print(f"True label: {test_label}")
        print(f"Has captions: {'input_ids' in test_batch_on_device}")
        print(f"Test image shape: {test_image.shape}")
        print(f"Test image device: {test_image.device}")
        print(f"Test image range: [{test_image.min():.3f}, {test_image.max():.3f}]")
        print(f"{'='*60}")
        
        for model_name, model in models.items():
            results_summary['total_tests'] += 1
            
            # Check if model loaded successfully
            if model is None:
                print(f"Skipping {model_name} - model not loaded")
                results_summary['failed_tests'] += 1
                results_summary['results'].append({
                    'model': model_name,
                    'image_id': test_image_id,
                    'success': False,
                    'error': 'Model not loaded'
                })
                continue
            
            elif model_name.startswith("mPBN"):
                # Run multimodal SHAP analysis with full batch dictionary
                success, result = run_shap_analysis_for_mpbn(
                    model=model,
                    model_name=model_name,
                    test_batch=test_batch_on_device,  # Pass device-corrected batch dict
                    test_image_id=test_image_id,
                    test_label=test_label,
                    background=background,
                    device=device,
                    output_folder=output_folder
                )
                
            if success:
                results_summary['successful_tests'] += 1
                results_summary['results'].append({
                    'model': model_name,
                    'image_id': test_image_id,
                    'success': True,
                    'plot_path': result
                })
            else:
                results_summary['failed_tests'] += 1
                results_summary['results'].append({
                    'model': model_name,
                    'image_id': test_image_id,
                    'success': False,
                    'error': result
                })
    
    # Save results summary
    summary_path = os.path.join(output_folder, 'analysis_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETED")
    print(f"{'='*60}")
    print(f"Total tests: {results_summary['total_tests']}")
    print(f"Successful: {results_summary['successful_tests']}")
    print(f"Failed: {results_summary['failed_tests']}")
    print(f"Success rate: {results_summary['successful_tests']/results_summary['total_tests']*100:.1f}%")
    print(f"Results saved to: {output_folder}")
    print(f"Summary saved to: {summary_path}")
    
    return output_folder, results_summary


if __name__ == "__main__":
    try:
        print("Starting comprehensive SHAP analysis script...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        print("Loading data...")

        with open("Food101/food-101/meta/test_full_annotation_customize.json", "r") as f:
            test_data_json = json.load(f)

        test_data_json = {k: v for d in test_data_json for k, v in d.items()}

        img_ids_to_analyze = [
                '2644276',
                # '3421619',
                # '302573',
                # '2652167',
                # '2997739',
                # '2306539',
                # '962203'
                ]

        img_to_analyze = {k: v for k, v in test_data_json.items() if k in img_ids_to_analyze}

        test_data_full = f101.Food101Dataset(test_data_json, transform='vgg16', target_transform='integer', load_captions=True)
        test_data_to_analyze = f101.Food101Dataset(img_to_analyze, transform='vgg16', target_transform='integer', load_captions=True)

        print(f"Full dataset size: {len(test_data_full)}")
        print(f"Images to analyze: {len(test_data_to_analyze)}")

        print("Loading models...")

        model_path_mPBN1 = torch.load("/var/scratch/yyg760/results_final/mPBN_1p_blip2/mPBN_1p_blip2_best.pth", map_location=device)
        model_path_mPBN2 = torch.load("/var/scratch/yyg760/results_final/mPBN_2p_blip2/mPBN_2p_blip2_best.pth", map_location=device)
        model_path_mPBN5 = torch.load("/var/scratch/yyg760/results_final/mPBN_5p_blip2/mPBN_5p_blip2_best.pth", map_location=device)
        model_path_mPBN10 = torch.load("/var/scratch/yyg760/results_final/mPBN_10p_blip2/mPBN_10p_blip2_best.pth", map_location=device)

        model_mPBN1_config = model_path_mPBN1['model_config']
        model_mPBN2_config = model_path_mPBN2['model_config']
        model_mPBN5_config = model_path_mPBN5['model_config']
        model_mPBN10_config = model_path_mPBN10['model_config']

        model_mPBN1_state_dict = model_path_mPBN1['state_dict']
        model_mPBN2_state_dict = model_path_mPBN2['state_dict']
        model_mPBN5_state_dict = model_path_mPBN5['state_dict']
        model_mPBN10_state_dict = model_path_mPBN10['state_dict']

        model_mPBN1_new_state_dict = OrderedDict()
        for k, v in model_mPBN1_state_dict.items():
            if k.startswith('module.'):
                name = k[7:] 
            else:
                name = k
            model_mPBN1_new_state_dict[name] = v

        model_mPBN2_new_state_dict = OrderedDict()
        for k, v in model_mPBN2_state_dict.items():
            if k.startswith('module.'):
                name = k[7:] 
            else:
                name = k
            model_mPBN2_new_state_dict[name] = v

        model_mPBN5_new_state_dict = OrderedDict()
        for k, v in model_mPBN5_state_dict.items():
            if k.startswith('module.'):
                name = k[7:] 
            else:
                name = k
            model_mPBN5_new_state_dict[name] = v

        model_mPBN10_new_state_dict = OrderedDict()
        for k, v in model_mPBN10_state_dict.items():
            if k.startswith('module.'):
                name = k[7:] 
            else:
                name = k
            model_mPBN10_new_state_dict[name] = v

        num_classes = test_data_full.get_num_classes()
        print(f"Number of classes: {num_classes}")
        
        model_mPBN1 = VisualBertPPNet(num_prototypes_per_class=1, num_classes=num_classes)
        model_mPBN2 = VisualBertPPNet(num_prototypes_per_class=2, num_classes=num_classes)
        model_mPBN5 = VisualBertPPNet(num_prototypes_per_class=5, num_classes=num_classes)
        model_mPBN10 = VisualBertPPNet(num_prototypes_per_class=10, num_classes=num_classes)


        print("Loading model weights...")

        model_mPBN1.load_state_dict(model_mPBN1_new_state_dict)
        model_mPBN2.load_state_dict(model_mPBN2_new_state_dict)
        model_mPBN5.load_state_dict(model_mPBN5_new_state_dict)
        model_mPBN10.load_state_dict(model_mPBN10_new_state_dict)

        model_mPBN1 = torch.nn.DataParallel(model_mPBN1)
        model_mPBN2 = torch.nn.DataParallel(model_mPBN2)
        model_mPBN5 = torch.nn.DataParallel(model_mPBN5)
        model_mPBN10 = torch.nn.DataParallel(model_mPBN10)
        
        print("Model weights loaded successfully!")

        # Prepare background data using actual dataset items with captions
        print("Collecting background data...")
        background_data = []
        max_background_samples = 10  # Limit background samples for efficiency
        
        # Use test_data_full to get background samples with captions
        for i in range(min(max_background_samples, len(test_data_full))):
            sample = test_data_full[i]  # This returns the full batch dict with captions
            
            # Move all tensor components to device
            sample_on_device = {}
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    sample_on_device[key] = value.to(device)
                else:
                    sample_on_device[key] = value
            
            background_data.append(sample_on_device)

        print(f"Background data collected: {len(background_data)} samples")
        if len(background_data) > 0:
            print(f"First background sample keys: {list(background_data[0].keys())}")
            print(f"Visual embeds shape: {background_data[0]['visual_embeds'].shape}")
            print(f"Visual embeds device: {background_data[0]['visual_embeds'].device}")
            print(f"Input IDs shape: {background_data[0]['input_ids'].shape}")
            print(f"Input IDs device: {background_data[0]['input_ids'].device}")
            print(f"Has captions: {'input_ids' in background_data[0]}")

        # Set background as the list of batch dictionaries
        background = background_data

        # Run comprehensive SHAP analysis

        models = {
            'mPBN_1p': model_mPBN1,
            'mPBN_2p': model_mPBN2,
            'mPBN_5p': model_mPBN5,
            'mPBN_10p': model_mPBN10
        }

        output_folder, results_summary = run(models, device, background, test_data_to_analyze)

    except Exception as e:
        print(f"Error occurred: {e}")
        traceback.print_exc()



