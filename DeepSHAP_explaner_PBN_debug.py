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
    
    return torch.clamp(tensor, 0, 1)  # Ensure values are in [0,1]

def prepare_image_for_shap(image_tensor):
    """
    Prepare image tensor for SHAP visualization
    """
    # Denormalize from ImageNet normalization
    denorm_tensor = denormalize_image(image_tensor)
    
    # Convert to numpy and transpose to (H, W, C)
    if len(denorm_tensor.shape) == 4:  # Batch
        image_np = denorm_tensor[0].detach().cpu().numpy()
    else:  # Single image
        image_np = denorm_tensor.detach().cpu().numpy()
    
    # Transpose from (C, H, W) to (H, W, C)
    image_np = np.transpose(image_np, (1, 2, 0))
    
    return image_np

class ProtoPNetWrapper(nn.Module):
    """Wrapper class to make ProtoPNet compatible with SHAP by returning only logits and fixing in-place operations"""
    def __init__(self, model):
        super().__init__()
        self.original_model = model
        
        # Create a SHAP-compatible version by replacing in-place ReLU operations
        self.shap_compatible_model = self._create_shap_compatible_model(model)
        
    def _replace_inplace_relu(self, module):
        """Recursively replace in-place ReLU with non-in-place ReLU"""
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU) and child.inplace:
                setattr(module, name, nn.ReLU(inplace=False))
            else:
                self._replace_inplace_relu(child)
    
    def _create_shap_compatible_model(self, original_model):
        """Create a copy of the model with non-in-place operations"""
        import copy
        
        # Create a deep copy of the model
        shap_model = copy.deepcopy(original_model)
        
        # Replace all in-place ReLU operations
        self._replace_inplace_relu(shap_model)
        
        return shap_model
        
    def forward(self, x):
        logits, _ = self.shap_compatible_model(x)
        # print(f"Logits shape: {logits.shape}")
        return logits



def run_shap_analysis_for_baseline(model, model_name, test_image, test_image_id, test_label, 
                                     background, device, output_folder):
    
    print(f"Running SHAP analysis for {model_name} on image {test_image_id}...")
    
    model.eval()
    
    # Extract images from background data for baseline analysis
    if isinstance(background, list) and len(background) > 0 and isinstance(background[0], dict):
        # Background is list of batch dictionaries - extract just visual_embeds
        background_images = torch.stack([item['visual_embeds'] for item in background])
        print(f"Extracted {background_images.size(0)} images from background batch dictionaries")
    else:
        # Background is already tensor format (fallback)
        background_images = background
        print(f"Using tensor background directly")
    
    # Move to device and denormalize
    background_images = background_images.to(device)
    denorm_background = denormalize_image(background_images)
    explainer = shap.DeepExplainer(model, denorm_background)
    print(f"Using {model_name} for SHAP analysis")
    
    test_image_denorm = denormalize_image(test_image)
    
    # Compute SHAP values using the activation map
    try:
        print(f"Computing SHAP values...")
        model_output = model(test_image_denorm)
        
        # Handle different model output formats
        if isinstance(model_output, tuple):
            logits, _ = model_output
        else:
            logits = model_output
            
        shap_values = explainer.shap_values(test_image_denorm, check_additivity=False)
        print(f"SHAP values computed for test image")
    except Exception as shap_error:
        print(f"SHAP value calculation failed for {model_name}: {shap_error}")
        return False, f"SHAP value calculation failed: {shap_error}"

    # Simplify SHAP value processing
    if isinstance(shap_values, list):
        # Multiple outputs (one per class)
        target_class = test_label.item() if hasattr(test_label, 'item') else int(test_label)
        print(f"SHAP values is a list with {len(shap_values)} elements")
        print(f"Target class: {target_class}")
            
        if target_class < len(shap_values):
            shap_value = shap_values[target_class]
            print(f"SHAP value shape for class {target_class}: {shap_value.shape}")
        else:
            print(f"Target class {target_class} out of range (max: {len(shap_values)-1})")
            return False, f"Target class {target_class} out of range"
    else:
        # Single output
        shap_value = shap_values
        print(f"SHAP values is not a list, type: {type(shap_values)}")
        print(f"SHAP value shape: {shap_value.shape if hasattr(shap_value, 'shape') else 'No shape attribute'}")
        
        # Handle multi-dimensional SHAP values
        if hasattr(shap_value, 'shape') and len(shap_value.shape) >= 2:
            target_class = test_label.item() if hasattr(test_label, 'item') else int(test_label)
            print(f"Extracting for target class: {target_class}")
            # For baseline models, we might need to handle the output differently
            # Let's just use the raw SHAP values for now
    
    # Convert SHAP values to numpy and proper format for visualization
    # DETACH REASONING: SHAP values are PyTorch tensors with gradient tracking enabled.
    # We need to detach them to:
    # 1. Remove gradient information (not needed for visualization)
    # 2. Free up memory by breaking computational graph references
    # 3. Convert to numpy format that matplotlib can handle
    if hasattr(shap_value, 'detach'):
        shap_numpy = shap_value.detach().cpu().numpy()
    else:
        shap_numpy = shap_value
        
    # Handle different shapes and prepare for visualization
    # TRANSPOSE REASONING: PyTorch uses CHW (Channels, Height, Width) format 
    # but matplotlib expects HWC (Height, Width, Channels) format for image display.
    # The transpose operation (1, 2, 0) reorders dimensions as follows:
    # - Original axis 0 (channels) → becomes axis 2 (last dimension)
    # - Original axis 1 (height) → becomes axis 0 (first dimension)  
    # - Original axis 2 (width) → becomes axis 1 (second dimension)
    print(f"SHAP numpy shape before processing: {shap_numpy.shape}")
    
    # Handle different shapes and prepare for visualization
    if len(shap_numpy.shape) == 5:  # (batch, channels, height, width, classes)
        target_class = test_label.item() if hasattr(test_label, 'item') else int(test_label)
        print(f"Extracting attributions for target class: {target_class}")
        shap_for_class = shap_numpy[0, :, :, :, target_class]  # Extract target class: (channels, height, width)
        shap_for_plot = np.transpose(shap_for_class, (1, 2, 0))  # (H, W, C)
    elif len(shap_numpy.shape) == 4:  # (batch, channels, height, width)
        shap_for_plot = np.transpose(shap_numpy[0], (1, 2, 0))  # (H, W, C)
    elif len(shap_numpy.shape) == 3:  # (channels, height, width)
        # Transpose CHW → HWC directly
        shap_for_plot = np.transpose(shap_numpy, (1, 2, 0))
    else:
        print(f"Unexpected SHAP value shape after processing: {shap_numpy.shape}")
        return False, f"Unexpected SHAP value shape after processing: {shap_numpy.shape}"
    print(f"SHAP numpy shape after processing: {shap_numpy.shape}")
    
    # Prepare test image for visualization
    test_image_for_plot = prepare_image_for_shap(test_image)
    
    # Create detailed output folder structure
    date_str = datetime.now().strftime("%Y%m%d")
    model_output_folder = os.path.join(output_folder, "shap_images", date_str, model_name)
    os.makedirs(model_output_folder, exist_ok=True)
    
    # Create visualization
    try:
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
        axes[2].imshow(shap_sum, cmap='PiYG', alpha=0.7)
        axes[2].set_title('Original + SHAP Overlay')
        axes[2].axis('off')
        
        label_value = test_label.item() if hasattr(test_label, 'item') else int(test_label)
        plt.suptitle(f'SHAP Analysis: {model_name} - Image {test_image_id} (Label: {label_value})', fontsize=16)
        plt.tight_layout()
        
        # Save the plot
        plot_filename = f"{model_name}_image_{test_image_id}_label_{label_value}.png"
        plot_path = os.path.join(model_output_folder, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"SHAP analysis completed for {model_name} - Image {test_image_id}")
        return True, plot_path
        
    except Exception as plot_error:
        print(f"Plotting failed for {model_name}: {plot_error}")
        return False, f"Plotting failed: {plot_error}"
    
def run_shap_analysis_for_pbn(model, model_name, test_image, test_image_id, test_label, 
                                     background, device, output_folder):
    try:
        print(f"Running SHAP analysis for {model_name} on image {test_image_id}...")
        
        model.eval()
        
        if hasattr(model, 'module'):
            actual_model = model.module
        else:
            actual_model = model
        
        try:
            wrapped_model = ProtoPNetWrapper(actual_model)
            print(f"Type of wrapped_model: {type(wrapped_model)}")      
            # Extract images from background data for unimodal analysis
            if isinstance(background, list) and len(background) > 0 and isinstance(background[0], dict):
                # Background is list of batch dictionaries - extract just visual_embeds
                background_images = torch.stack([item['visual_embeds'] for item in background])
                print(f"Extracted {background_images.size(0)} images from background batch dictionaries")
            else:
                # Background is already tensor format (fallback)
                background_images = background
                print(f"Using tensor background directly")
            
            # Move to device and denormalize
            background_images = background_images.to(device)
            denorm_background = denormalize_image(background_images)
            explainer = shap.DeepExplainer(wrapped_model, denorm_background)
            print(f"Using ProtoPNetWrapper for {model_name}")
        except Exception as wrapper_error:
            print(f"ProtoPNetWrapper failed for {model_name}: {wrapper_error}")
            return False, f"ProtoPNetWrapper failed: {wrapper_error}"

        test_image_denorm = denormalize_image(test_image)
        print(f"Test image denorm shape: {test_image_denorm.shape}")
        
        # Compute SHAP values using the activation map
        try:
            print(f"Computing SHAP values...")
            shap_values = explainer.shap_values(test_image_denorm, check_additivity=False)
            print(f"SHAP values computed for test image")
        except Exception as shap_error:
            print(f"SHAP value calculation failed for {model_name}: {shap_error}")
            return False, f"SHAP value calculation failed: {shap_error}"
        
        # Simplify SHAP value processing
        if isinstance(shap_values, list):
            # Multiple outputs (one per class)
            target_class = test_label.item() if hasattr(test_label, 'item') else int(test_label)
                
            if target_class < len(shap_values):
                shap_value = shap_values[target_class]
                print(f"SHAP value shape for class {target_class}: {shap_value.shape}")
            else:
                print(f"Target class {target_class} out of range (max: {len(shap_values)-1})")
                return False, f"Target class {target_class} out of range"
        else:
            # Single output - could be all classes at once
            shap_value = shap_values
            print(f"Single SHAP value shape: {shap_value.shape}")
            
            # Handle 5D tensor: (batch, channels, height, width, classes)
            if len(shap_value.shape) == 5:
                target_class = test_label.item() if hasattr(test_label, 'item') else int(test_label)
                print(f"Extracting attributions for target class: {target_class}")
                
                # Extract attributions for the target class only
                shap_value = shap_value[..., target_class]  # Now shape: (batch, channels, height, width)
                print(f"Extracted SHAP value shape: {shap_value.shape}")
        
        # Convert SHAP values to numpy and proper format for visualization
        # DETACH REASONING: SHAP values are PyTorch tensors with gradient tracking enabled.
        # We need to detach them to:
        # 1. Remove gradient information (not needed for visualization)
        # 2. Free up memory by breaking computational graph references
        # 3. Convert to numpy format that matplotlib can handle
        if hasattr(shap_value, 'detach'):
            shap_numpy = shap_value.detach().cpu().numpy()
        else:
            shap_numpy = shap_value
            
        # Handle different shapes and prepare for visualization
        # TRANSPOSE REASONING: PyTorch uses CHW (Channels, Height, Width) format 
        # but matplotlib expects HWC (Height, Width, Channels) format for image display.
        # The transpose operation (1, 2, 0) reorders dimensions as follows:
        # - Original axis 0 (channels) → becomes axis 2 (last dimension)
        # - Original axis 1 (height) → becomes axis 0 (first dimension)  
        # - Original axis 2 (width) → becomes axis 1 (second dimension)
        print(f"SHAP numpy shape before processing: {shap_numpy.shape}")
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
            print(f"Unexpected SHAP value shape after processing: {shap_numpy.shape}")
            return False, f"Unexpected SHAP value shape after processing: {shap_numpy.shape}"
        print(f"SHAP numpy shape after processing: {shap_numpy.shape}")
        # Prepare test image for visualization
        # test_image = test_image.unsqueeze(0)
        test_image_for_plot = prepare_image_for_shap(test_image)
        
        # Create output folder
        date_str = datetime.now().strftime("%Y%m%d")
        model_output_folder = os.path.join(output_folder, "shap_images", date_str, model_name)
        os.makedirs(model_output_folder, exist_ok=True)
        
        # Create visualization
        print(f"Creating visualization...")
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
        axes[2].imshow(shap_sum, cmap='PiYG', alpha=0.7)
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
        # Get test image
        test_batch = test_data_to_analyze[img_idx]
        test_image = test_batch['visual_embeds'].unsqueeze(0).view(-1, 3, 224, 224)
        test_label = test_batch['label']
        test_image_id = test_data_to_analyze.get_image_id(img_idx)
        
        # Move test image to the same device as background
        if isinstance(test_image, tuple):
            test_image = test_image[0]
        print(f"Type of test_image after processing: {type(test_image)}")
        test_image = test_image.to(device)
        print(f"Type of test_image after moving to device: {type(test_image)}")
        print(f"Test image moved to device: {test_image.device}")
        
        print(f"\n{'='*60}")
        print(f"Processing image {img_idx+1}/{len(test_data_to_analyze)}: {test_image_id}")
        print(f"True label: {test_label}")
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
            
            if model_name.startswith("baseline"):
                success, result = run_shap_analysis_for_baseline(
                    model=model,
                    model_name=model_name,
                    test_image=test_image,
                    test_image_id=test_image_id,
                    test_label=test_label,
                    background=background,
                    device=device,
                    output_folder=output_folder
                )
            else:
                # Run SHAP analysis
                success, result = run_shap_analysis_for_pbn(
                    model=model,
                    model_name=model_name,
                    test_image=test_image,
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
        model_path_baseline = torch.load("/var/scratch/yyg760/results_final/baseline/vgg16_baseline_food101_best.pth", map_location=device)
        model_path_PBN1 = torch.load("/var/scratch/yyg760/results_final/PBN_vgg16_1p/PBN_vgg16_1p_best.pth", map_location=device)
        model_path_PBN2 = torch.load("/var/scratch/yyg760/results_final/PBN_vgg16_2p/PBN_vgg16_2p_best.pth", map_location=device)
        model_path_PBN5 = torch.load("/var/scratch/yyg760/results_final/PBN_vgg16_5p/PBN_vgg16_5p_best.pth", map_location=device)
        model_path_PBN10 = torch.load("/var/scratch/yyg760/results_final/PBN_vgg16_10p/PBN_vgg16_10p_best.pth", map_location=device)

        model_PBN1_config = model_path_PBN1['model_config']
        model_PBN2_config = model_path_PBN2['model_config']
        model_PBN5_config = model_path_PBN5['model_config']
        model_PBN10_config = model_path_PBN10['model_config']

        model_PBN1_state_dict = model_path_PBN1['state_dict']
        model_PBN2_state_dict = model_path_PBN2['state_dict']
        model_PBN5_state_dict = model_path_PBN5['state_dict']
        model_PBN10_state_dict = model_path_PBN10['state_dict']

        model_PBN1_new_state_dict = OrderedDict()
        for k, v in model_PBN1_state_dict.items():
            if k.startswith('module.'):
                name = k[7:] 
            else:
                name = k
            model_PBN1_new_state_dict[name] = v

        model_PBN2_new_state_dict = OrderedDict()
        for k, v in model_PBN2_state_dict.items():
            if k.startswith('module.'):
                name = k[7:] 
            else:
                name = k
            model_PBN2_new_state_dict[name] = v

        model_PBN5_new_state_dict = OrderedDict()
        for k, v in model_PBN5_state_dict.items():
            if k.startswith('module.'):
                name = k[7:] 
            else:
                name = k
            model_PBN5_new_state_dict[name] = v

        model_PBN10_new_state_dict = OrderedDict()
        for k, v in model_PBN10_state_dict.items():
            if k.startswith('module.'):
                name = k[7:] 
            else:
                name = k
            model_PBN10_new_state_dict[name] = v

        num_classes = test_data_full.get_num_classes()
        print(f"Number of classes: {num_classes}")

        model_baseline = VGG16(num_classes=num_classes).to(device)

        model_PBN1 = construct_PPNet(base_architecture=model_PBN1_config['base_architecture'], 
                                    pretrained=model_PBN1_config['pretrained'], 
                                    prototype_shape=model_PBN1_config['prototype_shape'], 
                                    num_classes=model_PBN1_config['num_classes'],
                                    add_on_layers_type=model_PBN1_config['add_on_layers_type'],
                                    img_size=model_PBN1_config['img_size'])
        
        model_PBN2 = construct_PPNet(base_architecture=model_PBN2_config['base_architecture'], 
                                    pretrained=model_PBN2_config['pretrained'], 
                                    prototype_shape=model_PBN2_config['prototype_shape'], 
                                    num_classes=model_PBN2_config['num_classes'],
                                    add_on_layers_type=model_PBN2_config['add_on_layers_type'],
                                    img_size=model_PBN2_config['img_size'])
        
        model_PBN5 = construct_PPNet(base_architecture=model_PBN5_config['base_architecture'], 
                                    pretrained=model_PBN5_config['pretrained'], 
                                    prototype_shape=model_PBN5_config['prototype_shape'], 
                                    num_classes=model_PBN5_config['num_classes'],
                                    add_on_layers_type=model_PBN5_config['add_on_layers_type'],
                                    img_size=model_PBN5_config['img_size'])

        model_PBN10 = construct_PPNet(base_architecture=model_PBN10_config['base_architecture'], 
                                    pretrained=model_PBN10_config['pretrained'], 
                                    prototype_shape=model_PBN10_config['prototype_shape'], 
                                    num_classes=model_PBN10_config['num_classes'],
                                    add_on_layers_type=model_PBN10_config['add_on_layers_type'],
                                    img_size=model_PBN10_config['img_size'])


        print("Loading model weights...")
        model_baseline.load_state_dict(model_path_baseline)
        model_PBN1.load_state_dict(model_PBN1_new_state_dict)
        model_PBN2.load_state_dict(model_PBN2_new_state_dict)
        model_PBN5.load_state_dict(model_PBN5_new_state_dict)
        model_PBN10.load_state_dict(model_PBN10_new_state_dict)

        model_PBN1 = torch.nn.DataParallel(model_PBN1)
        model_PBN2 = torch.nn.DataParallel(model_PBN2)
        model_PBN5 = torch.nn.DataParallel(model_PBN5)
        model_PBN10 = torch.nn.DataParallel(model_PBN10)
        
        print("Model weights loaded successfully!")

        # Prepare background data
        test_loader_full = DataLoader(test_data_full, batch_size=20, shuffle=True, collate_fn=collate_fn)

        # Select 10 random indices from the test dataset
        random_indices = random.sample(range(len(test_data_full)), 10)

        # Create a subset of the test dataset using these indices
        background_subset = Subset(test_data_full, random_indices)

        # Create a DataLoader for the background subset
        background_loader = DataLoader(background_subset, batch_size=20, shuffle=False, collate_fn=collate_fn)

        # Collect images from the DataLoader
        background_images = []
        for batch in background_loader:
            images = batch['visual_embeds']
            print(f"Type of images before processing: {type(images)}")
            if isinstance(images, tuple):
                print("Images is a tuple, extracting the first element.")
                images = images[0]
            print(f"Type of images after processing: {type(images)}")
            background_images.append(images)

        # Concatenate all images into a single tensor
        try:
            background_images = torch.cat(background_images, dim=0)
            print(f"Type of background_images after concatenation: {type(background_images)}")
            background_images = background_images.to(device)
            print(f"Background images moved to device: {background_images.device}")
        except Exception as e:
            print(f"Error when moving background_images to device: {e}")

        # Ensure the background is on the correct device
        background = background_images

        print(f"Background tensor shape: {background.shape}")
        print(f"Background tensor device: {background.device}")
        print(f"Background tensor range: [{background.min():.3f}, {background.max():.3f}]")

        models = {
            'baseline_VGG16': model_baseline,
            'PBN1_VGG16': model_PBN1,
            'PBN2_VGG16': model_PBN2,
            'PBN5_VGG16': model_PBN5,
            'PBN10_VGG16': model_PBN10,
        }

        # Run comprehensive SHAP analysis
        output_folder, results_summary = run(models, device, background, test_data_to_analyze)

    except Exception as e:
        print(f"Error occurred: {e}")
        traceback.print_exc()



