import torch, torchvision
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
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
from multimodal_PBN import VisualBertPPNet
from unimodal_ProtoPNet import *
import matplotlib.pyplot as plt
from datetime import datetime
import traceback

# Update the path to use the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
        # ProtoPNet returns (logits, min_distances), we only want logits for SHAP
        output = self.shap_compatible_model(x)
        if isinstance(output, tuple):
            return output[0]  # Return only logits
        return output

class SHAPCompatibleVisualBertPPNet(nn.Module):
    """SHAP-compatible version of VisualBertPPNet that avoids in-place operations"""
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model
        
        # Create a copy of VGG16 features with non-in-place ReLU
        vgg16 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
        vgg16_features = list(vgg16.features)
        
        # Replace in-place ReLU with non-in-place version
        for i, layer in enumerate(vgg16_features):
            if isinstance(layer, nn.ReLU) and layer.inplace:
                vgg16_features[i] = nn.ReLU(inplace=False)
        
        self.vgg16_features = nn.Sequential(*vgg16_features)
        
        # Copy weights from original model
        with torch.no_grad():
            for i, layer in enumerate(self.vgg16_features):
                if hasattr(layer, 'weight'):
                    # Find corresponding layer in original model
                    original_layer = original_model.vgg16_features[i]
                    if hasattr(original_layer, 'weight'):
                        layer.weight.copy_(original_layer.weight)
                        if hasattr(layer, 'bias') and layer.bias is not None:
                            layer.bias.copy_(original_layer.bias)
        
        # Copy other components
        self.encoder = original_model.encoder
        self.visual_projection = original_model.visual_projection
        self.prototype_vectors = original_model.prototype_vectors
        self.last_layer = original_model.last_layer
        self.prototype_class_identity = original_model.prototype_class_identity
        
    def forward(self, batch):
        # Handle visual embeddings
        visual_embeds = batch['visual_embeds']  # (B, 3, 224, 224)
        
        # Extract VGG16 features using non-in-place version
        visual_embeds = self.vgg16_features(visual_embeds)  # (B, 512, H', W')
        
        # Project through our projection pipeline
        visual_embeds = self.visual_projection[0](visual_embeds)  # AdaptiveAvgPool2d
        visual_embeds = self.visual_projection[1](visual_embeds)  # Flatten
        visual_embeds = self.visual_projection[2](visual_embeds)  # ReLU
        visual_embeds = self.visual_projection[3](visual_embeds)  # Linear projection
        visual_embeds = visual_embeds.unsqueeze(1)  # Add sequence dimension

        # Prepare inputs for VisualBERT
        inputs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'token_type_ids': batch['token_type_ids'],
            'visual_embeds': visual_embeds,
            'visual_attention_mask': batch['visual_attention_mask']
        }

        # Get VisualBERT output
        out = self.encoder(**inputs).last_hidden_state   # (B, L, H)
        cls = out[:, 0]                                # (B, H)
        
        # ---- ProtoPNet distance & logits ----
        dists = ((cls.unsqueeze(1) - self.prototype_vectors)**2).sum(-1)
        logits = -self.last_layer(dists)               # minus dist = similarity
        return logits, dists

class MultimodalModelWrapper(nn.Module):
    """Wrapper class to make multimodal model compatible with SHAP DeepExplainer"""
    def __init__(self, model, input_ids, attention_mask, token_type_ids):
        super().__init__()
        self.model = model
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input_ids = input_ids.to(device)
        self.attention_mask = attention_mask.to(device)
        self.token_type_ids = token_type_ids.to(device)
        
    def forward(self, visual_embeds):
        # Ensure input is properly cloned to avoid in-place modification issues
        if isinstance(visual_embeds, tuple):
            visual_embeds = visual_embeds[0]  
        visual_embeds = visual_embeds.clone()
        
        # Create batch dictionary with text inputs
        batch = {
            'visual_embeds': visual_embeds,
            'input_ids': self.input_ids.expand(visual_embeds.size(0), -1).to(device),
            'attention_mask': self.attention_mask.expand(visual_embeds.size(0), -1).to(device),
            'token_type_ids': self.token_type_ids.expand(visual_embeds.size(0), -1).to(device),
            'visual_attention_mask': torch.ones(visual_embeds.size(0), 1, dtype=torch.long, device=device)
        }
        
        # Forward pass through the multimodal model
        logits, dists = self.model(batch)
        return logits

def run_shap_analysis_for_model_image(model, model_name, test_image, test_image_id, test_label, 
                                     background, device, output_folder, dummy_inputs=None):
    """
    Run SHAP analysis for a single model-image pair and save the result.
    
    Args:
        model: The model to analyze
        model_name: Name of the model for saving
        test_image: Image tensor to analyze
        test_image_id: ID of the test image
        test_label: True label of the image
        background: Background images for SHAP
        device: Device to run on
        output_folder: Folder to save results
        dummy_inputs: Dummy inputs for multimodal models (if needed)
    """
    try:
        print(f"Running SHAP analysis for {model_name} on image {test_image_id}...")
        
        # Create model-specific explainer
        if model_name.startswith('mPBN'):
            # For multimodal models, try wrapper first, then fallback
            try:
                if dummy_inputs is not None:
                    wrapper = MultimodalModelWrapper(model, *dummy_inputs).to(device)
                    explainer = shap.DeepExplainer(wrapper, background[:5])  # Use smaller background
                else:
                    print(f"Skipping {model_name} - dummy inputs not provided")
                    return False, "Dummy inputs not provided"
            except Exception as wrapper_error:
                print(f"Wrapper failed for {model_name}, trying alternative approach: {wrapper_error}")
                # Try using GradientExplainer as fallback
                try:
                    explainer = shap.GradientExplainer(model, background[:3], batch_size=1)
                    print(f"Using GradientExplainer for {model_name}")
                except Exception as grad_error:
                    print(f"GradientExplainer also failed: {grad_error}")
                    return False, f"Both wrapper and GradientExplainer failed: {wrapper_error}, {grad_error}"
        elif model_name.startswith('PBN'):
            # For ProtoPNet models, wrap to return only logits
            try:
                wrapped_model = ProtoPNetWrapper(model).to(device)
                explainer = shap.DeepExplainer(wrapped_model, background)
                print(f"Using ProtoPNetWrapper for {model_name}")
            except Exception as wrapper_error:
                print(f"ProtoPNetWrapper failed for {model_name}: {wrapper_error}")
                print("Full traceback:")
                traceback.print_exc()
                return False, f"ProtoPNetWrapper failed: {wrapper_error}"
        else:
            # For unimodal baseline models, use directly
            try:
                explainer = shap.DeepExplainer(model, background)
                print(f"Using direct DeepExplainer for {model_name}")
            except Exception as explainer_error:
                print(f"Direct explainer failed for {model_name}: {explainer_error}")
                print("Full traceback:")
                traceback.print_exc()
                return False, f"Direct explainer failed: {explainer_error}"
        
        # Calculate SHAP values
        try:
            if hasattr(explainer, 'shap_values'):
                # DeepExplainer
                shap_values = explainer.shap_values(test_image, check_additivity=False)
            else:
                # GradientExplainer
                shap_values, indices = explainer.shap_values(test_image, ranked_outputs=1, nsamples=50)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]  # Take first output
        except Exception as shap_error:
            print(f"SHAP value calculation failed for {model_name}: {shap_error}")
            print("Full traceback:")
            traceback.print_exc()
            return False, f"SHAP value calculation failed: {shap_error}"
        
        # Convert to numpy for plotting - handle different output formats
        if isinstance(shap_values, list):
            # Multiple outputs (one per class) - select the target class
            # Handle different label types
            if hasattr(test_label, 'item'):
                target_class = test_label.item()
            elif isinstance(test_label, int):
                target_class = test_label
            else:
                target_class = int(test_label)
                
            if target_class < len(shap_values):
                shap_value = shap_values[target_class]  # Select the target class
                print(f"SHAP value shape for class {target_class}: {shap_value.shape}")
                
                # Convert to numpy if it's a tensor
                if hasattr(shap_value, 'numpy'):
                    shap_value_np = shap_value.detach().cpu().numpy() if shap_value.requires_grad else shap_value.cpu().numpy()
                else:
                    shap_value_np = shap_value
                
                if len(shap_value_np.shape) == 4:  # (batch, channels, height, width)
                    # Convert to (height, width, channels) for plotting
                    shap_numpy = [np.transpose(shap_value_np[0], (1, 2, 0))]
                elif len(shap_value_np.shape) == 5:  # (batch, channels, height, width, classes)
                    # This is the case we're seeing: (1, 3, 224, 224, 101)
                    # We need to select the target class and convert to (height, width, channels)
                    shap_value_class = shap_value_np[0, :, :, :, target_class]  # (3, 224, 224)
                    shap_numpy = [np.transpose(shap_value_class, (1, 2, 0))]  # (224, 224, 3)
                else:
                    print(f"Unexpected SHAP value shape for class {target_class}: {shap_value_np.shape}")
                    return False, f"Unexpected SHAP value shape for class {target_class}: {shap_value_np.shape}"
            else:
                print(f"Target class {target_class} out of range (max: {len(shap_values)-1})")
                return False, f"Target class {target_class} out of range (max: {len(shap_values)-1})"
        else:
            # Single output - this should be the target class already
            print(f"Single SHAP value shape: {shap_values.shape}")
            
            # Convert to numpy if it's a tensor
            if hasattr(shap_values, 'numpy'):
                shap_values_np = shap_values.detach().cpu().numpy() if shap_values.requires_grad else shap_values.cpu().numpy()
            else:
                shap_values_np = shap_values
                
            if len(shap_values_np.shape) == 4:  # (batch, channels, height, width)
                shap_numpy = [np.transpose(shap_values_np[0], (1, 2, 0))]
            elif len(shap_values_np.shape) == 5:  # (batch, channels, height, width, classes)
                # Select the target class
                if hasattr(test_label, 'item'):
                    target_class = test_label.item()
                elif isinstance(test_label, int):
                    target_class = test_label
                else:
                    target_class = int(test_label)
                    
                shap_value_class = shap_values_np[0, :, :, :, target_class]  # (3, 224, 224)
                shap_numpy = [np.transpose(shap_value_class, (1, 2, 0))]  # (224, 224, 3)
            else:
                print(f"Unexpected SHAP value shape: {shap_values_np.shape}")
                return False, f"Unexpected SHAP value shape: {shap_values_np.shape}"
        
        # Convert test image to numpy for plotting
        if hasattr(test_image, 'numpy'):
            test_numpy = np.transpose(test_image.detach().cpu().numpy()[0], (1, 2, 0))  # (height, width, channels)
        else:
            test_numpy = np.transpose(test_image[0], (1, 2, 0))  # (height, width, channels)
        
        # Create detailed output folder structure
        date_str = datetime.now().strftime("%Y%m%d")
        model_output_folder = os.path.join(output_folder, "shap_images", date_str, model_name)
        os.makedirs(model_output_folder, exist_ok=True)
        
        # Save SHAP plot
        plt.figure(figsize=(12, 8))
        shap.image_plot(shap_numpy, -test_numpy, show=False)
        
        # Handle different label types for title and filename
        if hasattr(test_label, 'item'):
            label_value = test_label.item()
        elif isinstance(test_label, int):
            label_value = test_label
        else:
            label_value = int(test_label)
            
        plt.title(f'SHAP Analysis: {model_name} - Image {test_image_id} (Label: {label_value})')
        plt.tight_layout()
        
        # Save the plot
        plot_filename = f"{model_name}_image_{test_image_id}_label_{label_value}.png"
        plot_path = os.path.join(model_output_folder, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"SHAP analysis completed for {model_name} - Image {test_image_id}")
        return True, plot_path
        
    except Exception as e:
        print(f"SHAP analysis failed for {model_name} - Image {test_image_id}: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return False, str(e)

def run_comprehensive_shap_analysis():
    """
    Run SHAP analysis for all models on all test images.
    """
    # Create output folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"shap_results_{timestamp}"
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Results will be saved to: {output_folder}")
    
    # Get global variables from main script
    global model_baseline, model_PBN1, model_PBN10, model_mPBN10, device, test_data_to_analyze, background
    
    # Define models to test
    models_to_test = {
        'baseline_VGG16': model_baseline,
        'PBN1_VGG16': model_PBN1,
        'PBN10_VGG16': model_PBN10,
        'mPBN10_VisualBERT': model_mPBN10
    }
    
    # Create dummy inputs for multimodal models
    dummy_input_ids = torch.zeros(1, 128, dtype=torch.long, device=device)
    dummy_attention_mask = torch.ones(1, 128, dtype=torch.long, device=device)
    dummy_token_type_ids = torch.zeros(1, 128, dtype=torch.long, device=device)
    dummy_inputs = (dummy_input_ids, dummy_attention_mask, dummy_token_type_ids)
    
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
        
        print(f"\n{'='*60}")
        print(f"Processing image {img_idx+1}/{len(test_data_to_analyze)}: {test_image_id}")
        print(f"True label: {test_label}")
        print(f"{'='*60}")
        
        for model_name, model in models_to_test.items():
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
            
            # Run SHAP analysis
            success, result = run_shap_analysis_for_model_image(
                model=model,
                model_name=model_name,
                test_image=test_image,
                test_image_id=test_image_id,
                test_label=test_label,
                background=background,
                device=device,
                output_folder=output_folder,
                dummy_inputs=dummy_inputs if model_name.startswith('mPBN') else None
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

try:
    print("Starting comprehensive SHAP analysis script...")
    
    print("Loading data...")

    with open("Food101/food-101/meta/test_full_annotation_customize.json", "r") as f:
        test_data_json = json.load(f)

    test_data_json = {k: v for d in test_data_json for k, v in d.items()}

    img_ids_to_analyze = ['2644276',
   '3421619',
   '302573',
   '2652167',
   '2997739',
   '2306539',
   '962203',]

    img_to_analyze = {k: v for k, v in test_data_json.items() if k in img_ids_to_analyze}

    test_data_full = f101.Food101Dataset(test_data_json, transform='vgg16', target_transform='integer', load_captions=True)
    test_data_to_analyze = f101.Food101Dataset(img_to_analyze, transform='vgg16', target_transform='integer', load_captions=True)

    print(f"Full dataset size: {len(test_data_full)}")
    print(f"Images to analyze: {len(test_data_to_analyze)}")

    print("Loading models...")
    model_path_baseline = "example_models_SHAPDEV/dummy_vgg16_baseline_food101_best.pth"
    model_path_PBN1 = "example_models_SHAPDEV/PBN_vgg16_1prototype_best.pth"
    model_path_PBN10 = "example_models_SHAPDEV/PBN_vgg16_10prototype_best.pth"
    model_path_mPBN10 = "example_models_SHAPDEV/dummy_mPBN_10prototype_best.pth"

    num_classes = test_data_full.get_num_classes()
    print(f"Number of classes: {num_classes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_baseline = VGG16(num_classes=num_classes).to(device)

    model_PBN1 = construct_PPNet(base_architecture='vgg16', 
                                pretrained=True, 
                                prototype_shape=(num_classes*1, 128, 7, 7), 
                                num_classes=num_classes,
                                add_on_layers_type='bottleneck',
                                img_size=224).to(device)

    model_PBN10 = construct_PPNet(base_architecture='vgg16', 
                                pretrained=True, 
                                prototype_shape=(num_classes*10, 128, 1, 1), 
                                num_classes=num_classes,
                                add_on_layers_type='bottleneck',
                                img_size=224).to(device)

    model_mPBN10 = VisualBertPPNet(num_prototypes=10, num_classes=num_classes).to(device)

    print("Loading model weights...")
    model_baseline.load_state_dict(torch.load(model_path_baseline, map_location=device))
    
    # Load ProtoPNet models with proper handling of save format
    try:
        pbn1_checkpoint = torch.load(model_path_PBN1, map_location=device)
        if 'state_dict' in pbn1_checkpoint:
            model_PBN1.load_state_dict(pbn1_checkpoint['state_dict'])
        else:
            model_PBN1.load_state_dict(pbn1_checkpoint)
        print("PBN1 model loaded successfully")
    except Exception as e:
        print(f"PBN1 model loading failed: {e}")
        model_PBN1 = None
    
    try:
        pbn10_checkpoint = torch.load(model_path_PBN10, map_location=device)
        if 'state_dict' in pbn10_checkpoint:
            model_PBN10.load_state_dict(pbn10_checkpoint['state_dict'])
        else:
            model_PBN10.load_state_dict(pbn10_checkpoint)
        print("PBN10 model loaded successfully")
    except Exception as e:
        print(f"PBN10 model loading failed: {e}")
        model_PBN10 = None
    
    try:
        mpbn10_checkpoint = torch.load(model_path_mPBN10, map_location=device)
        if 'state_dict' in mpbn10_checkpoint:
            model_mPBN10.load_state_dict(mpbn10_checkpoint['state_dict'])
        else:
            model_mPBN10.load_state_dict(mpbn10_checkpoint)
        print("mPBN10 model loaded successfully")
    except Exception as e:
        print(f"mPBN10 model loading failed: {e}")
        model_mPBN10 = None
    
    print("Model weights loaded successfully!")

    # Prepare background data
    test_loader_full = DataLoader(test_data_full, batch_size=20, shuffle=True)
    background_batch = next(iter(test_loader_full))
    background_images = background_batch['visual_embeds']
    background_images = background_images.view(-1, 3, 224, 224)
    background = background_images[:10].to(device)  # Use first 10 images as background

    print(f"Background tensor shape: {background.shape}")

    # Run comprehensive SHAP analysis
    output_folder, results_summary = run_comprehensive_shap_analysis()

except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()




