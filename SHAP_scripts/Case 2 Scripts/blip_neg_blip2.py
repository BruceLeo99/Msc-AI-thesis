import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DeepSHAP_explaner_mPBN_batched import *

output_folder = f"/var/scratch/yyg760/SHAP_results/SHAP_blip_neg_blip2"
test_img_ids_filepath1 = "blip_neg_1p.txt"
test_img_ids_filepath2 = "blip_neg_2p.txt"
test_img_ids_filepath3 = "blip_neg_5p.txt"
test_img_ids_filepath4 = "blip_neg_10p.txt"

caption_type = "blip2"

background_img_ids_filepath = "background_images_202.txt"

os.makedirs(output_folder, exist_ok=True)

print("Starting comprehensive SHAP analysis script...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading data...")

# Load background image IDs from file
background_img_ids = []
with open(background_img_ids_filepath, "r") as f:
    background_img_ids = f.readlines()
    background_img_ids = [img_id.strip() for img_id in background_img_ids]

# Load test image IDs from file
img_ids_to_analyze1 = []
with open(test_img_ids_filepath1, "r") as f:
    img_ids_to_analyze1 = f.readlines()
    img_ids_to_analyze1 = [img_id.strip() for img_id in img_ids_to_analyze1]

img_ids_to_analyze2 = []
with open(test_img_ids_filepath2, "r") as f:
    img_ids_to_analyze2 = f.readlines()
    img_ids_to_analyze2 = [img_id.strip() for img_id in img_ids_to_analyze2]

img_ids_to_analyze3 = []
with open(test_img_ids_filepath3, "r") as f:
    img_ids_to_analyze3 = f.readlines()
    img_ids_to_analyze3 = [img_id.strip() for img_id in img_ids_to_analyze3]

img_ids_to_analyze4 = []
with open(test_img_ids_filepath4, "r") as f:
    img_ids_to_analyze4 = f.readlines()
    img_ids_to_analyze4 = [img_id.strip() for img_id in img_ids_to_analyze4]

# Load test data for analysis
with open("Food101/food-101/meta/test_full_annotation_customize.json", "r") as f:
    test_data_json = json.load(f)
test_data_json = {k: v for d in test_data_json for k, v in d.items()}

img_to_analyze1 = {k: v for k, v in test_data_json.items() if k in img_ids_to_analyze1}
img_to_analyze2 = {k: v for k, v in test_data_json.items() if k in img_ids_to_analyze2}
img_to_analyze3 = {k: v for k, v in test_data_json.items() if k in img_ids_to_analyze3}
img_to_analyze4 = {k: v for k, v in test_data_json.items() if k in img_ids_to_analyze4}

test_data_full_blip2 = f101.Food101Dataset(test_data_json, transform='vgg16', target_transform='integer', load_captions=True, caption_type="blip2")
test_data_blip2_1p = f101.Food101Dataset(img_to_analyze1, transform='vgg16', target_transform='integer', load_captions=True, caption_type="blip2")
test_data_blip2_2p = f101.Food101Dataset(img_to_analyze2, transform='vgg16', target_transform='integer', load_captions=True, caption_type="blip2")
test_data_blip2_5p = f101.Food101Dataset(img_to_analyze3, transform='vgg16', target_transform='integer', load_captions=True, caption_type="blip2")
test_data_blip2_10p = f101.Food101Dataset(img_to_analyze4, transform='vgg16', target_transform='integer', load_captions=True, caption_type="blip2")

print(f"Full dataset size: {len(test_data_full_blip2)}")
print(f"Images to analyze: {len(test_data_blip2_1p)}")
print(f"Images to analyze: {len(test_data_blip2_2p)}")
print(f"Images to analyze: {len(test_data_blip2_5p)}")
print(f"Images to analyze: {len(test_data_blip2_10p)}")

print("Loading models...")

model_path_mPBN1_blip2 = torch.load(f"/var/scratch/yyg760/results_final/mPBN_1p_{caption_type}/mPBN_1p_{caption_type}_best.pth", map_location=device)
model_path_mPBN2_blip2 = torch.load(f"/var/scratch/yyg760/results_final/mPBN_2p_{caption_type}/mPBN_2p_{caption_type}_best.pth", map_location=device)
model_path_mPBN5_blip2 = torch.load(f"/var/scratch/yyg760/results_final/mPBN_5p_{caption_type}/mPBN_5p_{caption_type}_best.pth", map_location=device)
model_path_mPBN10_blip2 = torch.load(f"/var/scratch/yyg760/results_final/mPBN_10p_{caption_type}/mPBN_10p_{caption_type}_best.pth", map_location=device)

model_mPBN1_blip2_config = model_path_mPBN1_blip2['model_config']
model_mPBN2_blip2_config = model_path_mPBN2_blip2['model_config']
model_mPBN5_blip2_config = model_path_mPBN5_blip2['model_config']
model_mPBN10_blip2_config = model_path_mPBN10_blip2['model_config']

model_mPBN1_blip2_state_dict = model_path_mPBN1_blip2['state_dict']
model_mPBN2_blip2_state_dict = model_path_mPBN2_blip2['state_dict']
model_mPBN5_blip2_state_dict = model_path_mPBN5_blip2['state_dict']
model_mPBN10_blip2_state_dict = model_path_mPBN10_blip2['state_dict']

model_mPBN1_blip2_new_state_dict = OrderedDict()
for k, v in model_mPBN1_blip2_state_dict.items():
    if k.startswith('module.'):
        name = k[7:] 
    else:
        name = k
    model_mPBN1_blip2_new_state_dict[name] = v

model_mPBN2_blip2_new_state_dict = OrderedDict()
for k, v in model_mPBN2_blip2_state_dict.items():
    if k.startswith('module.'):
        name = k[7:] 
    else:
        name = k
    model_mPBN2_blip2_new_state_dict[name] = v

model_mPBN5_blip2_new_state_dict = OrderedDict()
for k, v in model_mPBN5_blip2_state_dict.items():
    if k.startswith('module.'):
        name = k[7:] 
    else:
        name = k
    model_mPBN5_blip2_new_state_dict[name] = v

model_mPBN10_blip2_new_state_dict = OrderedDict()
for k, v in model_mPBN10_blip2_state_dict.items():
    if k.startswith('module.'):
        name = k[7:] 
    else:
        name = k
    model_mPBN10_blip2_new_state_dict[name] = v

num_classes = test_data_full_blip2.get_num_classes()
print(f"Number of classes: {num_classes}")

model_mPBN1_blip2 = VisualBertPPNet(num_prototypes_per_class=1, num_classes=num_classes)
model_mPBN2_blip2 = VisualBertPPNet(num_prototypes_per_class=2, num_classes=num_classes)
model_mPBN5_blip2 = VisualBertPPNet(num_prototypes_per_class=5, num_classes=num_classes)
model_mPBN10_blip2 = VisualBertPPNet(num_prototypes_per_class=10, num_classes=num_classes)


print("Loading model weights...")

model_mPBN1_blip2.load_state_dict(model_mPBN1_blip2_new_state_dict)
model_mPBN2_blip2.load_state_dict(model_mPBN2_blip2_new_state_dict)
model_mPBN5_blip2.load_state_dict(model_mPBN5_blip2_new_state_dict)
model_mPBN10_blip2.load_state_dict(model_mPBN10_blip2_new_state_dict)

model_mPBN1_blip2 = torch.nn.DataParallel(model_mPBN1_blip2)
model_mPBN2_blip2 = torch.nn.DataParallel(model_mPBN2_blip2)
model_mPBN5_blip2 = torch.nn.DataParallel(model_mPBN5_blip2)
model_mPBN10_blip2 = torch.nn.DataParallel(model_mPBN10_blip2)

models_1p_blip2 = {
    'mPBN_1p_blip2': model_mPBN1_blip2,
}

models_2p_blip2 = {
    'mPBN_2p_blip2': model_mPBN2_blip2,
}
models_5p_blip2 = {
    'mPBN_5p_blip2': model_mPBN5_blip2,
}
models_10p_blip2 = {
    'mPBN_10p_blip2': model_mPBN10_blip2,
}

print("Model weights loaded successfully!")


print("Loading background images...")
# Load background dataset from val dataset (since background_images_202.txt contains val IDs)
background_dataset_blip2 = load_background_from_image_ids(
    image_ids=background_img_ids,
    dataset_type='val',  # Change to 'test' if your background IDs are from test set
    transform='vgg16',
    target_transform='integer',
    load_captions=True,
    caption_type=caption_type
)

# For mPBN, we need the full batch dictionaries (with captions), not just tensors
print("Collecting background data with captions...")
background_data_blip2 = []
for i in range(len(background_dataset_blip2)):
    sample = background_dataset_blip2[i]  # This returns the full batch dict with captions
    
    # Move all tensor components to device
    sample_on_device = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            sample_on_device[key] = value.to(device)
        else:
            sample_on_device[key] = value
    
    background_data_blip2.append(sample_on_device)

print(f"Background data collected: {len(background_data_blip2)} samples with captions")
if len(background_data_blip2) > 0:
    print(f"First background sample keys: {list(background_data_blip2[0].keys())}")
    print(f"Visual embeds shape: {background_data_blip2[0]['visual_embeds'].shape}")
    print(f"Input IDs shape: {background_data_blip2[0]['input_ids'].shape}")

# Set background as the list of batch dictionaries
background_blip2 = background_data_blip2

print(f"Background Images Loaded. Number of samples: {len(background_blip2)}")


os.makedirs(output_folder, exist_ok=True)

# Configure batch size for background processing (adjust based on GPU memory)
background_batch_size = 15  # Start conservative, can increase if memory allows

# Run comprehensive SHAP analysis with batched processing
print("Star SHAP analysis...")
output_folder, results_summary = run(models_1p_blip2, device, background_blip2, test_data_blip2_1p, output_folder, background_batch_size) 
print(f"Analysis completed for image IDs: {img_ids_to_analyze1}") 

output_folder, results_summary = run(models_2p_blip2, device, background_blip2, test_data_blip2_2p, output_folder, background_batch_size) 
print(f"Analysis completed for image IDs: {img_ids_to_analyze2}") 

output_folder, results_summary = run(models_5p_blip2, device, background_blip2, test_data_blip2_5p, output_folder, background_batch_size) 
print(f"Analysis completed for image IDs: {img_ids_to_analyze3}") 

output_folder, results_summary = run(models_10p_blip2, device, background_blip2, test_data_blip2_10p, output_folder, background_batch_size) 
print(f"Analysis completed for image IDs: {img_ids_to_analyze4}") 