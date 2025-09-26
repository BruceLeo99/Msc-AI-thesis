import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DeepSHAP_explaner_mPBN_batched import *

output_folder = f"/var/scratch/yyg760/SHAP_results/Case 1/mpbn_pos_5p_blip2"
test_img_ids_filepath1 = "mpbn_pos_1p_blip.txt"
test_img_ids_filepath2 = "mpbn_pos_2p_blip.txt"
test_img_ids_filepath3 = "mpbn_pos_5p_blip.txt"
test_img_ids_filepath4 = "mpbn_pos_10p_blip.txt"
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

test_data_full = f101.Food101Dataset(test_data_json, transform='vgg16', target_transform='integer', load_captions=True, caption_type=caption_type)
test_data_to_analyze1 = f101.Food101Dataset(img_to_analyze1, transform='vgg16', target_transform='integer', load_captions=True, caption_type=caption_type)
test_data_to_analyze2 = f101.Food101Dataset(img_to_analyze2, transform='vgg16', target_transform='integer', load_captions=True, caption_type=caption_type)
test_data_to_analyze3 = f101.Food101Dataset(img_to_analyze3, transform='vgg16', target_transform='integer', load_captions=True, caption_type=caption_type)
test_data_to_analyze4 = f101.Food101Dataset(img_to_analyze4, transform='vgg16', target_transform='integer', load_captions=True, caption_type=caption_type)

print(f"Full dataset size: {len(test_data_full)}")
print(f"Images to analyze: {len(test_data_to_analyze1)}")
print(f"Images to analyze: {len(test_data_to_analyze2)}")
print(f"Images to analyze: {len(test_data_to_analyze3)}")
print(f"Images to analyze: {len(test_data_to_analyze4)}")

print("Loading models...")

model_path_mPBN1 = torch.load(f"/var/scratch/yyg760/results_final/mPBN_1p_{caption_type}/mPBN_1p_{caption_type}_best.pth", map_location=device)
model_path_mPBN2 = torch.load(f"/var/scratch/yyg760/results_final/mPBN_2p_{caption_type}/mPBN_2p_{caption_type}_best.pth", map_location=device)
model_path_mPBN5 = torch.load(f"/var/scratch/yyg760/results_final/mPBN_5p_{caption_type}/mPBN_5p_{caption_type}_best.pth", map_location=device)
model_path_mPBN10 = torch.load(f"/var/scratch/yyg760/results_final/mPBN_10p_{caption_type}/mPBN_10p_{caption_type}_best.pth", map_location=device)

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

models_1p = {
    'mPBN_1p_blip': model_mPBN1,
}

models_2p = {
    'mPBN_2p_blip': model_mPBN2,
}
models_5p = {
    'mPBN_5p_blip': model_mPBN5,
}
models_10p = {
    'mPBN_10p_blip': model_mPBN10,
}

print("Model weights loaded successfully!")


print("Loading background images...")
# Load background dataset from val dataset (since background_images_202.txt contains val IDs)
background_dataset = load_background_from_image_ids(
    image_ids=background_img_ids,
    dataset_type='val',  # Change to 'test' if your background IDs are from test set
    transform='vgg16',
    target_transform='integer',
    load_captions=True,
    caption_type=caption_type
)

# For mPBN, we need the full batch dictionaries (with captions), not just tensors
print("Collecting background data with captions...")
background_data = []
for i in range(len(background_dataset)):
    sample = background_dataset[i]  # This returns the full batch dict with captions
    
    # Move all tensor components to device
    sample_on_device = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            sample_on_device[key] = value.to(device)
        else:
            sample_on_device[key] = value
    
    background_data.append(sample_on_device)

print(f"Background data collected: {len(background_data)} samples with captions")
if len(background_data) > 0:
    print(f"First background sample keys: {list(background_data[0].keys())}")
    print(f"Visual embeds shape: {background_data[0]['visual_embeds'].shape}")
    print(f"Input IDs shape: {background_data[0]['input_ids'].shape}")

# Set background as the list of batch dictionaries
background = background_data

print(f"Background Images Loaded. Number of samples: {len(background)}")


os.makedirs(output_folder, exist_ok=True)

# Configure batch size for background processing (adjust based on GPU memory)
background_batch_size = 15  # Start conservative, can increase if memory allows

# Run comprehensive SHAP analysis with batched processing
print("Star SHAP analysis...")
# output_folder, results_summary = run(models_1p, device, background, test_data_to_analyze1, output_folder, background_batch_size) 
# print(f"Analysis completed for image IDs: {img_ids_to_analyze1}") 

# output_folder, results_summary = run(models_2p, device, background, test_data_to_analyze2, output_folder, background_batch_size) 
# print(f"Analysis completed for image IDs: {img_ids_to_analyze2}") 

output_folder, results_summary = run(models_5p, device, background, test_data_to_analyze3, output_folder, background_batch_size) 
print(f"Analysis completed for image IDs: {img_ids_to_analyze3}") 

# output_folder, results_summary = run(models_10p, device, background, test_data_to_analyze4, output_folder, background_batch_size) 
# print(f"Analysis completed for image IDs: {img_ids_to_analyze4}") 