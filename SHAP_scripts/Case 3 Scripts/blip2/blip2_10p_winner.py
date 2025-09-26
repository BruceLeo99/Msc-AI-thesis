import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DeepSHAP_explaner_mPBN_batched import *

main_output_folder = f"/var/scratch/yyg760/SHAP_results/Case 3/blip2/"
os.makedirs(main_output_folder, exist_ok=True)

# test_img_ids_filepath1 = "blip2_1p_vs_2p.txt"
# test_img_ids_filepath2 = "blip2_1p_vs_5p.txt"
# test_img_ids_filepath3 = "blip2_1p_vs_10p.txt"

# test_img_ids_filepath4 = "blip2_2p_vs_1p.txt"
# test_img_ids_filepath5 = "blip2_2p_vs_5p.txt"
# test_img_ids_filepath6 = "blip2_2p_vs_10p.txt"

# test_img_ids_filepath7 = "blip2_5p_vs_1p.txt"
# test_img_ids_filepath8 = "blip2_5p_vs_2p.txt"
# test_img_ids_filepath9 = "blip2_5p_vs_10p.txt"

test_img_ids_filepath10 = "blip2_10p_vs_1p.txt"
test_img_ids_filepath11 = "blip2_10p_vs_2p.txt"
test_img_ids_filepath12 = "blip2_10p_vs_5p.txt"

# sub_output_folder1 = f"{main_output_folder}/mpbn_1p_blip2_winner/vs_2p/"
# os.makedirs(sub_output_folder1, exist_ok=True)

# sub_output_folder2 = f"{main_output_folder}/mpbn_1p_blip2_winner/vs_5p/"
# os.makedirs(sub_output_folder2, exist_ok=True)

# sub_output_folder3 = f"{main_output_folder}/mpbn_1p_blip2_winner/vs_10p/"
# os.makedirs(sub_output_folder3, exist_ok=True)

# sub_output_folder4 = f"{main_output_folder}/mpbn_2p_blip2_winner/vs_1p/"
# os.makedirs(sub_output_folder4, exist_ok=True)

# sub_output_folder5 = f"{main_output_folder}/mpbn_2p_blip2_winner/vs_5p/"
# os.makedirs(sub_output_folder5, exist_ok=True)

# sub_output_folder6 = f"{main_output_folder}/mpbn_2p_blip2_winner/vs_10p/"
# os.makedirs(sub_output_folder6, exist_ok=True)

# sub_output_folder7 = f"{main_output_folder}/mpbn_5p_blip2_winner/vs_1p/"
# os.makedirs(sub_output_folder7, exist_ok=True)

# sub_output_folder8 = f"{main_output_folder}/mpbn_5p_blip2_winner/vs_2p/"
# os.makedirs(sub_output_folder8, exist_ok=True)

# sub_output_folder9 = f"{main_output_folder}/mpbn_5p_blip2_winner/vs_10p/"
# os.makedirs(sub_output_folder9, exist_ok=True)

sub_output_folder10 = f"{main_output_folder}/mpbn_10p_blip2_winner/vs_1p/"
os.makedirs(sub_output_folder10, exist_ok=True)

sub_output_folder11 = f"{main_output_folder}/mpbn_10p_blip2_winner/vs_2p/"
os.makedirs(sub_output_folder11, exist_ok=True)

sub_output_folder12 = f"{main_output_folder}/mpbn_10p_blip2_winner/vs_5p/"
os.makedirs(sub_output_folder12, exist_ok=True)

caption_type = "blip2"

background_img_ids_filepath = "background_images_202.txt"

def get_list_of_ids(filepath):
    img_id_list = []
    with open(filepath, "r") as f:
        img_id_list = f.readlines()
        img_id_list = [img_id.strip() for img_id in img_id_list]
    return img_id_list

print("Starting comprehensive SHAP analysis script...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading data...")

# Load background image IDs from file
background_img_ids = get_list_of_ids(background_img_ids_filepath)

# Load test image IDs from file
# img_ids_to_analyze1 = get_list_of_ids(test_img_ids_filepath1)
# img_ids_to_analyze2 = get_list_of_ids(test_img_ids_filepath2)
# img_ids_to_analyze3 = get_list_of_ids(test_img_ids_filepath3)
# img_ids_to_analyze4 = get_list_of_ids(test_img_ids_filepath4)
# img_ids_to_analyze5 = get_list_of_ids(test_img_ids_filepath5)
# img_ids_to_analyze6 = get_list_of_ids(test_img_ids_filepath6)
# img_ids_to_analyze7 = get_list_of_ids(test_img_ids_filepath7)
# img_ids_to_analyze8 = get_list_of_ids(test_img_ids_filepath8)
# img_ids_to_analyze9 = get_list_of_ids(test_img_ids_filepath9)
img_ids_to_analyze10 = get_list_of_ids(test_img_ids_filepath10)
img_ids_to_analyze11 = get_list_of_ids(test_img_ids_filepath11)
img_ids_to_analyze12 = get_list_of_ids(test_img_ids_filepath12)

# Load test data for analysis
with open("Food101/food-101/meta/test_full_annotation_customize.json", "r") as f:
    test_data_json = json.load(f)
test_data_json = {k: v for d in test_data_json for k, v in d.items()}

# img_to_analyze1 = {k: v for k, v in test_data_json.items() if k in img_ids_to_analyze1}
# img_to_analyze2 = {k: v for k, v in test_data_json.items() if k in img_ids_to_analyze2}
# img_to_analyze3 = {k: v for k, v in test_data_json.items() if k in img_ids_to_analyze3}
# img_to_analyze4 = {k: v for k, v in test_data_json.items() if k in img_ids_to_analyze4}
# img_to_analyze5 = {k: v for k, v in test_data_json.items() if k in img_ids_to_analyze5}
# img_to_analyze6 = {k: v for k, v in test_data_json.items() if k in img_ids_to_analyze6}
# img_to_analyze7 = {k: v for k, v in test_data_json.items() if k in img_ids_to_analyze7}
# img_to_analyze8 = {k: v for k, v in test_data_json.items() if k in img_ids_to_analyze8}
# img_to_analyze9 = {k: v for k, v in test_data_json.items() if k in img_ids_to_analyze9}
img_to_analyze10 = {k: v for k, v in test_data_json.items() if k in img_ids_to_analyze10}
img_to_analyze11 = {k: v for k, v in test_data_json.items() if k in img_ids_to_analyze11}
img_to_analyze12 = {k: v for k, v in test_data_json.items() if k in img_ids_to_analyze12}

test_data_full_blip2 = f101.Food101Dataset(test_data_json, transform='vgg16', target_transform='integer', load_captions=True, caption_type=caption_type)
# test_data_1 = f101.Food101Dataset(img_to_analyze1, transform='vgg16', target_transform='integer', load_captions=True, caption_type=caption_type)
# test_data_2 = f101.Food101Dataset(img_to_analyze2, transform='vgg16', target_transform='integer', load_captions=True, caption_type=caption_type)
# test_data_3 = f101.Food101Dataset(img_to_analyze3, transform='vgg16', target_transform='integer', load_captions=True, caption_type=caption_type)
# test_data_4 = f101.Food101Dataset(img_to_analyze4, transform='vgg16', target_transform='integer', load_captions=True, caption_type=caption_type)
# test_data_5 = f101.Food101Dataset(img_to_analyze5, transform='vgg16', target_transform='integer', load_captions=True, caption_type=caption_type)
# test_data_6 = f101.Food101Dataset(img_to_analyze6, transform='vgg16', target_transform='integer', load_captions=True, caption_type=caption_type)
# test_data_7 = f101.Food101Dataset(img_to_analyze7, transform='vgg16', target_transform='integer', load_captions=True, caption_type=caption_type)
# test_data_8 = f101.Food101Dataset(img_to_analyze8, transform='vgg16', target_transform='integer', load_captions=True, caption_type=caption_type)
# test_data_9 = f101.Food101Dataset(img_to_analyze9, transform='vgg16', target_transform='integer', load_captions=True, caption_type=caption_type)
test_data_10 = f101.Food101Dataset(img_to_analyze10, transform='vgg16', target_transform='integer', load_captions=True, caption_type=caption_type)
test_data_11 = f101.Food101Dataset(img_to_analyze11, transform='vgg16', target_transform='integer', load_captions=True, caption_type=caption_type)
test_data_12 = f101.Food101Dataset(img_to_analyze12, transform='vgg16', target_transform='integer', load_captions=True, caption_type=caption_type)

print(f"Full dataset size: {len(test_data_full_blip2)}")
# print(f"Images to analyze: {len(test_data_1)}")
# print(f"Images to analyze: {len(test_data_2)}")
# print(f"Images to analyze: {len(test_data_3)}")
# print(f"Images to analyze: {len(test_data_4)}")
# print(f"Images to analyze: {len(test_data_5)}")
# print(f"Images to analyze: {len(test_data_6)}")
# print(f"Images to analyze: {len(test_data_7)}")
# print(f"Images to analyze: {len(test_data_8)}")
# print(f"Images to analyze: {len(test_data_9)}")
print(f"Images to analyze: {len(test_data_10)}")
print(f"Images to analyze: {len(test_data_11)}")
print(f"Images to analyze: {len(test_data_12)}")

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

print("Setting up SHAP analysis...")

# Continue with SHAP analysis for each dataset and model combination...
# Run SHAP analysis for test_data_10 vs models_10p_blip2 and save to sub_output_folder10
# Run SHAP analysis for test_data_11 vs models_10p_blip2 and save to sub_output_folder11  
# Run SHAP analysis for test_data_12 vs models_10p_blip2 and save to sub_output_folder12

print("SHAP analysis completed!") 