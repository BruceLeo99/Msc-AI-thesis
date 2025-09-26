import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DeepSHAP_explaner_mPBN_batched import *

main_output_folder = f"/var/scratch/yyg760/SHAP_results/Case 1/mpbn_pos_10p_blip"
os.makedirs(main_output_folder, exist_ok=True)

test_img_ids_filepath1 = "mpbn_pos_10p_blip.txt"

caption_type = "blip"

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
img_ids_to_analyze1 = get_list_of_ids(test_img_ids_filepath1)


# Load test data for analysis
with open("Food101/food-101/meta/test_full_annotation_customize.json", "r") as f:
    test_data_json = json.load(f)
test_data_json = {k: v for d in test_data_json for k, v in d.items()}

img_to_analyze1 = {k: v for k, v in test_data_json.items() if k in img_ids_to_analyze1}

test_data_full_blip = f101.Food101Dataset(test_data_json, transform='vgg16', target_transform='integer', load_captions=True, caption_type=caption_type)

print(f"Full dataset size: {len(test_data_full_blip)}")


print("Loading models...")

model_path_mPBN10_blip = torch.load(f"/var/scratch/yyg760/results_final/mPBN_10p_{caption_type}/mPBN_10p_{caption_type}_best.pth", map_location=device)

model_mPBN10_blip_config = model_path_mPBN10_blip['model_config']

model_mPBN10_blip_state_dict = model_path_mPBN10_blip['state_dict']


model_mPBN10_blip_new_state_dict = OrderedDict()
for k, v in model_mPBN10_blip_state_dict.items():
    if k.startswith('module.'):
        name = k[7:] 
    else:
        name = k
    model_mPBN10_blip_new_state_dict[name] = v

num_classes = test_data_full_blip.get_num_classes()
print(f"Number of classes: {num_classes}")

model_mPBN10_blip = VisualBertPPNet(num_prototypes_per_class=10, num_classes=num_classes)

print("Loading model weights...")

model_mPBN10_blip.load_state_dict(model_mPBN10_blip_new_state_dict)

model_mPBN10_blip = torch.nn.DataParallel(model_mPBN10_blip)

models_10p_blip = {
    'mPBN_10p_blip': model_mPBN10_blip,
}



print("Model weights loaded successfully!")

print("Loading background images...")
# Load background dataset from val dataset (since background_images_202.txt contains val IDs)
background_dataset_blip = load_background_from_image_ids(
    image_ids=background_img_ids,
    dataset_type='val',  # Change to 'test' if your background IDs are from test set
    transform='vgg16',
    target_transform='integer',
    load_captions=True,
    caption_type=caption_type
)

# For mPBN, we need the full batch dictionaries (with captions), not just tensors
print("Collecting background data with captions...")
background_data_blip = []
for i in range(len(background_dataset_blip)):
    sample = background_dataset_blip[i]  # This returns the full batch dict with captions
    
    # Move all tensor components to device
    sample_on_device = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            sample_on_device[key] = value.to(device)
        else:
            sample_on_device[key] = value
    
    background_data_blip.append(sample_on_device)

print(f"Background data collected: {len(background_data_blip)} samples with captions")
if len(background_data_blip) > 0:
    print(f"First background sample keys: {list(background_data_blip[0].keys())}")
    print(f"Visual embeds shape: {background_data_blip[0]['visual_embeds'].shape}")
    print(f"Input IDs shape: {background_data_blip[0]['input_ids'].shape}")

# Set background as the list of batch dictionaries
background_blip = background_data_blip

print(f"Background Images Loaded. Number of samples: {len(background_blip)}")

# Configure batch size for background processing (adjust based on GPU memory)
background_batch_size = 15  # Start conservative, can increase if memory allows

# Run comprehensive SHAP analysis with batched processing

print("Star SHAP analysis...")
main_output_folder, results_summary = run(models_10p_blip, device, background_blip, img_to_analyze1, main_output_folder, background_batch_size) 
print(f"Analysis completed for image IDs: {img_ids_to_analyze1}") 

print("SHAP analysis completed!") 