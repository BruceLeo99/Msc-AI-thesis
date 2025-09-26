from DeepSHAP_explaner_PBN_batched import *

output_folder = f"/var/scratch/yyg760/SHAP_results/Case 1/pbn_neg_10p"
test_img_ids_filepath1 = "mpbn_neg_1p.txt"
test_img_ids_filepath2 = "mpbn_neg_2p.txt"
test_img_ids_filepath3 = "mpbn_neg_5p.txt"
test_img_ids_filepath4 = "mpbn_neg_10p.txt"
caption_type = "blip"

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

test_data_full = f101.Food101Dataset(test_data_json, transform='vgg16', target_transform='integer', load_captions=True)
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


print("Loading background images...")
# Load background dataset from val dataset (since background_images_202.txt contains val IDs)
background_dataset = load_background_from_image_ids(
    image_ids=background_img_ids,
    dataset_type='val',  # Change to 'test' if your background IDs are from test set
    transform='vgg16',
    target_transform='integer',
    load_captions=True
)

# Convert to DataLoader and then to tensor format
background_loader = DataLoader(background_dataset, batch_size=20, shuffle=False, collate_fn=collate_fn)

# Collect images from DataLoader
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
background_images = torch.cat(background_images, dim=0)
print(f"Type of background_images after concatenation: {type(background_images)}")
print(f"Background images shape: {background_images.shape}")

# Keep background images on CPU initially to save GPU memory
background = background_images

print(f"Background Images Loaded.Background tensor shape: {background.shape}")

models_1p = {
    'PBN1_VGG16': model_PBN1,
}

models_2p = {
    'PBN2_VGG16': model_PBN2,
}

models_5p = {
    'PBN5_VGG16': model_PBN5,
}

models_10p = {
    'PBN10_VGG16': model_PBN10,
}

os.makedirs(output_folder, exist_ok=True)

# Configure batch size for background processing (adjust based on GPU memory)
background_batch_size = 20  # Start conservative, can increase if memory allows

# Run comprehensive SHAP analysis with batched processing
print("Star SHAP analysis...")
# output_folder, results_summary = run(models_1p, device, background, test_data_to_analyze1, output_folder, background_batch_size) 
# print(f"Analysis completed for image IDs: {img_ids_to_analyze1}") 

# output_folder, results_summary = run(models_2p, device, background, test_data_to_analyze2, output_folder, background_batch_size) 
# print(f"Analysis completed for image IDs: {img_ids_to_analyze2}") 

# output_folder, results_summary = run(models_5p, device, background, test_data_to_analyze3, output_folder, background_batch_size) 
# print(f"Analysis completed for image IDs: {img_ids_to_analyze3}") 

output_folder, results_summary = run(models_10p, device, background, test_data_to_analyze4, output_folder, background_batch_size) 
print(f"Analysis completed for image IDs: {img_ids_to_analyze4}") 