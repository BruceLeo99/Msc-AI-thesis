import json
import time
from Food101_captioning import generate_caption, generate_caption_blip2


with open('train_full_annotation_7.json', 'r') as f:
    data = json.load(f)

all_data = []

for img_id, img_info in data.items():
    print(f"Processing image {img_id}: ", img_id)
    food_name = img_info['label']
    img_path = img_info['filepath']


    blip_caption = generate_caption(img_path, food_name)
    blip2_caption = generate_caption_blip2(img_path, food_name)

    all_data.append({
        img_id: {
            'img_id': img_id,
            'label': food_name,
            'filepath': img_path,
            'blip_caption': blip_caption,
            'blip2_caption': blip2_caption
        }
    })


with open('train_blip_caption_7.json', 'w') as f:
    json.dump(all_data, f)


with open('test_full_annotation_7.json', 'r') as f:
    data = json.load(f)

all_data = []

for img_id, img_info in data.items():
    print(f"Processing image {img_id}: ", img_id)
    food_name = img_info['label']
    img_path = img_info['filepath']


    blip_caption = generate_caption(img_path, food_name)
    blip2_caption = generate_caption_blip2(img_path, food_name)

    all_data.append({
        img_id: {
            'img_id': img_id,
            'label': food_name,
            'filepath': img_path,
            'blip_caption': blip_caption,
            'blip2_caption': blip2_caption
        }
    })


with open('test_blip_caption_7.json', 'w') as f:
    json.dump(all_data, f) 