import os
import requests
from PIL import Image
from io import BytesIO
import pandas as pd
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from MSCOCO_preprocessing_local import prepare_data_from_preselected_categories

def download_image(url, save_path, max_retries=3):
    """Download a single image with retry logic"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Save the image
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait before retry
            else:
                print(f"Failed to download {url}: {e}")
                return False

def download_dataset_images(dataset_csv, data_type='train', max_workers=10):
    """
    Download all images for a dataset locally
    
    Args:
        dataset_csv: Path to your chosen categories CSV
        data_type: 'train' or 'test'
        max_workers: Number of parallel downloads
    """
    
    # Create local directories
    os.makedirs(f'coco_images/{data_type}', exist_ok=True)
    
    # Load the dataset to get image URLs
    print(f"Loading {data_type} dataset...")
    if data_type == 'train':
        train_data, val_data = prepare_data_from_preselected_categories(
            dataset_csv, 'train', split_val=True, 
            transform=None, target_transform=None, load_captions=False
        )
        # Combine train and val for downloading
        all_data = train_data.data + val_data.data
    else:
        test_data = prepare_data_from_preselected_categories(
            dataset_csv, 'test',
            transform=None, target_transform=None, load_captions=False
        )
        all_data = test_data.data
    
    print(f"Found {len(all_data)} images to download")
    
    # Prepare download tasks
    download_tasks = []
    for item in all_data:
        img_id = item['img_id']
        url = item['url']
        save_path = f'coco_images/{data_type}/{img_id}.jpg'
        
        # Skip if already downloaded
        if not os.path.exists(save_path):
            download_tasks.append((url, save_path))
    
    print(f"Need to download {len(download_tasks)} new images")
    
    # Download images in parallel
    successful_downloads = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_url = {
            executor.submit(download_image, url, path): (url, path) 
            for url, path in download_tasks
        }
        
        # Process completed downloads with progress bar
        with tqdm(total=len(download_tasks), desc=f"Downloading {data_type} images") as pbar:
            for future in as_completed(future_to_url):
                url, path = future_to_url[future]
                try:
                    success = future.result()
                    if success:
                        successful_downloads += 1
                except Exception as e:
                    print(f"Download failed: {e}")
                pbar.update(1)
    
    print(f"Successfully downloaded {successful_downloads}/{len(download_tasks)} images")
    return successful_downloads

if __name__ == "__main__":
    # Download images for both datasets
    datasets = [
        "chosen_categories_3_10.csv",
        "chosen_categories_6_20.csv"
    ]
    
    for dataset in datasets:
        if os.path.exists(dataset):
            print(f"\n{'='*60}")
            print(f"Downloading images for {dataset}")
            print(f"{'='*60}")
            
            # Download train images
            download_dataset_images(dataset, 'train', max_workers=10)
            
            # Download test images  
            download_dataset_images(dataset, 'test', max_workers=10)
        else:
            print(f"Dataset {dataset} not found, skipping...")
    
    print("\nAll downloads complete! Now your training will be much faster 🚀") 