#!/usr/bin/env python3
"""
Download Brain Tumor Dataset from Hugging Face
"""

import os
import shutil
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import cv2
import numpy as np

def download_brain_tumor_dataset(output_dir="data/raw"):
    """Download brain tumor dataset from Hugging Face"""
    
    print("="*60)
    print("Downloading Brain Tumor Dataset from Hugging Face")
    print("="*60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load dataset
        print("\n📥 Loading dataset from Hugging Face...")
        dataset = load_dataset("sartajbhuvaji/Brain-Tumor-Classification", trust_remote_code=True)
        
        print(f"✅ Dataset loaded successfully!")
        print(f"Available splits: {list(dataset.keys())}")
        
        # Class mapping
        class_map = {
            0: "glioma",
            1: "meningioma", 
            2: "pituitary",
            3: "normal"
        }
        
        # Save images to disk
        for split_name, split_data in dataset.items():
            print(f"\n📁 Processing {split_name} split...")
            split_path = output_path / split_name
            
            for class_id, class_name in class_map.items():
                (split_path / class_name).mkdir(parents=True, exist_ok=True)
            
            for idx, item in enumerate(tqdm(split_data, desc=f"Saving {split_name}")):
                image = item['image']
                label = item['label']
                class_name = class_map[label]
                
                # Save image
                save_path = split_path / class_name / f"{split_name}_{idx}.jpg"
                if hasattr(image, 'save'):
                    image.save(save_path)
                else:
                    cv2.imwrite(str(save_path), cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        
        print(f"\n✅ Dataset saved to {output_path}")
        
        # Print statistics
        print("\n📊 Dataset Statistics:")
        for split_name in dataset.keys():
            split_path = output_path / split_name
            total = sum(len(list((split_path / cls).glob("*.jpg"))) for cls in class_map.values())
            print(f"  {split_name}: {total} images")
            
            for class_name in class_map.values():
                count = len(list((split_path / class_name).glob("*.jpg")))
                print(f"    - {class_name}: {count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\n📌 Alternative: Manual download from Kaggle")
        print("   https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")
        return False

if __name__ == "__main__":
    download_brain_tumor_dataset()