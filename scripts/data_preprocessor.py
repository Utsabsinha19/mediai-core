#!/usr/bin/env python3
"""
Data Preprocessing for Medical Images
"""

import os
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import joblib

class MedicalImageDataset(Dataset):
    """Custom Dataset for Medical Images"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, torch.tensor(label, dtype=torch.long)

class DataPreprocessor:
    """Complete data preprocessing pipeline"""
    
    def __init__(self, data_dir="data/raw", processed_dir="data/processed"):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.label_encoder = LabelEncoder()
        
        # Training augmentations
        self.train_transform = A.Compose([
            A.Resize(224, 224),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Validation transforms (no augmentation)
        self.val_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def load_data(self):
        """Load images from directory structure"""
        images = []
        labels = []
        class_names = []
        
        # Check if using Hugging Face structure
        for split in ['train', 'val', 'test']:
            split_path = self.data_dir / split
            if split_path.exists():
                for class_dir in split_path.iterdir():
                    if class_dir.is_dir():
                        if class_dir.name not in class_names:
                            class_names.append(class_dir.name)
                        for img_path in class_dir.glob("*.jpg"):
                            images.append(str(img_path))
                            labels.append(class_dir.name)
        
        # If no Hugging Face structure, try standard structure
        if not images:
            for class_dir in self.data_dir.iterdir():
                if class_dir.is_dir():
                    class_names.append(class_dir.name)
                    for img_path in class_dir.glob("*.jpg"):
                        images.append(str(img_path))
                        labels.append(class_dir.name)
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        print(f"\n📊 Data Loading Complete:")
        print(f"  Total images: {len(images)}")
        print(f"  Classes: {class_names}")
        print(f"  Class distribution:")
        for class_name in class_names:
            count = labels.count(class_name)
            print(f"    - {class_name}: {count}")
        
        return images, encoded_labels, class_names
    
    def split_data(self, images, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Split data into train/val/test sets"""
        # First split: train+val vs test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            images, labels, test_size=test_ratio, stratify=labels, random_state=42
        )
        
        # Second split: train vs val
        val_size = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, stratify=y_train_val, random_state=42
        )
        
        print(f"\n📊 Data Split Complete:")
        print(f"  Training: {len(X_train)} images")
        print(f"  Validation: {len(X_val)} images")
        print(f"  Test: {len(X_test)} images")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def create_dataloaders(self, X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
        """Create PyTorch DataLoaders"""
        train_dataset = MedicalImageDataset(X_train, y_train, self.train_transform)
        val_dataset = MedicalImageDataset(X_val, y_val, self.val_transform)
        test_dataset = MedicalImageDataset(X_test, y_test, self.val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        return train_loader, val_loader, test_loader
    
    def extract_deep_features(self, dataloader, model, device='cuda'):
        """Extract features using deep learning model for ML algorithms"""
        import torch.nn as nn
        
        model = model.to(device)
        model.eval()
        
        # Remove classifier to get features
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        feature_extractor = feature_extractor.to(device)
        
        features_list = []
        labels_list = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Extracting features"):
                images = images.to(device)
                features = feature_extractor(images)
                features = features.view(features.size(0), -1)
                features_list.append(features.cpu().numpy())
                labels_list.append(labels.numpy())
        
        features = np.concatenate(features_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        
        return features, labels
    
    def save_preprocessed_data(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Save preprocessed data to disk"""
        data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'class_names': self.label_encoder.classes_.tolist()
        }
        
        joblib.dump(data, self.processed_dir / 'preprocessed_data.pkl')
        joblib.dump(self.label_encoder, self.processed_dir / 'label_encoder.pkl')
        
        print(f"\n✅ Preprocessed data saved to {self.processed_dir}")

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    images, labels, class_names = preprocessor.load_data()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocessor.split_data(images, labels)
    preprocessor.save_preprocessed_data(X_train, y_train, X_val, y_val, X_test, y_test)