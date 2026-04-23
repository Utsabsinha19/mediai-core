import cv2
import numpy as np
from PIL import Image
import io
import os
import uuid
from typing import Tuple, Optional
import aiofiles

class ImageProcessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
    
    def validate_file(self, filename: str) -> bool:
        """Validate file extension"""
        allowed_extensions = ['.jpg', '.jpeg', '.png', '.dcm']
        ext = os.path.splitext(filename)[1].lower()
        return ext in allowed_extensions
    
    async def process_image(self, image_bytes: bytes, filename: str) -> np.ndarray:
        """Process uploaded image for model input"""
        # Load image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize
        image = image.resize(self.target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        image_array = np.array(image) / 255.0
        
        # Normalize
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image_array = (image_array - mean) / std
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array.astype(np.float32)
    
    async def save_uploaded_file(self, image_bytes: bytes, filename: str) -> str:
        """Save uploaded file to disk"""
        upload_dir = "backend/static/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        # Generate unique filename
        ext = os.path.splitext(filename)[1]
        unique_filename = f"{uuid.uuid4().hex}{ext}"
        filepath = os.path.join(upload_dir, unique_filename)
        
        # Save file
        async with aiofiles.open(filepath, 'wb') as f:
            await f.write(image_bytes)
        
        return filepath
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract simple features for ML models"""
        features = []
        
        # Color histograms
        for i in range(3):
            hist = np.histogram(image[0, :, :, i], bins=32, range=(0, 1))[0]
            features.extend(hist)
        
        # Texture features (simplified)
        gray = cv2.cvtColor((image[0] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        features.append(edge_density)
        
        return np.array(features).reshape(1, -1)