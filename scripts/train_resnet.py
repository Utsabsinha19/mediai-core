#!/usr/bin/env python3
"""
Train ResNet50 for Brain Tumor Classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import joblib
from datetime import datetime

class ResNetTrainer:
    """Complete ResNet training pipeline"""
    
    def __init__(self, num_classes, device='cuda', model_name='resnet50'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.model_name = model_name
        self.model = self._create_model()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.best_val_acc = 0
        self.best_model_state = None
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_confidence': []
        }
    
    def _create_model(self):
        """Create ResNet model with custom classifier"""
        if self.model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            
            # Freeze early layers
            for param in model.parameters():
                param.requires_grad = False
            
            # Unfreeze last few layers
            for param in model.layer4.parameters():
                param.requires_grad = True
            for param in model.fc.parameters():
                param.requires_grad = True
            
            # Custom classifier
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.1),
                nn.Linear(256, self.num_classes)
            )
        elif self.model_name == 'resnet101':
            model = models.resnet101(pretrained=True)
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, self.num_classes)
            )
        else:
            model = models.resnet34(pretrained=True)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_classes)
        
        return model.to(self.device)
    
    def train_epoch(self, train_loader, optimizer):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validating"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                val_loss += loss.item()
                probs = torch.nn.functional.softmax(output, dim=1)
                pred = output.argmax(dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        val_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        val_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # Calculate confidence scores
        confidences = [np.max(prob) for prob in all_probs]
        avg_confidence = np.mean(confidences)
        
        metrics = {
            'loss': val_loss,
            'accuracy': val_acc,
            'precision': val_precision,
            'recall': val_recall,
            'f1_score': val_f1,
            'confidence': avg_confidence
        }
        
        return metrics, all_preds, all_labels
    
    def train(self, train_loader, val_loader, epochs=50, lr=0.001):
        """Full training pipeline"""
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        reduce_scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        print(f"\n🚀 Training {self.model_name.upper()} on {self.device}")
        print("="*60)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer)
            
            # Validate
            val_metrics, _, _ = self.validate(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1_score'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['val_confidence'].append(val_metrics['confidence'])
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"Val F1: {val_metrics['f1_score']:.4f}, Val Confidence: {val_metrics['confidence']:.4f}")
            
            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.best_model_state = self.model.state_dict().copy()
                print(f"✨ New best model! Accuracy: {self.best_val_acc:.4f}")
            
            # Reduce LR on plateau
            reduce_scheduler.step(val_metrics['loss'])
            if scheduler:
                scheduler.step()
        
        # Load best model
        self.model.load_state_dict(self.best_model_state)
        
        return self.history
    
    def save_model(self, path):
        """Save trained model"""
        torch.save({
            'model_state_dict': self.best_model_state,
            'num_classes': self.num_classes,
            'best_val_acc': self.best_val_acc,
            'model_name': self.model_name,
            'history': self.history
        }, path)
        print(f"\n✅ Model saved to {path}")
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss plot
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        axes[0, 0].set_xlabel('Epochs')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0, 1].plot(epochs, self.history['train_acc'], 'b-', label='Train Acc')
        axes[0, 1].plot(epochs, self.history['val_acc'], 'r-', label='Val Acc')
        axes[0, 1].set_xlabel('Epochs')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 Score plot
        axes[1, 0].plot(epochs, self.history['val_f1'], 'g-', label='Val F1')
        axes[1, 0].set_xlabel('Epochs')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('Validation F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Confidence plot
        axes[1, 1].plot(epochs, self.history['val_confidence'], 'purple', label='Val Confidence')
        axes[1, 1].set_xlabel('Epochs')
        axes[1, 1].set_ylabel('Confidence')
        axes[1, 1].set_title('Validation Confidence Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Training plot saved to {save_path}")
        
        plt.show()