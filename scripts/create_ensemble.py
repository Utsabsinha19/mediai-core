#!/usr/bin/env python3
"""
Create Weighted Ensemble of All Models
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.optimize import minimize
import joblib
import json

class EnsembleModel:
    """Weighted ensemble of all trained models"""
    
    def __init__(self, models):
        """
        models: dict with keys 'resnet', 'xgboost', 'random_forest', 'lightgbm'
        """
        self.models = models
        self.weights = None
        self.best_weights = None
        self.model_names = list(models.keys())
    
    def optimize_weights(self, X_val, y_val):
        """Optimize ensemble weights using validation data"""
        
        def negative_log_loss(weights):
            weights = np.abs(weights)
            weights = weights / weights.sum()
            
            ensemble_probs = np.zeros((len(y_val), len(np.unique(y_val))))
            
            for i, model_name in enumerate(self.model_names):
                model = self.models[model_name]
                
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X_val)
                else:
                    # For deep learning models, handle differently
                    probs = self._get_deep_learning_probs(model, X_val)
                
                ensemble_probs += weights[i] * probs
            
            # Negative log loss
            pred_probs = ensemble_probs[np.arange(len(y_val)), y_val]
            log_loss = -np.mean(np.log(pred_probs + 1e-10))
            
            return log_loss
        
        # Optimize weights
        initial_weights = np.ones(len(self.models)) / len(self.models)
        result = minimize(negative_log_loss, initial_weights,
                         method='L-BFGS-B',
                         bounds=[(0, 1) for _ in range(len(self.models))])
        
        self.weights = np.abs(result.x)
        self.weights = self.weights / self.weights.sum()
        
        print(f"\n🎯 Optimized Ensemble Weights:")
        for name, weight in zip(self.model_names, self.weights):
            print(f"  {name}: {weight:.4f}")
        
        return self.weights
    
    def _get_deep_learning_probs(self, model, X_val):
        """Get probabilities from deep learning models"""
        # This would be implemented for ResNet
        # For now, return placeholder
        return np.random.rand(len(X_val), 4)
    
    def predict_proba(self, X):
        """Get ensemble prediction probabilities"""
        if self.weights is None:
            weights = np.ones(len(self.models)) / len(self.models)
        else:
            weights = self.weights
        
        ensemble_probs = None
        
        for i, model_name in enumerate(self.model_names):
            model = self.models[model_name]
            
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)
            else:
                probs = self._get_deep_learning_probs(model, X)
            
            if ensemble_probs is None:
                ensemble_probs = weights[i] * probs
            else:
                ensemble_probs += weights[i] * probs
        
        return ensemble_probs
    
    def predict(self, X):
        """Get ensemble predictions"""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    
    def evaluate(self, X_test, y_test, class_names=None):
        """Evaluate ensemble performance"""
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        confidences = np.max(y_pred_proba, axis=1)
        avg_confidence = np.mean(confidences)
        
        # Calculate weighted score
        weighted_score = 0.7 * accuracy + 0.3 * avg_confidence
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confidence': avg_confidence,
            'weighted_score': weighted_score
        }
        
        print(f"\n📊 Ensemble Model Evaluation Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Confidence: {avg_confidence:.4f}")
        print(f"  Weighted Score: {weighted_score:.4f}")
        
        return metrics, y_pred, y_pred_proba
    
    def save_model(self, path):
        """Save ensemble model"""
        joblib.dump({
            'weights': self.weights,
            'model_names': self.model_names,
            'models': self.models
        }, path)
        print(f"\n✅ Ensemble model saved to {path}")