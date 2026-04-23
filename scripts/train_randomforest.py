#!/usr/bin/env python3
"""
Train Random Forest for Brain Tumor Classification
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm import tqdm
import optuna

class RandomForestTrainer:
    """Complete Random Forest training pipeline"""
    
    def __init__(self, num_classes, n_jobs=-1):
        self.num_classes = num_classes
        self.n_jobs = n_jobs
        self.model = None
        self.best_params = None
        self.feature_importance = None
        self.history = {}
    
    def hyperparameter_tuning(self, X_train, y_train, n_trials=50):
        """Hyperparameter tuning using Optuna"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None])
            }
            
            # Cross-validation
            cv_scores = []
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            for train_idx, val_idx in skf.split(X_train, y_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                
                model = RandomForestClassifier(**params, random_state=42, n_jobs=self.n_jobs)
                model.fit(X_tr, y_tr)
                
                preds = model.predict(X_val)
                score = accuracy_score(y_val, preds)
                cv_scores.append(score)
            
            return np.mean(cv_scores)
        
        print("\n🔍 Running hyperparameter optimization...")
        study = optuna.create_study(direction='maximize', study_name='rf_tuning')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = study.best_params
        print(f"\n✅ Best parameters: {self.best_params}")
        print(f"✅ Best CV score: {study.best_value:.4f}")
        
        return self.best_params
    
    def train(self, X_train, y_train, params=None):
        """Train Random Forest model"""
        
        if params is None:
            params = {
                'n_estimators': 500,
                'max_depth': 20,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'bootstrap': True,
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': self.n_jobs
            }
        
        print("\n🚀 Training Random Forest Model")
        print("="*60)
        
        self.model = RandomForestClassifier(**params)
        self.model.fit(X_train, y_train)
        
        # Get feature importance
        self.feature_importance = self.model.feature_importances_
        
        print("\n✅ Random Forest training complete!")
        
        return self.model
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Calculate confidence scores
        confidences = np.max(y_pred_proba, axis=1)
        avg_confidence = np.mean(confidences)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confidence': avg_confidence
        }
        
        print(f"\n📊 Random Forest Evaluation Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Confidence: {avg_confidence:.4f}")
        
        return metrics, y_pred, y_pred_proba
    
    def plot_feature_importance(self, feature_names=None, top_k=20, save_path=None):
        """Plot feature importance"""
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(self.feature_importance))]
        
        # Get top k features
        indices = np.argsort(self.feature_importance)[-top_k:]
        top_features = [feature_names[i] for i in indices]
        top_importance = self.feature_importance[indices]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_importance, color='green')
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Feature Importance')
        plt.title('Random Forest - Top Feature Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, y_test, y_pred, class_names, save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Random Forest - Confusion Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def save_model(self, path):
        """Save trained model"""
        joblib.dump({
            'model': self.model,
            'best_params': self.best_params,
            'feature_importance': self.feature_importance,
            'num_classes': self.num_classes
        }, path)
        print(f"\n✅ Random Forest model saved to {path}")