#!/usr/bin/env python3
"""
Train LightGBM for Brain Tumor Classification
"""

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import optuna

class LightGBMTrainer:
    """Complete LightGBM training pipeline"""
    
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.model = None
        self.best_params = None
        self.feature_importance = None
    
    def hyperparameter_tuning(self, X_train, y_train, n_trials=50):
        """Hyperparameter tuning using Optuna"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            }
            
            if self.num_classes > 2:
                params['objective'] = 'multiclass'
                params['num_class'] = self.num_classes
                params['metric'] = 'multi_logloss'
            else:
                params['objective'] = 'binary'
                params['metric'] = 'binary_logloss'
            
            # Cross-validation
            cv_scores = []
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            for train_idx, val_idx in skf.split(X_train, y_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                
                model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(20, verbose=False)])
                
                preds = model.predict(X_val)
                score = accuracy_score(y_val, preds)
                cv_scores.append(score)
            
            return np.mean(cv_scores)
        
        print("\n🔍 Running hyperparameter optimization...")
        study = optuna.create_study(direction='maximize', study_name='lgbm_tuning')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = study.best_params
        print(f"\n✅ Best parameters: {self.best_params}")
        print(f"✅ Best CV score: {study.best_value:.4f}")
        
        return self.best_params
    
    def train(self, X_train, y_train, X_val, y_val, params=None):
        """Train LightGBM model"""
        
        if params is None:
            params = {
                'n_estimators': 500,
                'max_depth': 10,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'min_child_samples': 20,
                'random_state': 42
            }
        
        if self.num_classes > 2:
            params['objective'] = 'multiclass'
            params['num_class'] = self.num_classes
            params['metric'] = 'multi_logloss'
        else:
            params['objective'] = 'binary'
            params['metric'] = 'binary_logloss'
        
        print("\n🚀 Training LightGBM Model")
        print("="*60)
        
        self.model = lgb.LGBMClassifier(**params, verbose=-1)
        
        # Train with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=True)]
        )
        
        # Get feature importance
        self.feature_importance = self.model.feature_importances_
        
        print("\n✅ LightGBM training complete!")
        
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
        
        print(f"\n📊 LightGBM Evaluation Results:")
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
        plt.barh(range(len(top_features)), top_importance, color='orange')
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Feature Importance')
        plt.title('LightGBM - Top Feature Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def save_model(self, path):
        """Save trained model"""
        joblib.dump({
            'model': self.model,
            'best_params': self.best_params,
            'feature_importance': self.feature_importance,
            'num_classes': self.num_classes
        }, path)
        print(f"\n✅ LightGBM model saved to {path}")