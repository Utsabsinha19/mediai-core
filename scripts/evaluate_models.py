#!/usr/bin/env python3
"""
Evaluate and Compare All Trained Models
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import json

class ModelEvaluator:
    """Evaluate and compare trained models"""
    
    def __init__(self, models_dir="../backend/ml/weights"):
        self.models_dir = models_dir
        self.models = {}
        self.results = {}
    
    def load_models(self):
        """Load all trained models"""
        print("📥 Loading trained models...")
        
        # Load XGBoost
        try:
            xgb_data = joblib.load(f"{self.models_dir}/xgboost_model.pkl")
            self.models['xgboost'] = xgb_data['model']
            print("✅ Loaded XGBoost model")
        except:
            print("⚠️ XGBoost model not found")
        
        # Load Random Forest
        try:
            rf_data = joblib.load(f"{self.models_dir}/random_forest.pkl")
            self.models['random_forest'] = rf_data['model']
            print("✅ Loaded Random Forest model")
        except:
            print("⚠️ Random Forest model not found")
        
        # Load LightGBM
        try:
            lgbm_data = joblib.load(f"{self.models_dir}/lightgbm_model.pkl")
            self.models['lightgbm'] = lgbm_data['model']
            print("✅ Loaded LightGBM model")
        except:
            print("⚠️ LightGBM model not found")
        
        # Load Ensemble
        try:
            ensemble_data = joblib.load(f"{self.models_dir}/ensemble_model.pkl")
            self.models['ensemble'] = ensemble_data
            print("✅ Loaded Ensemble model")
        except:
            print("⚠️ Ensemble model not found")
    
    def evaluate_all(self, X_test, y_test, class_names):
        """Evaluate all loaded models"""
        
        for model_name, model in self.models.items():
            print(f"\n📊 Evaluating {model_name.upper()}...")
            
            if model_name == 'ensemble':
                y_pred = model['models']['xgboost'].predict(X_test)
                y_pred_proba = model['models']['xgboost'].predict_proba(X_test)
            else:
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            confidence = np.mean(np.max(y_pred_proba, axis=1))
            
            self.results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confidence': confidence,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            print(f"  Confidence: {confidence:.4f}")
            
            # Print classification report
            print(f"\n  Classification Report:")
            print(classification_report(y_test, y_pred, target_names=class_names))
    
    def plot_roc_curves(self, X_test, y_test, class_names, save_path=None):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(12, 8))
        
        # Binarize labels
        y_test_bin = label_binarize(y_test, classes=range(len(class_names)))
        n_classes = len(class_names)
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for idx, (model_name, result) in enumerate(self.results.items()):
            y_pred_proba = result['y_pred_proba']
            
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=colors[idx], lw=2,
                        label=f'{model_name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n✅ ROC curves saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrices(self, class_names, save_path=None):
        """Plot confusion matrices for all models"""
        n_models = len(self.results)
        fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (model_name, result) in enumerate(self.results.items()):
            cm = confusion_matrix(self.results[list(self.results.keys())[0]]['y_pred'] if idx == 0 else result['y_pred'], 
                                  result['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names, ax=axes[idx])
            axes[idx].set_title(f'{model_name.upper()} - Confusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Confusion matrices saved to {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(self, save_path=None):
        """Generate evaluation report"""
        report_data = []
        
        for model_name, metrics in self.results.items():
            weighted_score = 0.7 * metrics['accuracy'] + 0.3 * metrics['confidence']
            report_data.append({
                'Model': model_name.upper(),
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1 Score': f"{metrics['f1_score']:.4f}",
                'Confidence': f"{metrics['confidence']:.4f}",
                'Weighted Score': f"{weighted_score:.4f}"
            })
        
        df = pd.DataFrame(report_data)
        
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"\n✅ Evaluation report saved to {save_path}")
        
        return df

if __name__ == "__main__":
    # This would be used with test data
    print("Evaluation script ready. Load test data to evaluate models.")
    