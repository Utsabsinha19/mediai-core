#!/usr/bin/env python3
"""
Master Training Script - Train All Models and Compare Results
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import joblib

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.data_preprocessor import DataPreprocessor
from scripts.train_resnet import ResNetTrainer
from scripts.train_xgboost import XGBoostTrainer
from scripts.train_randomforest import RandomForestTrainer
from scripts.train_lightgbm import LightGBMTrainer
from scripts.create_ensemble import EnsembleModel

class MasterTrainer:
    """Master training pipeline for all models"""
    
    def __init__(self, data_dir="data/raw", output_dir="../backend/ml/weights"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.preprocessor = DataPreprocessor(data_dir)
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.X_train_ml = None
        self.X_val_ml = None
        self.X_test_ml = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.class_names = None
        
        self.resnet_model = None
        self.xgb_model = None
        self.rf_model = None
        self.lgbm_model = None
        self.ensemble = None
        
        self.results = {}
    
    def prepare_data(self):
        """Prepare data for all models"""
        print("\n" + "="*60)
        print("STEP 1: DATA PREPARATION")
        print("="*60)
        
        # Load and preprocess data
        images, labels, class_names = self.preprocessor.load_data()
        self.class_names = class_names
        
        # Split data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.preprocessor.split_data(
            images, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )
        
        # Create dataloaders for deep learning
        self.train_loader, self.val_loader, self.test_loader = self.preprocessor.create_dataloaders(
            X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32
        )
        
        # Extract features for ML models using ResNet
        print("\n📊 Extracting features for ML models...")
        
        # First train ResNet to use as feature extractor
        temp_trainer = ResNetTrainer(num_classes=len(class_names))
        self.X_train_ml, self.y_train = temp_trainer.extract_features(self.train_loader)
        self.X_val_ml, self.y_val = temp_trainer.extract_features(self.val_loader)
        self.X_test_ml, self.y_test = temp_trainer.extract_features(self.test_loader)
        
        print(f"\n✅ Data preparation complete!")
        print(f"  Training features: {self.X_train_ml.shape}")
        print(f"  Validation features: {self.X_val_ml.shape}")
        print(f"  Test features: {self.X_test_ml.shape}")
    
    def train_resnet(self):
        """Train ResNet model"""
        print("\n" + "="*60)
        print("STEP 2: TRAINING RESNET-50")
        print("="*60)
        
        trainer = ResNetTrainer(num_classes=len(self.class_names))
        history = trainer.train(self.train_loader, self.val_loader, epochs=30)
        
        # Evaluate
        metrics, _, _ = trainer.validate(self.test_loader)
        self.results['resnet50'] = metrics
        
        # Save model
        trainer.save_model(f"{self.output_dir}/resnet50_best.pth")
        self.resnet_model = trainer
        
        # Plot training history
        trainer.plot_training_history(save_path=f"{self.output_dir}/resnet_training_history.png")
        
        return trainer
    
    def train_xgboost(self):
        """Train XGBoost model"""
        print("\n" + "="*60)
        print("STEP 3: TRAINING XGBOOST")
        print("="*60)
        
        trainer = XGBoostTrainer(num_classes=len(self.class_names))
        
        # Optional: Hyperparameter tuning
        # trainer.hyperparameter_tuning(self.X_train_ml, self.y_train, n_trials=30)
        
        trainer.train(self.X_train_ml, self.y_train, self.X_val_ml, self.y_val)
        metrics, y_pred, _ = trainer.evaluate(self.X_test_ml, self.y_test)
        self.results['xgboost'] = metrics
        
        # Save model
        trainer.save_model(f"{self.output_dir}/xgboost_model.pkl")
        self.xgb_model = trainer
        
        # Plot feature importance
        trainer.plot_feature_importance(save_path=f"{self.output_dir}/xgboost_feature_importance.png")
        trainer.plot_confusion_matrix(self.y_test, y_pred, self.class_names, 
                                      save_path=f"{self.output_dir}/xgboost_confusion_matrix.png")
        
        return trainer
    
    def train_randomforest(self):
        """Train Random Forest model"""
        print("\n" + "="*60)
        print("STEP 4: TRAINING RANDOM FOREST")
        print("="*60)
        
        trainer = RandomForestTrainer(num_classes=len(self.class_names))
        
        # Optional: Hyperparameter tuning
        # trainer.hyperparameter_tuning(self.X_train_ml, self.y_train, n_trials=30)
        
        trainer.train(self.X_train_ml, self.y_train)
        metrics, y_pred, _ = trainer.evaluate(self.X_test_ml, self.y_test)
        self.results['random_forest'] = metrics
        
        # Save model
        trainer.save_model(f"{self.output_dir}/random_forest.pkl")
        self.rf_model = trainer
        
        # Plot feature importance
        trainer.plot_feature_importance(save_path=f"{self.output_dir}/rf_feature_importance.png")
        trainer.plot_confusion_matrix(self.y_test, y_pred, self.class_names,
                                      save_path=f"{self.output_dir}/rf_confusion_matrix.png")
        
        return trainer
    
    def train_lightgbm(self):
        """Train LightGBM model"""
        print("\n" + "="*60)
        print("STEP 5: TRAINING LIGHTGBM")
        print("="*60)
        
        trainer = LightGBMTrainer(num_classes=len(self.class_names))
        
        # Optional: Hyperparameter tuning
        # trainer.hyperparameter_tuning(self.X_train_ml, self.y_train, n_trials=30)
        
        trainer.train(self.X_train_ml, self.y_train, self.X_val_ml, self.y_val)
        metrics, y_pred, _ = trainer.evaluate(self.X_test_ml, self.y_test)
        self.results['lightgbm'] = metrics
        
        # Save model
        trainer.save_model(f"{self.output_dir}/lightgbm_model.pkl")
        self.lgbm_model = trainer
        
        # Plot feature importance
        trainer.plot_feature_importance(save_path=f"{self.output_dir}/lgbm_feature_importance.png")
        
        return trainer
    
    def create_ensemble(self):
        """Create ensemble of all models"""
        print("\n" + "="*60)
        print("STEP 6: CREATING ENSEMBLE MODEL")
        print("="*60)
        
        # Prepare models for ensemble
        models_for_ensemble = {
            'xgboost': self.xgb_model.model,
            'random_forest': self.rf_model.model,
            'lightgbm': self.lgbm_model.model
        }
        
        # Create ensemble
        ensemble = EnsembleModel(models_for_ensemble)
        
        # Optimize weights
        ensemble.optimize_weights(self.X_val_ml, self.y_val)
        
        # Evaluate ensemble
        metrics, _, _ = ensemble.evaluate(self.X_test_ml, self.y_test, self.class_names)
        self.results['ensemble'] = metrics
        
        # Save ensemble
        ensemble.save_model(f"{self.output_dir}/ensemble_model.pkl")
        self.ensemble = ensemble
        
        return ensemble
    
    def compare_models(self):
        """Compare all trained models"""
        print("\n" + "="*60)
        print("STEP 7: MODEL COMPARISON")
        print("="*60)
        
        # Create comparison dataframe
        comparison_data = []
        
        for model_name, metrics in self.results.items():
            weighted_score = 0.7 * metrics['accuracy'] + 0.3 * metrics['confidence']
            
            comparison_data.append({
                'Model': model_name.upper(),
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1 Score': f"{metrics['f1_score']:.4f}",
                'Confidence': f"{metrics['confidence']:.4f}",
                'Weighted Score': f"{weighted_score:.4f}"
            })
            
            print(f"\n{model_name.upper()}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1 Score: {metrics['f1_score']:.4f}")
            print(f"  Confidence: {metrics['confidence']:.4f}")
            print(f"  Weighted Score: {weighted_score:.4f}")
        
        # Create DataFrame
        df = pd.DataFrame(comparison_data)
        df.to_csv(f"{self.output_dir}/model_comparison.csv", index=False)
        
        # Find best model
        best_model = max(self.results.items(), 
                        key=lambda x: 0.7 * x[1]['accuracy'] + 0.3 * x[1]['confidence'])
        
        print("\n" + "="*60)
        print(f"🏆 BEST MODEL: {best_model[0].upper()}")
        print(f"🏆 Best Accuracy: {best_model[1]['accuracy']:.4f}")
        print(f"🏆 Best F1 Score: {best_model[1]['f1_score']:.4f}")
        print(f"🏆 Weighted Score: {0.7 * best_model[1]['accuracy'] + 0.3 * best_model[1]['confidence']:.4f}")
        print("="*60)
        
        return df, best_model
    
    def plot_comparison(self):
        """Plot model comparison charts"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        models = list(self.results.keys())
        accuracies = [self.results[m]['accuracy'] for m in models]
        f1_scores = [self.results[m]['f1_score'] for m in models]
        confidences = [self.results[m]['confidence'] for m in models]
        
        x = np.arange(len(models))
        width = 0.25
        
        # Bar chart comparison
        axes[0, 0].bar(x - width, accuracies, width, label='Accuracy', color='#667eea')
        axes[0, 0].bar(x, f1_scores, width, label='F1 Score', color='#764ba2')
        axes[0, 0].bar(x + width, confidences, width, label='Confidence', color='#27ae60')
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Scores')
        axes[0, 0].set_title('Model Performance Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(models, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Weighted scores
        weighted_scores = [0.7 * self.results[m]['accuracy'] + 0.3 * self.results[m]['confidence'] 
                          for m in models]
        colors = ['#27ae60' if i == np.argmax(weighted_scores) else '#667eea' 
                 for i in range(len(models))]
        axes[0, 1].bar(models, weighted_scores, color=colors)
        axes[0, 1].set_xlabel('Models')
        axes[0, 1].set_ylabel('Weighted Score')
        axes[0, 1].set_title('Model Weighted Scores (70% Accuracy + 30% Confidence)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Metrics heatmap
        metrics_data = np.array([accuracies, f1_scores, confidences])
        im = axes[1, 0].imshow(metrics_data, cmap='YlOrRd', aspect='auto')
        axes[1, 0].set_xticks(range(len(models)))
        axes[1, 0].set_xticklabels(models, rotation=45)
        axes[1, 0].set_yticks([0, 1, 2])
        axes[1, 0].set_yticklabels(['Accuracy', 'F1 Score', 'Confidence'])
        axes[1, 0].set_title('Performance Metrics Heatmap')
        plt.colorbar(im, ax=axes[1, 0])
        
        # Radar chart for best model
        best_idx = np.argmax(weighted_scores)
        best_metrics = [accuracies[best_idx], f1_scores[best_idx], confidences[best_idx]]
        
        angles = np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist()
        best_metrics += best_metrics[:1]
        angles += angles[:1]
        
        ax = plt.subplot(2, 2, 4, projection='polar')
        ax.plot(angles, best_metrics, 'o-', linewidth=2, color='#27ae60')
        ax.fill(angles, best_metrics, alpha=0.25, color='#27ae60')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(['Accuracy', 'F1 Score', 'Confidence'])
        ax.set_ylim(0, 1)
        ax.set_title(f'Best Model: {models[best_idx].upper()}')
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/model_comparison_charts.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\n✅ Comparison charts saved to {self.output_dir}/model_comparison_charts.png")
    
    def generate_report(self):
        """Generate comprehensive training report"""
        print("\n" + "="*60)
        print("STEP 8: GENERATING TRAINING REPORT")
        print("="*60)
        
        # Find best model
        best_model = max(self.results.items(), 
                        key=lambda x: 0.7 * x[1]['accuracy'] + 0.3 * x[1]['confidence'])
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'num_classes': len(self.class_names),
                'classes': self.class_names,
                'train_samples': len(self.y_train),
                'val_samples': len(self.y_val),
                'test_samples': len(self.y_test)
            },
            'model_results': self.results,
            'best_model': {
                'name': best_model[0],
                'accuracy': best_model[1]['accuracy'],
                'f1_score': best_model[1]['f1_score'],
                'confidence': best_model[1]['confidence'],
                'weighted_score': 0.7 * best_model[1]['accuracy'] + 0.3 * best_model[1]['confidence']
            }
        }
        
        # Save JSON report
        with open(f"{self.output_dir}/training_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate HTML report
        self._generate_html_report(report)
        
        print(f"\n✅ Reports saved to {self.output_dir}")
        print(f"  - training_report.json")
        print(f"  - training_report.html")
        print(f"  - model_comparison.csv")
        print(f"  - model_comparison_charts.png")
    
    def _generate_html_report(self, report):
        """Generate HTML report"""
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Training Report - Healthcare AI</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        h1 {{
            color: #667eea;
            text-align: center;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .model-card {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            transition: transform 0.3s;
        }}
        .model-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }}
        .model-card.best {{
            background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
            border: 2px solid #667eea;
        }}
        .model-name {{
            font-size: 1.3em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 10px;
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #e0e0e0;
        }}
        .best-badge {{
            display: inline-block;
            background: #27ae60;
            color: white;
            padding: 3px 8px;
            border-radius: 20px;
            font-size: 0.8em;
            margin-top: 10px;
        }}
        .summary-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            margin: 20px 0;
        }}
        .summary-score {{
            font-size: 2.5em;
            font-weight: bold;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #667eea;
            color: white;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🏥 Healthcare AI Platform</h1>
        <h2 style="text-align:center">Model Training Report</h2>
        <p style="text-align:center">Generated: {report['timestamp']}</p>
        
        <div class="summary-box">
            <h3>🏆 Best Performing Model</h3>
            <div class="summary-score">{report['best_model']['name'].upper()}</div>
            <p>Accuracy: {report['best_model']['accuracy']:.2%} | F1 Score: {report['best_model']['f1_score']:.2%}</p>
            <p>Weighted Score: {report['best_model']['weighted_score']:.4f}</p>
        </div>
        
        <h3>📊 Model Performance Comparison</h3>
        <div class="metrics-grid">
"""
        
        for model_name, metrics in report['model_results'].items():
            is_best = model_name == report['best_model']['name']
            best_class = 'best' if is_best else ''
            weighted_score = 0.7 * metrics['accuracy'] + 0.3 * metrics['confidence']
            
            html_content += f"""
            <div class="model-card {best_class}">
                <div class="model-name">{model_name.upper()}</div>
                <div class="metric">
                    <span>🎯 Accuracy:</span>
                    <span>{metrics['accuracy']:.2%}</span>
                </div>
                <div class="metric">
                    <span>📊 Precision:</span>
                    <span>{metrics['precision']:.2%}</span>
                </div>
                <div class="metric">
                    <span>🔄 Recall:</span>
                    <span>{metrics['recall']:.2%}</span>
                </div>
                <div class="metric">
                    <span>🎯 F1 Score:</span>
                    <span>{metrics['f1_score']:.2%}</span>
                </div>
                <div class="metric">
                    <span>💪 Confidence:</span>
                    <span>{metrics['confidence']:.2%}</span>
                </div>
                <div class="metric">
                    <span>⚖️ Weighted Score:</span>
                    <span>{weighted_score:.4f}</span>
                </div>
"""
            if is_best:
                html_content += '<div class="best-badge">🏆 BEST MODEL</div>'
            html_content += """
            </div>
"""
        
        html_content += """
        </div>
        
        <h3>📈 Detailed Metrics Table</h3>
        <table>
            <thead>
                <tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1 Score</th><th>Confidence</th></tr>
            </thead>
            <tbody>
"""
        
        for model_name, metrics in report['model_results'].items():
            html_content += f"""
                <tr>
                    <td><strong>{model_name.upper()}</strong></td>
                    <td>{metrics['accuracy']:.2%}</td>
                    <td>{metrics['precision']:.2%}</td>
                    <td>{metrics['recall']:.2%}</td>
                    <td>{metrics['f1_score']:.2%}</td>
                    <td>{metrics['confidence']:.2%}</td>
                </tr>
"""
        
        html_content += f"""
            </tbody>
        </table>
        
        <h3>📋 Dataset Information</h3>
        <ul>
            <li>Number of Classes: {report['dataset_info']['num_classes']}</li>
            <li>Classes: {', '.join(report['dataset_info']['classes'])}</li>
            <li>Training Samples: {report['dataset_info']['train_samples']}</li>
            <li>Validation Samples: {report['dataset_info']['val_samples']}</li>
            <li>Test Samples: {report['dataset_info']['test_samples']}</li>
        </ul>
        
        <div style="background: #e8f5e9; padding: 15px; border-radius: 10px; margin-top: 20px;">
            <h4>👨‍⚕️ Clinical Recommendation</h4>
            <p>Based on the AI model analysis, we recommend:</p>
            <ul>
                <li>Use the <strong>{report['best_model']['name'].upper()}</strong> model for production deployment</li>
                <li>Model achieves {report['best_model']['accuracy']:.2%} accuracy with {report['best_model']['confidence']:.2%} confidence</li>
                <li>Ensemble model recommended for critical cases</li>
                <li>Always validate predictions with clinical experts</li>
            </ul>
        </div>
        
        <p style="text-align: center; margin-top: 30px; color: #666;">
            Generated by Healthcare AI Platform | Multi-Model Training Pipeline
        </p>
    </div>
</body>
</html>
"""
        
        with open(f"{self.output_dir}/training_report.html", 'w') as f:
            f.write(html_content)
    
    def run(self):
        """Run complete training pipeline"""
        print("\n" + "🚀"*30)
        print("COMPLETE MODEL TRAINING PIPELINE")
        print("🚀"*30)
        
        # Step 1: Prepare data
        self.prepare_data()
        
        # Step 2: Train ResNet
        self.train_resnet()
        
        # Step 3: Train XGBoost
        self.train_xgboost()
        
        # Step 4: Train Random Forest
        self.train_randomforest()
        
        # Step 5: Train LightGBM
        self.train_lightgbm()
        
        # Step 6: Create Ensemble
        self.create_ensemble()
        
        # Step 7: Compare models
        self.compare_models()
        
        # Step 8: Plot comparison
        self.plot_comparison()
        
        # Step 9: Generate report
        self.generate_report()
        
        print("\n" + "✅"*30)
        print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("✅"*30)
        print(f"\n📁 All models saved to: {self.output_dir}")
        print(f"📊 Reports saved to: {self.output_dir}")
        
        return self.results

if __name__ == "__main__":
    trainer = MasterTrainer()
    results = trainer.run()
    print(results)
    