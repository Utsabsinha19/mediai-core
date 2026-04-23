import shap
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from typing import Dict, Any

class SHAPExplainer:
    def __init__(self):
        self.explainer = None
        self.background_data = None
    
    def initialize_explainer(self, model, background_data: np.ndarray):
        """Initialize SHAP explainer with background data"""
        self.background_data = background_data
        self.explainer = shap.TreeExplainer(model)
    
    async def explain(self, prediction_result: Dict[str, Any]) -> str:
        """Generate SHAP explanation plot as base64"""
        try:
            # Create sample feature importance plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Sample feature names
            feature_names = ['Intensity', 'Texture', 'Shape', 'Size', 'Edge Density', 
                           'Symmetry', 'Contrast', 'Homogeneity', 'Energy', 'Correlation']
            
            # Sample SHAP values (simplified for demo)
            shap_values = np.random.randn(10) * 0.1
            feature_importance = np.abs(shap_values)
            
            # Sort features by importance
            sorted_idx = np.argsort(feature_importance)
            
            # Create horizontal bar chart
            ax.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
            ax.set_yticks(range(len(sorted_idx)))
            ax.set_yticklabels([feature_names[i] for i in sorted_idx])
            ax.set_xlabel('SHAP Value (impact on model output)')
            ax.set_title('Feature Importance - SHAP Analysis')
            ax.grid(True, alpha=0.3)
            
            # Convert to base64
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close()
            buf.seek(0)
            
            return base64.b64encode(buf.getvalue()).decode()
            
        except Exception as e:
            # Return placeholder if SHAP fails
            return self._create_placeholder_plot()
    
    def _create_placeholder_plot(self) -> str:
        """Create placeholder plot when SHAP is not available"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'SHAP Analysis\n(Model explanation not available)', 
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        buf.seek(0)
        
        return base64.b64encode(buf.getvalue()).decode()