from typing import List, Dict, Any

def get_medical_recommendations(disease: str, confidence: float) -> Dict[str, Any]:
    """Get medical recommendations based on disease and confidence"""
    
    recommendations = {
        "Glioma": {
            "immediate_actions": [
                "Schedule emergency consultation with a neurosurgeon",
                "Bring all imaging reports to appointment",
                "Do not delay treatment",
                "Seek second opinion if needed"
            ],
            "tests": [
                "MRI with contrast for detailed imaging",
                "Biopsy for definitive diagnosis",
                "Neurological examination",
                "Genetic testing for tumor markers"
            ],
            "treatments": [
                "Surgical resection",
                "Radiation therapy",
                "Chemotherapy",
                "Targeted therapy based on molecular profile"
            ],
            "follow_up": "Every 3-6 months for first 2 years, then annually"
        },
        "Meningioma": {
            "immediate_actions": [
                "Consult with neurologist within 2 weeks",
                "Monitor for neurological symptoms",
                "Document any changes in symptoms"
            ],
            "tests": [
                "MRI with contrast",
                "CT scan for calcification",
                "Vision field test if tumor near optic nerve",
                "Hormone level testing"
            ],
            "treatments": [
                "Active surveillance for asymptomatic cases",
                "Surgical removal for symptomatic tumors",
                "Radiation therapy for residual or recurrent tumors",
                "Stereotactic radiosurgery for small tumors"
            ],
            "follow_up": "Every 6-12 months with imaging"
        },
        "Pituitary": {
            "immediate_actions": [
                "Endocrinologist consultation",
                "Monitor hormone levels",
                "Track vision changes"
            ],
            "tests": [
                "Hormone panel (prolactin, GH, ACTH, TSH, FSH, LH)",
                "MRI with contrast",
                "Vision field testing",
                "Glucose tolerance test for GH-secreting tumors"
            ],
            "treatments": [
                "Medication for hormone-secreting tumors",
                "Transsphenoidal surgery for larger tumors",
                "Radiation therapy for residual tumors",
                "Hormone replacement therapy"
            ],
            "follow_up": "Every 3-6 months for hormone levels, annual MRI"
        },
        "Normal": {
            "immediate_actions": [
                "No immediate action needed",
                "Continue regular health check-ups",
                "Maintain healthy lifestyle"
            ],
            "tests": [
                "Routine annual physical examination",
                "Regular blood pressure monitoring",
                "Cholesterol and blood sugar screening"
            ],
            "treatments": [
                "Preventive healthcare",
                "Healthy diet and exercise",
                "Stress management",
                "Adequate sleep"
            ],
            "follow_up": "Annual check-up or as recommended by physician"
        }
    }
    
    base_rec = recommendations.get(disease, recommendations["Normal"])
    
    # Add confidence-based notes
    if confidence < 0.7:
        base_rec["note"] = "Low confidence prediction. Additional imaging or second opinion recommended."
    elif confidence < 0.85:
        base_rec["note"] = "Moderate confidence. Clinical correlation advised."
    else:
        base_rec["note"] = "High confidence prediction. Strongly recommend following the above guidance."
    
    return base_rec

def get_lifestyle_recommendations(disease: str) -> List[str]:
    """Get lifestyle recommendations"""
    lifestyle = {
        "Glioma": [
            "Avoid smoking and alcohol",
            "Maintain a healthy diet rich in antioxidants",
            "Regular exercise as tolerated",
            "Stress management techniques",
            "Adequate sleep (7-8 hours)",
            "Join support groups for emotional support"
        ],
        "Meningioma": [
            "Regular monitoring of symptoms",
            "Avoid head trauma",
            "Maintain healthy blood pressure",
            "Limit radiation exposure",
            "Regular eye examinations"
        ],
        "Pituitary": [
            "Monitor for hormonal changes",
            "Maintain consistent medication schedule",
            "Track energy levels and mood changes",
            "Regular exercise",
            "Balanced diet for hormone regulation"
        ],
        "Normal": [
            "Exercise 30 minutes daily",
            "Eat a balanced diet with fruits and vegetables",
            "Stay hydrated (8 glasses of water daily)",
            "Get 7-8 hours of sleep",
            "Practice stress reduction techniques",
            "Regular health screenings"
        ]
    }
    
    return lifestyle.get(disease, lifestyle["Normal"])