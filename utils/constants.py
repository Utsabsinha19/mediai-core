# Disease classes and their descriptions
DISEASE_INFO = {
    "Glioma": {
        "description": "A type of tumor that occurs in the brain and spinal cord",
        "severity": "High",
        "urgency": "Immediate consultation recommended",
        "common_symptoms": ["Headaches", "Seizures", "Nausea", "Vision problems"]
    },
    "Meningioma": {
        "description": "A tumor that arises from the meninges, the membranes surrounding the brain",
        "severity": "Medium",
        "urgency": "Consult within 2 weeks",
        "common_symptoms": ["Headaches", "Weakness in limbs", "Seizures", "Personality changes"]
    },
    "Pituitary": {
        "description": "Tumor that develops in the pituitary gland",
        "severity": "Medium",
        "urgency": "Endocrinologist consultation recommended",
        "common_symptoms": ["Hormonal imbalances", "Vision problems", "Headaches", "Fatigue"]
    },
    "Normal": {
        "description": "No abnormalities detected",
        "severity": "Low",
        "urgency": "Routine check-up recommended",
        "common_symptoms": []
    }
}

# Model configurations
MODEL_CONFIGS = {
    "resnet50": {
        "input_size": 224,
        "num_classes": 4,
        "pretrained": True
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    }
}

# API response messages
API_MESSAGES = {
    "success": "Operation completed successfully",
    "unauthorized": "Authentication required",
    "forbidden": "You don't have permission to access this resource",
    "not_found": "Resource not found",
    "validation_error": "Invalid input data",
    "server_error": "Internal server error"
}

# File upload settings
UPLOAD_SETTINGS = {
    "max_file_size_mb": 10,
    "allowed_extensions": [".jpg", ".jpeg", ".png", ".dcm"],
    "image_quality": 95
}