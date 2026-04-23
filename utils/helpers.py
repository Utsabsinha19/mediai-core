import re
import uuid
from datetime import datetime
from typing import Dict, Any

def generate_report_id() -> str:
    """Generate unique report ID"""
    return f"RPT-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"

def format_confidence(confidence: float) -> str:
    """Format confidence score as percentage"""
    return f"{confidence:.1%}"

def get_confidence_level(confidence: float) -> str:
    """Get confidence level text"""
    if confidence >= 0.9:
        return "Very High"
    elif confidence >= 0.7:
        return "High"
    elif confidence >= 0.5:
        return "Medium"
    else:
        return "Low"

def get_confidence_color(confidence: float) -> str:
    """Get color for confidence score"""
    if confidence >= 0.9:
        return "#27ae60"  # Green
    elif confidence >= 0.7:
        return "#3498db"  # Blue
    elif confidence >= 0.5:
        return "#f39c12"  # Orange
    else:
        return "#e74c3c"  # Red

def truncate_string(text: str, max_length: int = 50) -> str:
    """Truncate string to max length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."

def safe_json_parse(data: Any) -> Dict:
    """Safely parse JSON data"""
    try:
        if isinstance(data, str):
            import json
            return json.loads(data)
        return data
    except:
        return {}

def calculate_age(birth_date: datetime) -> int:
    """Calculate age from birth date"""
    today = datetime.now()
    return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))