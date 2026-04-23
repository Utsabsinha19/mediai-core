import pytest
from typing import AsyncGenerator
from httpx import AsyncClient
from backend.main import app
from backend.core.database import MongoDB

@pytest.fixture
async def client() -> AsyncGenerator:
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture
async def test_db():
    """Setup test database"""
    # Connect to test database
    MongoDB.MONGODB_URL = "mongodb://localhost:27017"
    MongoDB.DATABASE_NAME = "healthcare_ai_test"
    await MongoDB.connect()
    
    yield MongoDB.database
    
    # Cleanup after tests
    await MongoDB.client.drop_database("healthcare_ai_test")
    await MongoDB.close()

@pytest.fixture
def sample_user():
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "Test123!@#",
        "full_name": "Test User"
    }

@pytest.fixture
def sample_image_bytes():
    """Create a sample image for testing"""
    from PIL import Image
    import io
    
    img = Image.new('RGB', (224, 224), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()