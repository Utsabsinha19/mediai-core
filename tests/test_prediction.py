import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_prediction_upload(client: AsyncClient, sample_user, sample_image_bytes):
    """Test image upload and prediction"""
    # Register and login
    await client.post("/api/v1/auth/register", json=sample_user)
    login_response = await client.post("/api/v1/auth/login", json={
        "username": sample_user["username"],
        "password": sample_user["password"]
    })
    token = login_response.json()["access_token"]
    
    # Upload image for prediction
    files = {"file": ("test_image.jpg", sample_image_bytes, "image/jpeg")}
    response = await client.post(
        "/api/v1/predict/upload",
        files=files,
        headers={"Authorization": f"Bearer {token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "report_id" in data
    assert "prediction" in data
    assert "confidence" in data

@pytest.mark.asyncio
async def test_get_prediction_history(client: AsyncClient, sample_user):
    """Test getting prediction history"""
    # Register and login
    await client.post("/api/v1/auth/register", json=sample_user)
    login_response = await client.post("/api/v1/auth/login", json={
        "username": sample_user["username"],
        "password": sample_user["password"]
    })
    token = login_response.json()["access_token"]
    user_id = login_response.json().get("user_id", "test_id")
    
    # Get history
    response = await client.get(
        f"/api/v1/predict/history/{user_id}",
        headers={"Authorization": f"Bearer {token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "predictions" in data
    assert isinstance(data["predictions"], list)

@pytest.mark.asyncio
async def test_get_report(client: AsyncClient, sample_user):
    """Test getting specific report"""
    # Register and login
    await client.post("/api/v1/auth/register", json=sample_user)
    login_response = await client.post("/api/v1/auth/login", json={
        "username": sample_user["username"],
        "password": sample_user["password"]
    })
    token = login_response.json()["access_token"]
    
    # Get a report (using a known report ID for testing)
    report_id = "60d5f4832f8fb814b56fa181"  # Replace with actual test report ID
    response = await client.get(
        f"/api/v1/predict/report/{report_id}",
        headers={"Authorization": f"Bearer {token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == report_id