import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_register_user(client: AsyncClient, sample_user):
    """Test user registration"""
    response = await client.post("/api/v1/auth/register", json=sample_user)
    assert response.status_code == 201
    data = response.json()
    assert data["username"] == sample_user["username"]
    assert data["email"] == sample_user["email"]

@pytest.mark.asyncio
async def test_login_user(client: AsyncClient, sample_user):
    """Test user login"""
    # First register
    await client.post("/api/v1/auth/register", json=sample_user)
    
    # Then login
    login_data = {
        "username": sample_user["username"],
        "password": sample_user["password"]
    }
    response = await client.post("/api/v1/auth/login", json=login_data)
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data

@pytest.mark.asyncio
async def test_get_current_user(client: AsyncClient, sample_user):
    """Test getting current user info"""
    # Register and login
    await client.post("/api/v1/auth/register", json=sample_user)
    login_response = await client.post("/api/v1/auth/login", json={
        "username": sample_user["username"],
        "password": sample_user["password"]
    })
    token = login_response.json()["access_token"]
    
    # Get current user
    response = await client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == sample_user["username"]