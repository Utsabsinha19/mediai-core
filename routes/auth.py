from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime, timedelta
from typing import Optional

from backend.core.database import MongoDB
from backend.core.security import (
    get_password_hash, verify_password, create_access_token, 
    create_refresh_token, decode_token
)
from backend.models.user import UserCreate, UserLogin, UserResponse
from backend.schemas.auth import Token, RefreshTokenRequest, ChangePasswordRequest
from backend.api.dependencies import get_current_user

router = APIRouter(prefix="/auth", tags=["Authentication"])
security = HTTPBearer()

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate):
    """Register a new user"""
    # Check if user exists
    existing_user = await MongoDB.get_user_by_username(user_data.username)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    existing_email = await MongoDB.get_user_by_email(user_data.email)
    if existing_email:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user
    user_dict = user_data.dict()
    user_dict["hashed_password"] = get_password_hash(user_dict.pop("password"))
    user_dict["created_at"] = datetime.utcnow()
    user_dict["updated_at"] = datetime.utcnow()
    
    collection = MongoDB.get_collection("users")
    result = await collection.insert_one(user_dict)
    
    # Return user response
    user_dict["id"] = str(result.inserted_id)
    user_dict.pop("hashed_password")
    
    return UserResponse(**user_dict)

@router.post("/login", response_model=Token)
async def login(user_data: UserLogin):
    """Login user and return tokens"""
    # Find user
    user = await MongoDB.get_user_by_username(user_data.username)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Verify password
    if not verify_password(user_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Update last login
    await MongoDB.get_collection("users").update_one(
        {"_id": user["_id"]},
        {"$set": {"last_login": datetime.utcnow()}}
    )
    
    # Create tokens
    token_data = {
        "sub": str(user["_id"]),
        "username": user["username"],
        "role": user["role"]
    }
    
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)
    
    return Token(access_token=access_token, refresh_token=refresh_token)

@router.post("/refresh", response_model=Token)
async def refresh_token(request: RefreshTokenRequest):
    """Refresh access token using refresh token"""
    try:
        payload = decode_token(request.refresh_token)
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type")
        
        # Create new tokens
        token_data = {
            "sub": payload["sub"],
            "username": payload["username"],
            "role": payload["role"]
        }
        
        access_token = create_access_token(token_data)
        refresh_token = create_refresh_token(token_data)
        
        return Token(access_token=access_token, refresh_token=refresh_token)
        
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current user information"""
    return UserResponse(
        id=current_user["id"],
        username=current_user["username"],
        email=current_user["email"],
        full_name=current_user.get("full_name"),
        role=current_user["role"],
        is_active=current_user["is_active"],
        created_at=current_user["created_at"].isoformat()
    )

@router.post("/change-password")
async def change_password(
    request: ChangePasswordRequest,
    current_user: dict = Depends(get_current_user)
):
    """Change user password"""
    # Verify old password
    if not verify_password(request.old_password, current_user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Incorrect old password")
    
    # Update password
    new_hashed = get_password_hash(request.new_password)
    await MongoDB.get_collection("users").update_one(
        {"_id": current_user["_id"]},
        {"$set": {"hashed_password": new_hashed, "updated_at": datetime.utcnow()}}
    )
    
    return {"message": "Password changed successfully"}

@router.post("/logout")
async def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Logout user (client should discard tokens)"""
    # In a real implementation, you might blacklist the token
    return {"message": "Logged out successfully"}