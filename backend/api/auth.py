from fastapi import APIRouter, Depends, HTTPException, status, Response, Cookie
from sqlalchemy.orm import Session
from typing import Optional
from database.connection import get_db
from database_models.db import User, Setting
from schemas.api import UserRegister, UserLogin, Token, UserProfile
from core.security import get_password_hash, verify_password, create_access_token, decode_access_token

router = APIRouter(prefix="/api", tags=["authentication"])

def get_current_user(token: str = None, db: Session = Depends(get_db)):
    if not token:
        # Check in cookies if not in headers
        raise HTTPException(
            status_code=status.HTTP_418_IM_A_TEAPOT,
            detail="Guest session"
        )
    username = decode_access_token(token)
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    return user

# Optional user dependency for Guest mode support
def get_optional_user(token: Optional[str] = None, db: Session = Depends(get_db)):
    if not token:
        return None
    username = decode_access_token(token)
    if not username:
        return None
    return db.query(User).filter(User.username == username).first()

@router.post("/register", response_model=Token)
def register(user_data: UserRegister, db: Session = Depends(get_db)):
    # Check if username or email exists
    db_user = db.query(User).filter(User.username == user_data.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    db_email = db.query(User).filter(User.email == user_data.email).first()
    if db_email:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_pwd = get_password_hash(user_data.password)
    new_user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_pwd
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # Create default settings for user
    default_settings = Setting(user_id=new_user.id)
    db.add(default_settings)
    db.commit()
    
    access_token = create_access_token(subject=new_user.username)
    return {"access_token": access_token, "token_type": "bearer", "username": new_user.username}

@router.post("/login", response_model=Token)
def login(credentials: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == credentials.username).first()
    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(subject=user.username)
    return {"access_token": access_token, "token_type": "bearer", "username": user.username}

@router.post("/logout")
def logout():
    return {"message": "Successfully logged out"}

@router.get("/me", response_model=UserProfile)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user
