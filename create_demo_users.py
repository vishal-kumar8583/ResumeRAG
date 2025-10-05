#!/usr/bin/env python3
"""
Script to create demo users for ResumeRAG application
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import SessionLocal, User, get_password_hash
from sqlalchemy.orm import Session

def create_demo_users():
    """Create demo users for testing"""
    db = SessionLocal()
    
    try:
        # Check if users already exist
        existing_user = db.query(User).filter(User.email == "user@demo.com").first()
        existing_recruiter = db.query(User).filter(User.email == "recruiter@demo.com").first()
        
        if existing_user:
            print("Demo user already exists")
        else:
            # Create regular user with shorter password
            user = User(
                email="user@demo.com",
                hashed_password=get_password_hash("pass123"),
                role="user"
            )
            db.add(user)
            print("Created demo user: user@demo.com")
        
        if existing_recruiter:
            print("Demo recruiter already exists")
        else:
            # Create recruiter with shorter password
            recruiter = User(
                email="recruiter@demo.com",
                hashed_password=get_password_hash("pass123"),
                role="recruiter"
            )
            db.add(recruiter)
            print("Created demo recruiter: recruiter@demo.com")
        
        db.commit()
        print("\nDemo users created successfully!")
        print("You can now login with:")
        print("- User: user@demo.com / pass123")
        print("- Recruiter: recruiter@demo.com / pass123")
        
    except Exception as e:
        db.rollback()
        print(f"Error creating demo users: {e}")
        
    finally:
        db.close()

if __name__ == "__main__":
    create_demo_users()
