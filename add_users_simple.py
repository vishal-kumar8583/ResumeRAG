import sqlite3
import hashlib
from datetime import datetime

def create_simple_hash(password):
    """Create a simple hash - just for demo purposes"""
    return hashlib.sha256(password.encode()).hexdigest()

def add_demo_users():
    try:
        # Connect to SQLite database
        conn = sqlite3.connect('resumerag.db')
        cursor = conn.cursor()
        
        # Check if users table exists and create if needed
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                hashed_password TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Add demo users
        users_to_add = [
            ('user@demo.com', 'demo123', 'user'),
            ('recruiter@demo.com', 'demo123', 'recruiter')
        ]
        
        for email, password, role in users_to_add:
            try:
                # Check if user exists
                cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
                if cursor.fetchone():
                    print(f"User {email} already exists")
                    continue
                
                # Simple password hash for demo
                hashed_password = create_simple_hash(password)
                
                cursor.execute(
                    'INSERT INTO users (email, hashed_password, role) VALUES (?, ?, ?)',
                    (email, hashed_password, role)
                )
                print(f"Added {role}: {email}")
                
            except sqlite3.IntegrityError:
                print(f"User {email} already exists")
        
        conn.commit()
        print("\nDemo users created! You can login with:")
        print("- User: user@demo.com / demo123")
        print("- Recruiter: recruiter@demo.com / demo123")
        
    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        conn.close()

if __name__ == "__main__":
    add_demo_users()
