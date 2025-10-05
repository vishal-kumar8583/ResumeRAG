import sqlite3
import bcrypt
from datetime import datetime

def add_demo_users():
    try:
        # Connect to SQLite database
        conn = sqlite3.connect('resumerag.db')
        cursor = conn.cursor()
        
        # Create properly hashed passwords
        user_password = bcrypt.hashpw(b'demo123', bcrypt.gensalt()).decode()
        recruiter_password = bcrypt.hashpw(b'demo123', bcrypt.gensalt()).decode()
        
        # Add demo users
        users_to_add = [
            ('user@demo.com', user_password, 'user'),
            ('recruiter@demo.com', recruiter_password, 'recruiter')
        ]
        
        for email, hashed_password, role in users_to_add:
            try:
                # Check if user exists
                cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
                if cursor.fetchone():
                    print(f"User {email} already exists")
                    continue
                
                cursor.execute(
                    'INSERT INTO users (email, hashed_password, role) VALUES (?, ?, ?)',
                    (email, hashed_password, role)
                )
                print(f"Added {role}: {email}")
                
            except sqlite3.IntegrityError:
                print(f"User {email} already exists")
        
        conn.commit()
        print("\nDemo users created successfully!")
        print("You can now login with:")
        print("- User: user@demo.com / demo123")
        print("- Recruiter: recruiter@demo.com / demo123")
        
    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    add_demo_users()
