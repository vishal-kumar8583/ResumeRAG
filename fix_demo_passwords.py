import sqlite3
import bcrypt

def update_demo_passwords():
    try:
        # Connect to SQLite database
        conn = sqlite3.connect('resumerag.db')
        cursor = conn.cursor()
        
        # Create properly hashed passwords with "password123"
        user_password = bcrypt.hashpw(b'password123', bcrypt.gensalt()).decode()
        recruiter_password = bcrypt.hashpw(b'password123', bcrypt.gensalt()).decode()
        
        # Update existing users
        cursor.execute('UPDATE users SET hashed_password = ? WHERE email = ?', 
                       (user_password, 'user@demo.com'))
        cursor.execute('UPDATE users SET hashed_password = ? WHERE email = ?', 
                       (recruiter_password, 'recruiter@demo.com'))
        
        conn.commit()
        print("Updated demo user passwords to 'password123'")
        print("You can now login with:")
        print("- User: user@demo.com / password123")
        print("- Recruiter: recruiter@demo.com / password123")
        
    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    update_demo_passwords()
