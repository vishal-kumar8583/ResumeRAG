import sqlite3
import bcrypt

def test_password_verification():
    try:
        # Get the hashed password from database
        conn = sqlite3.connect('resumerag.db')
        cursor = conn.cursor()
        cursor.execute('SELECT hashed_password FROM users WHERE email = ?', ('user@demo.com',))
        result = cursor.fetchone()
        
        if result:
            stored_hash = result[0]
            test_password = "demo123"
            
            print(f"Stored hash: {stored_hash[:50]}...")
            print(f"Test password: {test_password}")
            
            # Test with bcrypt directly
            is_valid = bcrypt.checkpw(test_password.encode(), stored_hash.encode())
            print(f"Password verification: {is_valid}")
            
            # Test with passlib (what the main app uses)
            try:
                from passlib.context import CryptContext
                pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
                is_valid_passlib = pwd_context.verify(test_password, stored_hash)
                print(f"Passlib verification: {is_valid_passlib}")
            except Exception as e:
                print(f"Passlib error: {e}")
                
        else:
            print("User not found in database")
            
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_password_verification()
