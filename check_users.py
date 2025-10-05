import sqlite3

try:
    conn = sqlite3.connect('resumerag.db')
    cursor = conn.cursor()
    cursor.execute('SELECT email, role FROM users')
    users = cursor.fetchall()
    
    print('Users in database:')
    for email, role in users:
        print(f'- {email} ({role})')
    
    if not users:
        print("No users found in database!")
        
    conn.close()
except Exception as e:
    print(f"Error: {e}")
