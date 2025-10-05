#!/usr/bin/env python3
"""
Test script to verify search functionality
"""

import requests
import json

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_EMAIL = "test@example.com"
TEST_PASSWORD = "testpassword123"

def test_search_functionality():
    """Test the search functionality"""
    
    print("🔍 Testing ResumeRAG Search Functionality")
    print("=" * 50)
    
    # Step 1: Register/Login
    print("\n1. Authenticating...")
    try:
        # Try to login first
        login_data = {
            "email": TEST_EMAIL,
            "password": TEST_PASSWORD
        }
        
        response = requests.post(f"{BASE_URL}/api/auth/login", json=login_data)
        
        if response.status_code == 200:
            token_data = response.json()
            token = token_data["access_token"]
            print("✅ Login successful")
        else:
            # Try to register
            register_data = {
                "email": TEST_EMAIL,
                "password": TEST_PASSWORD,
                "role": "user"
            }
            
            response = requests.post(f"{BASE_URL}/api/auth/register", json=register_data)
            
            if response.status_code == 200:
                token_data = response.json()
                token = token_data["access_token"]
                print("✅ Registration successful")
            else:
                print(f"❌ Authentication failed: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Authentication error: {e}")
        return False
    
    # Step 2: Test search with headers
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    print("\n2. Testing search functionality...")
    
    # Test queries
    test_queries = [
        "Python developer",
        "machine learning experience",
        "web development skills",
        "database management",
        "project management"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n   Test {i}: '{query}'")
        
        try:
            search_data = {
                "query": query,
                "k": 3
            }
            
            response = requests.post(
                f"{BASE_URL}/api/ask", 
                json=search_data, 
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Search successful")
                print(f"   📊 Found {len(result.get('answers', []))} results")
                print(f"   🔍 Search method: {result.get('search_method', 'unknown')}")
                print(f"   📝 Total resumes searched: {result.get('total_resumes_searched', 0)}")
                
                # Show first result if available
                if result.get('answers'):
                    first_answer = result['answers'][0]
                    print(f"   📄 Top match: {first_answer.get('filename', 'Unknown')}")
                    if 'combined_score' in first_answer:
                        print(f"   🎯 Combined score: {first_answer['combined_score']:.3f}")
                    elif 'similarity_score' in first_answer:
                        print(f"   🎯 Similarity score: {first_answer['similarity_score']:.3f}")
                    
                    # Show evidence snippets
                    snippets = first_answer.get('evidence_snippets', [])
                    if snippets:
                        print(f"   💡 Evidence: {snippets[0][:100]}...")
                
            else:
                print(f"   ❌ Search failed: {response.status_code}")
                print(f"   📝 Error: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Search error: {e}")
    
    # Step 3: Test advanced search endpoint
    print(f"\n3. Testing advanced search endpoint...")
    
    try:
        search_data = {
            "query": "Python developer with Django experience",
            "k": 5
        }
        
        response = requests.post(
            f"{BASE_URL}/api/search/advanced", 
            json=search_data, 
            headers=headers,
            params={"search_type": "semantic"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Advanced search successful")
            print(f"   📊 Found {len(result.get('results', []))} results")
            print(f"   🔍 Search type: {result.get('search_type', 'unknown')}")
        else:
            print(f"   ❌ Advanced search failed: {response.status_code}")
            print(f"   📝 Error: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Advanced search error: {e}")
    
    # Step 4: Test search suggestions
    print(f"\n4. Testing search suggestions...")
    
    try:
        response = requests.get(
            f"{BASE_URL}/api/search/suggestions", 
            headers=headers,
            params={"q": "python"}
        )
        
        if response.status_code == 200:
            result = response.json()
            suggestions = result.get('suggestions', [])
            print(f"   ✅ Suggestions successful")
            print(f"   💡 Found {len(suggestions)} suggestions")
            if suggestions:
                print(f"   📝 Sample suggestions: {suggestions[:3]}")
        else:
            print(f"   ❌ Suggestions failed: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Suggestions error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Search functionality test completed!")
    return True

if __name__ == "__main__":
    test_search_functionality()

