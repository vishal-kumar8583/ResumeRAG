#!/usr/bin/env python3
"""
Debug script to check resume search functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.resume_service import ResumeParsingService
import json

def test_resume_service():
    """Test the resume service directly"""
    
    print("🔧 Testing Resume Service Directly")
    print("=" * 50)
    
    # Initialize service
    print("\n1. Initializing Resume Service...")
    try:
        service = ResumeParsingService()
        print("✅ Resume service initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize service: {e}")
        return False
    
    # Test embedding creation
    print("\n2. Testing embedding creation...")
    test_text = "Python developer with 5 years of experience in web development using Django and React"
    
    try:
        embedding = service.create_embedding(test_text)
        print(f"✅ Embedding created successfully")
        print(f"📊 Embedding dimension: {len(embedding)}")
        print(f"📝 Sample values: {embedding[:5]}")
    except Exception as e:
        print(f"❌ Embedding creation failed: {e}")
        return False
    
    # Test skill extraction
    print("\n3. Testing skill extraction...")
    resume_text = """
    John Doe
    Software Engineer
    
    Experience:
    - 5 years of Python development
    - Django and Flask web frameworks
    - React and JavaScript frontend
    - PostgreSQL and MongoDB databases
    - AWS cloud services
    - Docker containerization
    
    Skills:
    Python, JavaScript, React, Django, Flask, PostgreSQL, MongoDB, AWS, Docker, Git
    """
    
    try:
        skills = service.extract_skills(resume_text)
        print(f"✅ Skills extracted successfully")
        print(f"📊 Found {len(skills)} skills: {skills}")
    except Exception as e:
        print(f"❌ Skill extraction failed: {e}")
        return False
    
    # Test similarity search
    print("\n4. Testing similarity search...")
    
    try:
        # Create test embeddings
        query_embedding = service.create_embedding("Python developer")
        resume_embeddings = [
            (1, service.create_embedding("Python developer with Django experience")),
            (2, service.create_embedding("Java developer with Spring framework")),
            (3, service.create_embedding("Frontend developer with React and JavaScript"))
        ]
        
        results = service.similarity_search(query_embedding, resume_embeddings, k=2)
        print(f"✅ Similarity search successful")
        print(f"📊 Found {len(results)} results")
        for resume_id, score in results:
            print(f"   📄 Resume {resume_id}: {score:.3f}")
    except Exception as e:
        print(f"❌ Similarity search failed: {e}")
        return False
    
    # Test advanced semantic search
    print("\n5. Testing advanced semantic search...")
    
    try:
        resume_data = [
            {
                'id': 1,
                'content': "Python developer with Django experience and machine learning skills",
                'embedding': json.dumps(service.create_embedding("Python developer with Django experience and machine learning skills")),
                'filename': "python_dev_resume.pdf"
            },
            {
                'id': 2,
                'content': "Java developer with Spring framework and microservices experience",
                'embedding': json.dumps(service.create_embedding("Java developer with Spring framework and microservices experience")),
                'filename': "java_dev_resume.pdf"
            }
        ]
        
        results = service.advanced_semantic_search("Python developer", resume_data, k=2)
        print(f"✅ Advanced semantic search successful")
        print(f"📊 Found {len(results)} results")
        for result in results:
            print(f"   📄 Resume {result['resume_id']}: combined_score={result['combined_score']:.3f}")
    except Exception as e:
        print(f"❌ Advanced semantic search failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 Resume service test completed successfully!")
    return True

if __name__ == "__main__":
    success = test_resume_service()
    if not success:
        print("\n❌ Some tests failed. Check the errors above.")
        sys.exit(1)
    else:
        print("\n✅ All tests passed!")
        sys.exit(0)

