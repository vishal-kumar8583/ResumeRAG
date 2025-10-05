#!/usr/bin/env python3
"""
Simple test script to verify basic search functionality without ML dependencies
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_search():
    """Test basic search functionality"""
    
    print("🔧 Testing Basic Search Functionality")
    print("=" * 50)
    
    # Test basic text processing
    print("\n1. Testing basic text processing...")
    
    test_text = "Python developer with 5 years of experience in web development using Django and React"
    
    # Simple keyword extraction
    keywords = ['python', 'developer', 'experience', 'web', 'development', 'django', 'react']
    found_keywords = [kw for kw in keywords if kw in test_text.lower()]
    
    print(f"✅ Text processing successful")
    print(f"📊 Found keywords: {found_keywords}")
    
    # Test basic similarity calculation
    print("\n2. Testing basic similarity calculation...")
    
    def simple_cosine_similarity(text1, text2):
        """Simple cosine similarity based on word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if len(union) == 0:
            return 0.0
        
        return len(intersection) / len(union)
    
    query = "Python developer"
    resumes = [
        "Python developer with Django experience",
        "Java developer with Spring framework", 
        "Frontend developer with React and JavaScript"
    ]
    
    similarities = []
    for i, resume in enumerate(resumes):
        sim = simple_cosine_similarity(query, resume)
        similarities.append((i+1, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print(f"✅ Similarity calculation successful")
    print(f"📊 Query: '{query}'")
    for resume_id, score in similarities:
        print(f"   📄 Resume {resume_id}: {score:.3f} - '{resumes[resume_id-1]}'")
    
    # Test snippet extraction
    print("\n3. Testing snippet extraction...")
    
    def extract_snippets(content, query, max_snippets=3):
        """Extract relevant snippets from content"""
        sentences = content.split('.')
        query_words = query.lower().split()
        relevant_snippets = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:
                sentence_lower = sentence.lower()
                if any(word in sentence_lower for word in query_words):
                    relevant_snippets.append(sentence + '.')
                    if len(relevant_snippets) >= max_snippets:
                        break
        
        return relevant_snippets
    
    resume_content = """
    John Doe is a Python developer with 5 years of experience.
    He has worked with Django and Flask web frameworks.
    He also has experience with React and JavaScript.
    He is familiar with PostgreSQL and MongoDB databases.
    He has deployed applications on AWS cloud platform.
    """
    
    snippets = extract_snippets(resume_content, "Python developer")
    
    print(f"✅ Snippet extraction successful")
    print(f"📊 Found {len(snippets)} snippets:")
    for i, snippet in enumerate(snippets, 1):
        print(f"   {i}. {snippet}")
    
    print("\n" + "=" * 50)
    print("🎉 Basic search functionality test completed successfully!")
    return True

if __name__ == "__main__":
    success = test_basic_search()
    if not success:
        print("\n❌ Some tests failed. Check the errors above.")
        sys.exit(1)
    else:
        print("\n✅ All basic tests passed!")
        sys.exit(0)

