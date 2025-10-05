import sqlite3
import json
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# We'll create jobs without embeddings first, then add them through the API
sample_jobs = [
    {
        "title": "Frontend Developer",
        "company": "TechStart Inc",
        "description": "We're looking for a passionate Frontend Developer to join our growing team. You'll work on building responsive web applications using modern JavaScript frameworks.",
        "requirements": [
            "3+ years experience with React or Vue.js",
            "Proficiency in HTML, CSS, JavaScript",
            "Experience with responsive design",
            "Knowledge of REST APIs",
            "Bachelor's degree preferred"
        ]
    },
    {
        "title": "Python Backend Developer",
        "company": "DataFlow Solutions",
        "description": "Join our backend team to build scalable APIs and data processing systems. You'll work with Python, databases, and cloud technologies.",
        "requirements": [
            "4+ years Python development experience",
            "Experience with Django or FastAPI",
            "Knowledge of PostgreSQL or MongoDB",
            "Understanding of API design principles",
            "Experience with AWS or Azure"
        ]
    },
    {
        "title": "Data Scientist",
        "company": "AI Innovations Lab",
        "description": "Work on cutting-edge machine learning projects to solve real-world problems. You'll analyze data, build ML models, and deploy solutions.",
        "requirements": [
            "Masters in Computer Science, Statistics, or related field",
            "3+ years experience with Python, R, or Julia",
            "Strong background in machine learning algorithms",
            "Experience with TensorFlow or PyTorch",
            "Knowledge of SQL and data visualization"
        ]
    },
    {
        "title": "Full Stack Developer",
        "company": "WebCraft Studios",
        "description": "Build end-to-end web applications using modern technologies. You'll work on both frontend and backend development.",
        "requirements": [
            "5+ years full-stack development experience",
            "Proficiency in React and Node.js",
            "Experience with databases (PostgreSQL, MongoDB)",
            "Knowledge of DevOps practices",
            "Strong problem-solving skills"
        ]
    },
    {
        "title": "DevOps Engineer",
        "company": "CloudScale Technologies",
        "description": "Manage and automate our cloud infrastructure. You'll work with containers, CI/CD pipelines, and monitoring systems.",
        "requirements": [
            "3+ years DevOps experience",
            "Expertise in AWS, Azure, or GCP",
            "Experience with Docker and Kubernetes",
            "Knowledge of CI/CD tools (Jenkins, GitLab CI)",
            "Scripting skills (Python, Bash)"
        ]
    },
    {
        "title": "UI/UX Designer",
        "company": "Design Forward Agency",
        "description": "Create beautiful and intuitive user interfaces for web and mobile applications. You'll work closely with developers and product managers.",
        "requirements": [
            "4+ years UI/UX design experience",
            "Proficiency in Figma, Sketch, or Adobe XD",
            "Understanding of user research methods",
            "Knowledge of design systems",
            "Portfolio showcasing web/mobile designs"
        ]
    },
    {
        "title": "Mobile App Developer",
        "company": "MobileFirst Solutions",
        "description": "Develop native iOS and Android applications. You'll work on consumer-facing apps with millions of users.",
        "requirements": [
            "3+ years mobile development experience",
            "Proficiency in Swift (iOS) or Kotlin (Android)",
            "Experience with React Native or Flutter",
            "Knowledge of mobile UI/UX principles",
            "Published apps in App Store or Google Play"
        ]
    },
    {
        "title": "Product Manager",
        "company": "InnovateTech Corp",
        "description": "Lead product strategy and development for our SaaS platform. You'll work with engineering, design, and business teams.",
        "requirements": [
            "5+ years product management experience",
            "MBA or relevant technical degree",
            "Experience with Agile methodologies",
            "Strong analytical and communication skills",
            "Background in SaaS or B2B products"
        ]
    },
    {
        "title": "Cybersecurity Analyst",
        "company": "SecureNet Systems",
        "description": "Protect our organization from cyber threats. You'll monitor security events, investigate incidents, and implement security measures.",
        "requirements": [
            "3+ years cybersecurity experience",
            "Knowledge of security frameworks (NIST, ISO 27001)",
            "Experience with SIEM tools",
            "Understanding of network security",
            "Security certifications (CISSP, CEH) preferred"
        ]
    },
    {
        "title": "Quality Assurance Engineer",
        "company": "TestPro Solutions",
        "description": "Ensure software quality through comprehensive testing. You'll design test cases, automate tests, and work with development teams.",
        "requirements": [
            "3+ years QA testing experience",
            "Experience with test automation tools (Selenium, Cypress)",
            "Knowledge of API testing",
            "Understanding of CI/CD processes",
            "Strong attention to detail"
        ]
    }
]

def create_sample_jobs():
    try:
        conn = sqlite3.connect('resumerag.db')
        cursor = conn.cursor()
        
        # Clear existing jobs
        cursor.execute('DELETE FROM jobs')
        
        for job in sample_jobs:
            cursor.execute('''
                INSERT INTO jobs (title, description, requirements, company, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                job['title'],
                job['description'],
                json.dumps(job['requirements']),
                job['company'],
                datetime.utcnow()
            ))
        
        conn.commit()
        print(f"Created {len(sample_jobs)} sample jobs successfully!")
        
        # List created jobs
        cursor.execute('SELECT title, company FROM jobs')
        jobs = cursor.fetchall()
        print("\nCreated jobs:")
        for title, company in jobs:
            print(f"- {title} at {company}")
            
        conn.close()
        
    except Exception as e:
        print(f"Error creating sample jobs: {e}")

if __name__ == "__main__":
    create_sample_jobs()
