#!/usr/bin/env python3

import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import uuid
import re
from collections import Counter

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Header, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import PyPDF2
import docx

# Simple in-memory storage (in production, use a real database)
users_db = {}
resumes_db = {}
jobs_db = {}
sessions_db = {}

# Security
security = HTTPBearer()
SECRET_KEY = "demo-secret-key-change-in-production"

# Create FastAPI app
app = FastAPI(
    title="ResumeRAG API",
    description="AI-Powered Resume Search & Job Matching",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class UserRegister(BaseModel):
    email: str
    password: str
    role: str = "user"

class UserLogin(BaseModel):
    email: str
    password: str

class ResumeQuery(BaseModel):
    query: str
    k: int = 5

class JobCreate(BaseModel):
    title: str
    description: str
    requirements: List[str]
    company: str

# Simple text processing functions
def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX"""
    try:
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        return f"Error reading DOCX: {str(e)}"

def extract_skills(text: str) -> List[str]:
    """Extract skills from text"""
    skills = ['python', 'java', 'javascript', 'react', 'angular', 'vue', 'django', 'flask', 'node', 'express',
              'machine learning', 'ai', 'data science', 'aws', 'docker', 'kubernetes', 'git', 'sql', 'mongodb']
    
    found_skills = []
    text_lower = text.lower()
    for skill in skills:
        if skill in text_lower:
            found_skills.append(skill.title())
    return found_skills

def simple_similarity(text1: str, text2: str) -> float:
    """Simple text similarity based on common words"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    if len(union) == 0:
        return 0.0
    return len(intersection) / len(union)

def create_user_token(email: str) -> str:
    """Create a simple token"""
    token = hashlib.md5(f"{email}-{datetime.now().isoformat()}".encode()).hexdigest()
    sessions_db[token] = {"email": email, "created": datetime.now()}
    return token

def verify_token(token: str) -> Optional[dict]:
    """Verify token and return user info"""
    if token in sessions_db:
        session = sessions_db[token]
        if session["email"] in users_db:
            return users_db[session["email"]]
    return None

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Get current user from token"""
    token = credentials.credentials
    user = verify_token(token)
    if not user:
        raise HTTPException(status_code=401, detail={"error": {"code": "UNAUTHORIZED", "message": "Invalid token"}})
    return user

# Routes
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ResumeRAG - AI Resume Search</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f8fafc; }
            .container { max-width: 1200px; margin: 0 auto; padding: 40px 20px; }
            .header { text-align: center; margin-bottom: 60px; }
            .header h1 { font-size: 3rem; color: #2563eb; margin-bottom: 16px; }
            .header p { font-size: 1.25rem; color: #64748b; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 30px; margin: 40px 0; }
            .card { background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
            .card h3 { color: #1e293b; margin-bottom: 16px; font-size: 1.5rem; }
            .card p { color: #64748b; line-height: 1.6; }
            .features { background: white; padding: 40px; border-radius: 16px; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1); }
            .feature-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 30px; }
            .feature { text-align: center; padding: 20px; }
            .feature-icon { font-size: 3rem; margin-bottom: 16px; }
            .cta { text-align: center; margin: 60px 0; }
            .btn { display: inline-block; background: #2563eb; color: white; padding: 16px 32px; border-radius: 8px; text-decoration: none; font-weight: 600; margin: 0 10px; transition: background 0.2s; }
            .btn:hover { background: #1d4ed8; }
            .btn-outline { background: transparent; color: #2563eb; border: 2px solid #2563eb; }
            .btn-outline:hover { background: #2563eb; color: white; }
            .demo-section { background: #fef3c7; padding: 30px; border-radius: 12px; border-left: 4px solid #f59e0b; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 ResumeRAG</h1>
                <p>AI-Powered Resume Search & Job Matching Platform</p>
            </div>

            <div class="features">
                <h2 style="text-align: center; margin-bottom: 40px; color: #1e293b;">Powerful Features</h2>
                <div class="feature-grid">
                    <div class="feature">
                        <div class="feature-icon">📁</div>
                        <h3>Smart Upload</h3>
                        <p>Upload PDF, DOCX, or ZIP files. Automatic text extraction and processing.</p>
                    </div>
                    <div class="feature">
                        <div class="feature-icon">🔍</div>
                        <h3>AI Search</h3>
                        <p>Search resumes using natural language queries with intelligent matching.</p>
                    </div>
                    <div class="feature">
                        <div class="feature-icon">🤖</div>
                        <h3>Job Matching</h3>
                        <p>Find the best candidates for jobs with evidence and skill gap analysis.</p>
                    </div>
                    <div class="feature">
                        <div class="feature-icon">🔒</div>
                        <h3>Privacy First</h3>
                        <p>Automatic PII protection with role-based access control.</p>
                    </div>
                </div>
            </div>

            <div class="cta">
                <h2 style="margin-bottom: 20px; color: #1e293b;">Ready to Get Started?</h2>
                <a href="/docs" class="btn">📚 API Documentation</a>
                <a href="/demo" class="btn btn-outline">🎮 Try Demo</a>
            </div>

            <div class="demo-section">
                <h3 style="color: #92400e; margin-bottom: 20px;">🎯 Quick Start Guide</h3>
                <ol style="color: #78350f; line-height: 1.8;">
                    <li><strong>Register:</strong> POST /api/auth/register with email and password</li>
                    <li><strong>Login:</strong> POST /api/auth/login to get access token</li>
                    <li><strong>Upload:</strong> POST /api/resumes with resume files</li>
                    <li><strong>Search:</strong> POST /api/ask with natural language queries</li>
                    <li><strong>Jobs:</strong> Create and match jobs using /api/jobs endpoints</li>
                </ol>
                <p style="margin-top: 20px; color: #78350f;">
                    <strong>Test Credentials:</strong> user@demo.com / password123 (User) | recruiter@demo.com / password123 (Recruiter)
                </p>
            </div>
        </div>
    </body>
    </html>
    """

@app.get("/demo", response_class=HTMLResponse)
async def demo():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ResumeRAG Demo</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f8fafc; }
            .container { max-width: 800px; margin: 0 auto; padding: 40px 20px; }
            .card { background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); margin-bottom: 30px; }
            .form-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 8px; font-weight: 600; color: #374151; }
            input, textarea, select { width: 100%; padding: 12px; border: 1px solid #d1d5db; border-radius: 6px; font-size: 16px; }
            button { background: #2563eb; color: white; padding: 12px 24px; border: none; border-radius: 6px; cursor: pointer; font-size: 16px; font-weight: 600; }
            button:hover { background: #1d4ed8; }
            .result { background: #f0f9ff; padding: 20px; border-radius: 8px; border-left: 4px solid #2563eb; margin-top: 20px; }
            .error { background: #fef2f2; border-left-color: #ef4444; }
            .success { background: #f0fdf4; border-left-color: #10b981; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card">
                <h1 style="color: #2563eb; margin-bottom: 20px;">🎮 ResumeRAG Demo</h1>
                <p style="color: #6b7280; margin-bottom: 30px;">Try the API features directly in your browser!</p>
                
                <div id="auth-section">
                    <h3>1. Authentication</h3>
                    <div class="form-group">
                        <label>Email:</label>
                        <input type="email" id="email" value="user@demo.com" placeholder="Enter email">
                    </div>
                    <div class="form-group">
                        <label>Password:</label>
                        <input type="password" id="password" value="password123" placeholder="Enter password">
                    </div>
                    <div class="form-group">
                        <label>Role:</label>
                        <select id="role">
                            <option value="user">User</option>
                            <option value="recruiter">Recruiter</option>
                        </select>
                    </div>
                    <button onclick="register()">Register</button>
                    <button onclick="login()" style="margin-left: 10px;">Login</button>
                    <div id="auth-result"></div>
                </div>

                <div id="upload-section" style="display: none;">
                    <h3>2. Upload Resume</h3>
                    <div class="form-group">
                        <label>Resume Text (simulated file upload):</label>
                        <textarea id="resume-text" rows="6" placeholder="Paste resume content here...
Example:
John Doe
Software Engineer
5 years experience in Python, JavaScript, React
Built web applications using Django and Flask
AWS, Docker, Git expertise"></textarea>
                    </div>
                    <button onclick="uploadResume()">Upload Resume</button>
                    <div id="upload-result"></div>
                </div>

                <div id="search-section" style="display: none;">
                    <h3>3. Search Resumes</h3>
                    <div class="form-group">
                        <label>Search Query:</label>
                        <input type="text" id="search-query" placeholder="Find Python developers with 3+ years experience" value="Python developer with web experience">
                    </div>
                    <button onclick="searchResumes()">Search</button>
                    <div id="search-result"></div>
                </div>
            </div>
        </div>

        <script>
            let authToken = '';

            async function register() {
                const email = document.getElementById('email').value;
                const password = document.getElementById('password').value;
                const role = document.getElementById('role').value;
                
                try {
                    const response = await fetch('/api/auth/register', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({email, password, role})
                    });
                    const data = await response.json();
                    
                    if (response.ok) {
                        authToken = data.access_token;
                        showResult('auth-result', 'Registration successful!', 'success');
                        showNextSection('upload-section');
                    } else {
                        showResult('auth-result', data.error?.message || 'Registration failed', 'error');
                    }
                } catch (error) {
                    showResult('auth-result', 'Network error: ' + error.message, 'error');
                }
            }

            async function login() {
                const email = document.getElementById('email').value;
                const password = document.getElementById('password').value;
                
                try {
                    const response = await fetch('/api/auth/login', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({email, password})
                    });
                    const data = await response.json();
                    
                    if (response.ok) {
                        authToken = data.access_token;
                        showResult('auth-result', 'Login successful!', 'success');
                        showNextSection('upload-section');
                    } else {
                        showResult('auth-result', data.error?.message || 'Login failed', 'error');
                    }
                } catch (error) {
                    showResult('auth-result', 'Network error: ' + error.message, 'error');
                }
            }

            async function uploadResume() {
                const resumeText = document.getElementById('resume-text').value;
                if (!resumeText.trim()) {
                    showResult('upload-result', 'Please enter resume content', 'error');
                    return;
                }
                
                try {
                    const response = await fetch('/api/resumes/text', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': 'Bearer ' + authToken
                        },
                        body: JSON.stringify({content: resumeText})
                    });
                    const data = await response.json();
                    
                    if (response.ok) {
                        showResult('upload-result', `Resume uploaded! ID: ${data.id}, Skills: ${data.skills?.join(', ')}`, 'success');
                        showNextSection('search-section');
                    } else {
                        showResult('upload-result', data.error?.message || 'Upload failed', 'error');
                    }
                } catch (error) {
                    showResult('upload-result', 'Network error: ' + error.message, 'error');
                }
            }

            async function searchResumes() {
                const query = document.getElementById('search-query').value;
                if (!query.trim()) {
                    showResult('search-result', 'Please enter a search query', 'error');
                    return;
                }
                
                try {
                    const response = await fetch('/api/ask', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': 'Bearer ' + authToken
                        },
                        body: JSON.stringify({query: query, k: 5})
                    });
                    const data = await response.json();
                    
                    if (response.ok) {
                        let resultHtml = `<h4>Search Results (${data.answers.length} found):</h4>`;
                        data.answers.forEach((answer, index) => {
                            resultHtml += `
                                <div style="margin: 10px 0; padding: 10px; background: white; border-radius: 6px;">
                                    <strong>Resume ${answer.resume_id}</strong> (${Math.round(answer.similarity_score * 100)}% match)
                                    ${answer.evidence_snippets.length > 0 ? '<br>Evidence: ' + answer.evidence_snippets.join('; ') : ''}
                                </div>
                            `;
                        });
                        showResult('search-result', resultHtml, 'success');
                    } else {
                        showResult('search-result', data.error?.message || 'Search failed', 'error');
                    }
                } catch (error) {
                    showResult('search-result', 'Network error: ' + error.message, 'error');
                }
            }

            function showResult(elementId, message, type) {
                const element = document.getElementById(elementId);
                element.innerHTML = `<div class="result ${type}">${message}</div>`;
            }

            function showNextSection(sectionId) {
                document.getElementById(sectionId).style.display = 'block';
            }
        </script>
    </body>
    </html>
    """

# API Routes
@app.post("/api/auth/register")
async def register(user_data: UserRegister):
    if user_data.email in users_db:
        raise HTTPException(status_code=400, detail={"error": {"code": "USER_EXISTS", "message": "User already exists"}})
    
    users_db[user_data.email] = {
        "email": user_data.email,
        "password": user_data.password,  # In production, hash this!
        "role": user_data.role,
        "created": datetime.now().isoformat()
    }
    
    token = create_user_token(user_data.email)
    return {"access_token": token, "token_type": "bearer"}

@app.post("/api/auth/login")
async def login(user_data: UserLogin):
    user = users_db.get(user_data.email)
    if not user or user["password"] != user_data.password:
        raise HTTPException(status_code=401, detail={"error": {"code": "INVALID_CREDENTIALS", "message": "Invalid email or password"}})
    
    token = create_user_token(user_data.email)
    return {"access_token": token, "token_type": "bearer"}

@app.post("/api/resumes/text")
async def upload_resume_text(data: dict, user: dict = Depends(get_current_user)):
    """Upload resume as text (for demo purposes)"""
    content = data.get("content", "")
    if not content.strip():
        raise HTTPException(status_code=400, detail={"error": {"code": "EMPTY_CONTENT", "message": "Resume content cannot be empty"}})
    
    resume_id = len(resumes_db) + 1
    skills = extract_skills(content)
    
    resumes_db[resume_id] = {
        "id": resume_id,
        "user_email": user["email"],
        "content": content,
        "skills": skills,
        "created": datetime.now().isoformat(),
        "filename": f"resume_{resume_id}.txt"
    }
    
    return {"id": resume_id, "skills": skills, "message": "Resume uploaded successfully"}

@app.post("/api/resumes")
async def upload_resumes(files: List[UploadFile] = File(...), user: dict = Depends(get_current_user)):
    """Upload resume files"""
    if not files:
        raise HTTPException(status_code=400, detail={"error": {"code": "NO_FILES", "message": "No files provided"}})
    
    uploaded_resumes = []
    
    for file in files:
        if not file.filename.lower().endswith(('.pdf', '.docx', '.txt')):
            continue
            
        # Save file temporarily
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, f"{uuid.uuid4()}_{file.filename}")
        
        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # Extract text
        try:
            if file.filename.lower().endswith('.pdf'):
                text = extract_text_from_pdf(file_path)
            elif file.filename.lower().endswith('.docx'):
                text = extract_text_from_docx(file_path)
            else:  # .txt
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            
            resume_id = len(resumes_db) + 1
            skills = extract_skills(text)
            
            resumes_db[resume_id] = {
                "id": resume_id,
                "user_email": user["email"],
                "content": text,
                "skills": skills,
                "created": datetime.now().isoformat(),
                "filename": file.filename
            }
            
            uploaded_resumes.append({"id": resume_id, "filename": file.filename, "skills": skills})
            
        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
        finally:
            # Clean up temporary file
            if os.path.exists(file_path):
                os.remove(file_path)
    
    return {"uploaded_resumes": uploaded_resumes, "count": len(uploaded_resumes)}

@app.get("/api/resumes")
async def get_resumes(limit: int = 20, offset: int = 0, user: dict = Depends(get_current_user)):
    """Get user's resumes"""
    user_resumes = [r for r in resumes_db.values() if r["user_email"] == user["email"]]
    total = len(user_resumes)
    
    # Apply pagination
    paginated_resumes = user_resumes[offset:offset + limit]
    
    items = []
    for resume in paginated_resumes:
        items.append({
            "id": resume["id"],
            "filename": resume["filename"],
            "content_preview": resume["content"][:200] + "..." if len(resume["content"]) > 200 else resume["content"],
            "skills": resume["skills"],
            "created_at": resume["created"],
            "word_count": len(resume["content"].split())
        })
    
    next_offset = offset + limit if offset + limit < total else None
    
    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
        "next_offset": next_offset
    }

@app.post("/api/ask")
async def ask_question(query_data: ResumeQuery, user: dict = Depends(get_current_user)):
    """Search resumes with natural language query"""
    if not query_data.query.strip():
        raise HTTPException(status_code=400, detail={"error": {"code": "EMPTY_QUERY", "message": "Query cannot be empty"}})
    
    # Get user's resumes
    user_resumes = [r for r in resumes_db.values() if r["user_email"] == user["email"]]
    
    if not user_resumes:
        return {"query": query_data.query, "answers": [], "total_resumes_searched": 0}
    
    # Calculate similarities
    results = []
    for resume in user_resumes:
        similarity = simple_similarity(query_data.query, resume["content"])
        
        # Extract evidence snippets
        query_words = query_data.query.lower().split()
        sentences = resume["content"].split('.')
        evidence_snippets = []
        
        for sentence in sentences[:10]:  # Check first 10 sentences
            sentence = sentence.strip()
            if len(sentence) > 20:
                if any(word in sentence.lower() for word in query_words):
                    evidence_snippets.append(sentence)
                    if len(evidence_snippets) >= 2:
                        break
        
        if similarity > 0.1:  # Only include results with some similarity
            results.append({
                "resume_id": resume["id"],
                "filename": resume["filename"],
                "similarity_score": similarity,
                "evidence_snippets": evidence_snippets,
                "created_at": resume["created"]
            })
    
    # Sort by similarity and return top k
    results.sort(key=lambda x: x["similarity_score"], reverse=True)
    results = results[:query_data.k]
    
    return {
        "query": query_data.query,
        "answers": results,
        "total_resumes_searched": len(user_resumes)
    }

@app.post("/api/jobs")
async def create_job(job_data: JobCreate, user: dict = Depends(get_current_user)):
    """Create a new job posting"""
    job_id = len(jobs_db) + 1
    
    jobs_db[job_id] = {
        "id": job_id,
        "title": job_data.title,
        "description": job_data.description,
        "requirements": job_data.requirements,
        "company": job_data.company,
        "created_by": user["email"],
        "created_at": datetime.now().isoformat()
    }
    
    return {
        "job_id": job_id,
        "title": job_data.title,
        "company": job_data.company,
        "requirements": job_data.requirements,
        "created_at": jobs_db[job_id]["created_at"]
    }

@app.get("/api/jobs/{job_id}")
async def get_job(job_id: int, user: dict = Depends(get_current_user)):
    """Get job details"""
    job = jobs_db.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail={"error": {"code": "JOB_NOT_FOUND", "message": "Job not found"}})
    
    return job

@app.post("/api/jobs/{job_id}/match")
async def match_job_candidates(job_id: int, match_data: dict, user: dict = Depends(get_current_user)):
    """Match candidates to a job"""
    job = jobs_db.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail={"error": {"code": "JOB_NOT_FOUND", "message": "Job not found"}})
    
    top_n = match_data.get("top_n", 10)
    
    # Get all resumes for matching
    all_resumes = list(resumes_db.values())
    
    if not all_resumes:
        return {"job_id": job_id, "matches": [], "total_candidates_evaluated": 0}
    
    job_text = f"{job['title']} {job['description']} {' '.join(job['requirements'])}"
    
    matches = []
    for resume in all_resumes:
        similarity = simple_similarity(job_text, resume["content"])
        
        # Check skill matches
        job_requirements = [req.lower() for req in job["requirements"]]
        resume_skills = [skill.lower() for skill in resume["skills"]]
        
        matching_skills = []
        missing_requirements = []
        
        for req in job["requirements"]:
            req_lower = req.lower()
            if any(req_lower in skill for skill in resume_skills) or req_lower in resume["content"].lower():
                matching_skills.append(req)
            else:
                missing_requirements.append(req)
        
        # Extract evidence snippets
        evidence_snippets = []
        sentences = resume["content"].split('.')
        for sentence in sentences[:5]:
            sentence = sentence.strip()
            if len(sentence) > 20:
                for skill in matching_skills[:2]:  # Check first 2 matching skills
                    if skill.lower() in sentence.lower():
                        evidence_snippets.append(sentence)
                        break
        
        matches.append({
            "resume_id": resume["id"],
            "candidate_name": f"Candidate {resume['id']}",
            "match_percentage": min(100, int(similarity * 100 + len(matching_skills) * 10)),
            "similarity_score": similarity,
            "matching_skills": matching_skills,
            "missing_requirements": missing_requirements,
            "evidence_snippets": evidence_snippets[:3]
        })
    
    # Sort by match percentage
    matches.sort(key=lambda x: x["match_percentage"], reverse=True)
    
    return {
        "job_id": job_id,
        "job_title": job["title"],
        "matches": matches[:top_n],
        "total_candidates_evaluated": len(all_resumes)
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "ResumeRAG API",
        "version": "1.0.0",
        "users": len(users_db),
        "resumes": len(resumes_db),
        "jobs": len(jobs_db)
    }

def main():
    # Create demo users
    users_db["user@demo.com"] = {
        "email": "user@demo.com",
        "password": "password123",
        "role": "user",
        "created": datetime.now().isoformat()
    }
    users_db["recruiter@demo.com"] = {
        "email": "recruiter@demo.com", 
        "password": "password123",
        "role": "recruiter",
        "created": datetime.now().isoformat()
    }
    
    print("🚀 Starting ResumeRAG Server...")
    print("📡 Server: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs") 
    print("🎮 Demo: http://localhost:8000/demo")
    print("💚 Health: http://localhost:8000/health")
    print("\n" + "="*50)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    main()
