from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from jose import JWTError, jwt
from datetime import datetime, timedelta
import os
import json
import hashlib
import redis
import asyncio
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel
import uvicorn
from collections import Counter

# Environment variables
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./resumerag.db")
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")

# Create uploads directory
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Database setup
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Redis setup for rate limiting - disabled for demo
redis_client = None
print("Redis disabled for demo - rate limiting disabled")

# Password hashing - using bcrypt directly due to passlib compatibility issues
import bcrypt
security = HTTPBearer()

# Pydantic models
class UserCreate(BaseModel):
    email: str
    password: str
    role: str = "user"

class UserLogin(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class ResumeQuery(BaseModel):
    query: str
    k: int = 5

class JobCreate(BaseModel):
    title: str
    description: str
    requirements: List[str]
    company: str

class JobMatch(BaseModel):
    top_n: int = 10

class ErrorResponse(BaseModel):
    error: Dict[str, Any]

# Database models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="user")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    resumes = relationship("Resume", back_populates="user")

class Resume(Base):
    __tablename__ = "resumes"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Text)  # JSON string of embedding vector
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    is_pii_redacted = Column(Boolean, default=False)
    
    user = relationship("User", back_populates="resumes")

class Job(Base):
    __tablename__ = "jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    requirements = Column(Text)  # JSON string
    company = Column(String, nullable=False)
    embedding = Column(Text)  # JSON string of embedding vector
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# FastAPI app
app = FastAPI(title="ResumeRAG API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
try:
    # Mount assets first (more specific)
    app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")
    # Mount all other static files  
    app.mount("/static", StaticFiles(directory="frontend/dist"), name="static")
    print("Static files mounted successfully")
except Exception as e:
    print(f"Warning: Could not mount static files: {e}")

# Dependencies
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.on_event("startup")
def seed_demo_users() -> None:
    """Create demo users if they don't exist, so the UI can login immediately."""
    try:
        db = SessionLocal()
        demo_users = [
            ("user@demo.com", "password123", "user"),
            ("recruiter@demo.com", "password123", "recruiter"),
        ]
        for email, password, role in demo_users:
            existing = db.query(User).filter(User.email == email).first()
            if not existing:
                db.add(User(email=email, hashed_password=get_password_hash(password), role=role))
        db.commit()
    except Exception as e:
        print(f"Startup seed skipped: {e}")
    finally:
        try:
            db.close()
        except Exception:
            pass

def create_error_response(code: str, message: str, field: str = None):
    error = {"code": code, "message": message}
    if field:
        error["field"] = field
    return {"error": error}

def verify_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())

def get_password_hash(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    credentials_exception = HTTPException(
        status_code=401,
        detail=create_error_response("UNAUTHORIZED", "Could not validate credentials"),
    )
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    return user

async def rate_limit_check(request: Request, user: User = Depends(get_current_user)):
    if not redis_client:
        return user
    
    key = f"rate_limit:{user.id}"
    current_minute = int(datetime.utcnow().timestamp() // 60)
    
    try:
        pipe = redis_client.pipeline()
        pipe.hincrby(key, current_minute, 1)
        pipe.expire(key, 120)  # Keep for 2 minutes
        results = pipe.execute()
        
        if results[0] > 60:
            raise HTTPException(
                status_code=429,
                detail=create_error_response("RATE_LIMIT", "Rate limit exceeded")
            )
    except:
        pass  # If Redis fails, continue without rate limiting
    
    return user

# Authentication endpoints
@app.post("/api/auth/register", response_model=Token)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    # Check if user exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=400,
            detail=create_error_response("USER_EXISTS", "Email already registered", "email")
        )
    
    # Create user
    hashed_password = get_password_hash(user_data.password)
    user = User(
        email=user_data.email,
        hashed_password=hashed_password,
        role=user_data.role
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    # Create token
    access_token = create_access_token(
        data={"sub": user.email},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/api/auth/login", response_model=Token)
async def login(user_data: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == user_data.email).first()
    if not user or not verify_password(user_data.password, user.hashed_password):
        raise HTTPException(
            status_code=401,
            detail=create_error_response("INVALID_CREDENTIALS", "Invalid email or password")
        )
    
    access_token = create_access_token(
        data={"sub": user.email},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/")
async def root():
    """Serve the landing page"""
    try:
        with open("landing.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return {"message": "ResumeRAG API is running. Landing page not found."}

@app.get("/app")
async def serve_app():
    """Serve the React application"""
    try:
        with open("frontend/dist/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return {"message": "Frontend application not built yet."}

# Serve React app for all frontend routes
@app.get("/login")
@app.get("/register")
@app.get("/upload")
@app.get("/search")
@app.get("/jobs")
@app.get("/candidates/{path:path}")
async def serve_frontend_routes():
    """Serve the React application for frontend routes"""
    try:
        with open("frontend/dist/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return {"message": "Frontend application not built yet."}

@app.get("/vite.svg")
async def serve_vite_svg():
    """Serve vite.svg icon"""
    try:
        import os
        from fastapi.responses import FileResponse, Response
        svg_path = "frontend/dist/vite.svg"
        if os.path.exists(svg_path):
            return FileResponse(svg_path)
        # Return lightweight inline SVG placeholder to avoid 500s when not built
        placeholder_svg = """
<svg xmlns='http://www.w3.org/2000/svg' width='64' height='64' viewBox='0 0 64 64'>
  <rect width='64' height='64' rx='12' fill='#f3f4f6'/>
  <circle cx='32' cy='32' r='14' fill='#3b82f6'/>
  <text x='32' y='38' text-anchor='middle' font-size='16' fill='white' font-family='Arial, sans-serif'>APP</text>
</svg>
""".strip()
        return Response(content=placeholder_svg, media_type="image/svg+xml")
    except Exception:
        return {"message": "Icon not found"}

# Import resume service
try:
    from services.resume_service import ResumeParsingService
except ImportError:
    print("Advanced resume service not available, using simplified version")
    from services.resume_service_simple import ResumeParsingService
import aiofiles
import zipfile
import tempfile
import shutil

# Initialize resume service
resume_service = ResumeParsingService()

# Resume endpoints
@app.post("/api/resumes")
async def upload_resumes(
    files: List[UploadFile] = File(...),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    user: User = Depends(rate_limit_check),
    db: Session = Depends(get_db)
):
    if not files:
        raise HTTPException(
            status_code=400,
            detail=create_error_response("NO_FILES", "No files provided")
        )
    
    # Check for idempotency
    if idempotency_key:
        existing_resumes = db.query(Resume).filter(
            Resume.user_id == user.id,
            Resume.original_filename.contains(idempotency_key)
        ).all()
        if existing_resumes:
            return {"message": "Files already processed", "resume_ids": [r.id for r in existing_resumes]}
    
    uploaded_resumes = []
    
    for file in files:
        try:
            # Validate file type
            if not file.filename.lower().endswith(('.pdf', '.docx', '.doc', '.txt', '.zip')):
                raise HTTPException(
                    status_code=400,
                    detail=create_error_response("INVALID_FILE_TYPE", f"Unsupported file type: {file.filename}")
                )
            
            # Handle ZIP files
            if file.filename.lower().endswith('.zip'):
                extracted_files = await handle_zip_upload(file, user, db)
                uploaded_resumes.extend(extracted_files)
                continue
            
            # Save file
            file_extension = file.filename.split('.')[-1]
            unique_filename = f"{user.id}_{int(datetime.utcnow().timestamp())}_{file.filename}"
            file_path = os.path.join(UPLOAD_DIR, unique_filename)
            
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            # Process resume
            try:
                processed_data = resume_service.process_resume(file_path, file.filename, user.role)
                
                # Save to database
                resume = Resume(
                    filename=unique_filename,
                    original_filename=file.filename,
                    content=processed_data['content'],
                    embedding=json.dumps(processed_data['embedding']),
                    user_id=user.id,
                    is_pii_redacted=processed_data['is_pii_redacted']
                )
                db.add(resume)
                db.commit()
                db.refresh(resume)
                
                uploaded_resumes.append({
                    "id": resume.id,
                    "filename": resume.original_filename,
                    "skills": processed_data['skills'],
                    "word_count": processed_data['word_count']
                })
                
            except Exception as e:
                # Clean up file on processing error
                if os.path.exists(file_path):
                    os.remove(file_path)
                raise HTTPException(
                    status_code=400,
                    detail=create_error_response("PROCESSING_ERROR", str(e))
                )
                
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=create_error_response("UPLOAD_ERROR", f"Error uploading {file.filename}: {str(e)}")
            )
    
    return {"uploaded_resumes": uploaded_resumes, "count": len(uploaded_resumes)}

async def handle_zip_upload(zip_file: UploadFile, user: User, db: Session) -> List[Dict]:
    """Handle ZIP file upload and extract resumes"""
    extracted_resumes = []
    
    # Save ZIP file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
        content = await zip_file.read()
        tmp_zip.write(content)
        tmp_zip_path = tmp_zip.name
    
    try:
        # Extract ZIP file
        with zipfile.ZipFile(tmp_zip_path, 'r') as zip_ref:
            with tempfile.TemporaryDirectory() as temp_dir:
                zip_ref.extractall(temp_dir)
                
                # Process extracted files
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file.lower().endswith(('.pdf', '.docx', '.doc', '.txt')):
                            file_path = os.path.join(root, file)
                            
                            try:
                                # Process resume
                                processed_data = resume_service.process_resume(file_path, file, user.role)
                                
                                # Copy to uploads directory
                                unique_filename = f"{user.id}_{int(datetime.utcnow().timestamp())}_{file}"
                                final_path = os.path.join(UPLOAD_DIR, unique_filename)
                                shutil.copy2(file_path, final_path)
                                
                                # Save to database
                                resume = Resume(
                                    filename=unique_filename,
                                    original_filename=file,
                                    content=processed_data['content'],
                                    embedding=json.dumps(processed_data['embedding']),
                                    user_id=user.id,
                                    is_pii_redacted=processed_data['is_pii_redacted']
                                )
                                db.add(resume)
                                db.commit()
                                db.refresh(resume)
                                
                                extracted_resumes.append({
                                    "id": resume.id,
                                    "filename": resume.original_filename,
                                    "skills": processed_data['skills'],
                                    "word_count": processed_data['word_count']
                                })
                                
                            except Exception as e:
                                print(f"Error processing {file}: {str(e)}")
                                continue
    finally:
        # Clean up temporary ZIP file
        os.unlink(tmp_zip_path)
    
    return extracted_resumes

@app.get("/api/resumes")
async def get_resumes(
    limit: int = 20,
    offset: int = 0,
    q: Optional[str] = None,
    user: User = Depends(rate_limit_check),
    db: Session = Depends(get_db)
):
    query = db.query(Resume).filter(Resume.user_id == user.id)
    
    # Apply search filter if provided
    if q:
        query = query.filter(Resume.content.contains(q))
    
    # Get total count
    total = query.count()
    
    # Apply pagination
    resumes = query.offset(offset).limit(limit).all()
    
    items = []
    for resume in resumes:
        content = resume.content
        if user.role != "recruiter" and resume.is_pii_redacted:
            content = resume_service.redact_pii(content)
        
        items.append({
            "id": resume.id,
            "filename": resume.original_filename,
            "content_preview": content[:200] + "..." if len(content) > 200 else content,
            "created_at": resume.created_at.isoformat(),
            "word_count": len(resume.content.split()),
            "is_pii_redacted": resume.is_pii_redacted
        })
    
    next_offset = offset + limit if offset + limit < total else None
    
    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
        "next_offset": next_offset
    }

@app.get("/api/resumes/{resume_id}")
async def get_resume(
    resume_id: int,
    user: User = Depends(rate_limit_check),
    db: Session = Depends(get_db)
):
    resume = db.query(Resume).filter(
        Resume.id == resume_id,
        Resume.user_id == user.id
    ).first()
    
    if not resume:
        raise HTTPException(
            status_code=404,
            detail=create_error_response("RESUME_NOT_FOUND", "Resume not found")
        )
    
    content = resume.content
    if user.role != "recruiter" and resume.is_pii_redacted:
        content = resume_service.redact_pii(content)
    
    # Extract skills and experience
    skills = resume_service.extract_skills(resume.content)
    experience = resume_service.extract_experience(resume.content)
    
    return {
        "id": resume.id,
        "filename": resume.original_filename,
        "content": content,
        "skills": skills,
        "experience": experience,
        "created_at": resume.created_at.isoformat(),
        "word_count": len(resume.content.split()),
        "is_pii_redacted": resume.is_pii_redacted
    }

@app.post("/api/ask")
async def ask_question(
    query_data: ResumeQuery,
    user: User = Depends(rate_limit_check),
    db: Session = Depends(get_db)
):
    if not query_data.query.strip():
        raise HTTPException(
            status_code=400,
            detail=create_error_response("EMPTY_QUERY", "Query cannot be empty", "query")
        )
    
    # Get user's resumes
    resumes = db.query(Resume).filter(Resume.user_id == user.id).all()
    
    if not resumes:
        return {
            "query": query_data.query,
            "answers": [],
            "total_resumes_searched": 0
        }
    
    try:
        # Try advanced semantic search first
        resume_data = []
        for resume in resumes:
            resume_data.append({
                'id': resume.id,
                'content': resume.content,
                'embedding': resume.embedding,
                'filename': resume.original_filename,
                'created_at': resume.created_at.isoformat()
            })
        
        # Use advanced semantic search
        search_results = resume_service.advanced_semantic_search(
            query_data.query, resume_data, query_data.k
        )
        
        # If advanced search fails or returns empty, fall back to basic search
        if not search_results:
            print("Advanced search failed, falling back to basic search")
            search_results = _fallback_basic_search(query_data.query, resumes, query_data.k)
        
        # Prepare answers with enhanced evidence
        answers = []
        for result in search_results:
            if isinstance(result, dict) and 'resume_id' in result:
                # Advanced search result
                resume_id = result['resume_id']
                resume = next(r for r in resumes if r.id == resume_id)
                
                content = resume.content
                if user.role != "recruiter" and resume.is_pii_redacted:
                    content = resume_service.redact_pii(content)
                
                # Enhanced snippet extraction
                relevant_snippets = resume_service._extract_relevant_snippets(
                    content, query_data.query, max_snippets=3
                )
                
                # Extract additional insights
                skills = resume_service.extract_skills(content)
                experience = resume_service.extract_experience(content)
                
                answers.append({
                    "resume_id": resume.id,
                    "filename": resume.original_filename,
                    "semantic_score": result.get('semantic_score', 0.0),
                    "fuzzy_score": result.get('fuzzy_score', 0.0),
                    "combined_score": result.get('combined_score', 0.0),
                    "evidence_snippets": relevant_snippets,
                    "skills_found": skills[:5],  # Top 5 skills
                    "experience_summary": experience[:2],  # Top 2 experiences
                    "created_at": resume.created_at.isoformat()
                })
            else:
                # Basic search result (tuple format)
                resume_id, similarity = result
                resume = next(r for r in resumes if r.id == resume_id)
                
                content = resume.content
                if user.role != "recruiter" and resume.is_pii_redacted:
                    content = resume_service.redact_pii(content)
                
                # Basic snippet extraction
                relevant_snippets = _extract_basic_snippets(content, query_data.query)
                
                # Extract skills and experience
                skills = resume_service.extract_skills(content)
                experience = resume_service.extract_experience(content)
                
                answers.append({
                    "resume_id": resume.id,
                    "filename": resume.original_filename,
                    "similarity_score": similarity,
                    "evidence_snippets": relevant_snippets,
                    "skills_found": skills[:5],
                    "experience_summary": experience[:2],
                    "created_at": resume.created_at.isoformat()
                })
        
        return {
            "query": query_data.query,
            "answers": answers,
            "total_resumes_searched": len(resumes),
            "search_method": "advanced_semantic" if search_results and isinstance(search_results[0], dict) else "basic_fallback"
        }
        
    except Exception as e:
        print(f"Error in ask_question: {e}")
        # Fallback to basic search
        search_results = _fallback_basic_search(query_data.query, resumes, query_data.k)
        
        answers = []
        for resume_id, similarity in search_results:
            resume = next(r for r in resumes if r.id == resume_id)
            
            content = resume.content
            if user.role != "recruiter" and resume.is_pii_redacted:
                content = resume_service.redact_pii(content)
            
            relevant_snippets = _extract_basic_snippets(content, query_data.query)
            skills = resume_service.extract_skills(content)
            experience = resume_service.extract_experience(content)
            
            answers.append({
                "resume_id": resume.id,
                "filename": resume.original_filename,
                "similarity_score": similarity,
                "evidence_snippets": relevant_snippets,
                "skills_found": skills[:5],
                "experience_summary": experience[:2],
                "created_at": resume.created_at.isoformat()
            })
        
        return {
            "query": query_data.query,
            "answers": answers,
            "total_resumes_searched": len(resumes),
            "search_method": "basic_fallback"
        }

def _fallback_basic_search(query: str, resumes: List[Resume], k: int) -> List[Tuple[int, float]]:
    """Fallback basic search when advanced search fails"""
    try:
        # Create simple query embedding
        query_embedding = resume_service.create_embedding(query)
        
        # Prepare resume embeddings
        resume_embeddings = []
        for resume in resumes:
            if resume.embedding:
                try:
                    embedding = json.loads(resume.embedding) if isinstance(resume.embedding, str) else resume.embedding
                    resume_embeddings.append((resume.id, embedding))
                except:
                    # If embedding is corrupted, create a new one
                    new_embedding = resume_service.create_embedding(resume.content)
                    resume_embeddings.append((resume.id, new_embedding))
        
        if not resume_embeddings:
            return []
        
        # Use basic similarity search
        return resume_service.similarity_search(query_embedding, resume_embeddings, k)
        
    except Exception as e:
        print(f"Error in fallback search: {e}")
        return []

def _extract_basic_snippets(content: str, query: str) -> List[str]:
    """Basic snippet extraction for fallback"""
    try:
        sentences = content.split('.')
        query_words = query.lower().split()
        relevant_snippets = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:
                sentence_lower = sentence.lower()
                if any(word in sentence_lower for word in query_words):
                    relevant_snippets.append(sentence + '.')
                    if len(relevant_snippets) >= 3:
                        break
        
        return relevant_snippets
    except:
        return []

# Advanced search endpoints
@app.post("/api/search/advanced")
async def advanced_search(
    query_data: ResumeQuery,
    search_type: str = "semantic",  # semantic, fuzzy, hybrid
    user: User = Depends(rate_limit_check),
    db: Session = Depends(get_db)
):
    """Advanced search with multiple algorithms"""
    if not query_data.query.strip():
        raise HTTPException(
            status_code=400,
            detail=create_error_response("EMPTY_QUERY", "Query cannot be empty", "query")
        )
    
    # Get user's resumes
    resumes = db.query(Resume).filter(Resume.user_id == user.id).all()
    
    if not resumes:
        return {
            "query": query_data.query,
            "results": [],
            "total_resumes_searched": 0,
            "search_type": search_type
        }
    
    # Prepare resume data
    resume_data = []
    for resume in resumes:
        resume_data.append({
            'id': resume.id,
            'content': resume.content,
            'embedding': resume.embedding,
            'filename': resume.original_filename,
            'created_at': resume.created_at.isoformat()
        })
    
    # Perform search based on type
    if search_type == "semantic":
        results = resume_service.advanced_semantic_search(
            query_data.query, resume_data, query_data.k
        )
    elif search_type == "fuzzy":
        results = resume_service._fuzzy_search_only(
            query_data.query, resume_data, query_data.k
        )
    else:  # hybrid
        results = resume_service.advanced_semantic_search(
            query_data.query, resume_data, query_data.k
        )
    
    # Format results
    formatted_results = []
    for result in results:
        resume_id = result['resume_id']
        resume = next(r for r in resumes if r.id == resume_id)
        
        content = resume.content
        if user.role != "recruiter" and resume.is_pii_redacted:
            content = resume_service.redact_pii(content)
        
        formatted_results.append({
            "resume_id": resume.id,
            "filename": resume.original_filename,
            "scores": {
                "semantic": result.get('semantic_score', 0),
                "fuzzy": result.get('fuzzy_score', 0),
                "combined": result.get('combined_score', 0)
            },
            "content_preview": content[:200] + "..." if len(content) > 200 else content,
            "created_at": resume.created_at.isoformat()
        })
    
    return {
        "query": query_data.query,
        "results": formatted_results,
        "total_resumes_searched": len(resumes),
        "search_type": search_type
    }

@app.get("/api/search/suggestions")
async def get_search_suggestions(
    q: str,
    user: User = Depends(rate_limit_check),
    db: Session = Depends(get_db)
):
    """Get search suggestions based on resume content"""
    if len(q.strip()) < 2:
        return {"suggestions": []}
    
    # Get user's resumes
    resumes = db.query(Resume).filter(Resume.user_id == user.id).all()
    
    if not resumes:
        return {"suggestions": []}
    
    # Extract suggestions from resume content
    suggestions = resume_service._generate_search_suggestions(q, resumes)
    
    return {"suggestions": suggestions}

# Analytics endpoints
@app.get("/api/analytics/search-stats")
async def get_search_stats(
    user: User = Depends(rate_limit_check),
    db: Session = Depends(get_db)
):
    """Get search statistics for the user"""
    # Get user's resumes
    resumes = db.query(Resume).filter(Resume.user_id == user.id).all()
    
    if not resumes:
        return {
            "total_resumes": 0,
            "total_searches": 0,
            "avg_resume_length": 0,
            "skill_distribution": {},
            "experience_levels": {}
        }
    
    # Calculate statistics
    total_resumes = len(resumes)
    total_words = sum(len(resume.content.split()) for resume in resumes)
    avg_resume_length = total_words / total_resumes if total_resumes > 0 else 0
    
    # Extract skills from all resumes
    all_skills = []
    experience_levels = []
    
    for resume in resumes:
        skills = resume_service.extract_skills(resume.content)
        all_skills.extend(skills)
        
        experience_info = resume_service._extract_experience_info(resume.content)
        experience_levels.append(experience_info['level'])
    
    # Count skill distribution
    skill_counts = Counter(all_skills)
    skill_distribution = dict(skill_counts.most_common(10))
    
    # Count experience levels
    exp_counts = Counter(experience_levels)
    experience_levels_dist = dict(exp_counts)
    
    return {
        "total_resumes": total_resumes,
        "total_searches": 0,  # Would need to track this separately
        "avg_resume_length": round(avg_resume_length, 2),
        "skill_distribution": skill_distribution,
        "experience_levels": experience_levels_dist,
        "last_updated": datetime.utcnow().isoformat()
    }

@app.get("/api/analytics/resume-insights")
async def get_resume_insights(
    resume_id: int,
    user: User = Depends(rate_limit_check),
    db: Session = Depends(get_db)
):
    """Get detailed insights for a specific resume"""
    resume = db.query(Resume).filter(
        Resume.id == resume_id,
        Resume.user_id == user.id
    ).first()
    
    if not resume:
        raise HTTPException(
            status_code=404,
            detail=create_error_response("RESUME_NOT_FOUND", "Resume not found")
        )
    
    # Extract comprehensive insights
    skills = resume_service.extract_skills(resume.content)
    experience = resume_service.extract_experience(resume.content)
    experience_info = resume_service._extract_experience_info(resume.content)
    
    # Analyze content quality
    content_analysis = resume_service._analyze_content_quality(resume.content)
    
    # Generate improvement suggestions
    suggestions = resume_service._generate_improvement_suggestions(resume.content, skills)
    
    return {
        "resume_id": resume.id,
        "filename": resume.original_filename,
        "skills": skills,
        "experience": experience,
        "experience_level": experience_info['level'],
        "years_experience": experience_info['years'],
        "content_analysis": content_analysis,
        "improvement_suggestions": suggestions,
        "word_count": len(resume.content.split()),
        "created_at": resume.created_at.isoformat()
    }

@app.get("/api/analytics/job-market-insights")
async def get_job_market_insights(
    user: User = Depends(rate_limit_check),
    db: Session = Depends(get_db)
):
    """Get job market insights based on user's skills"""
    # Get user's resumes
    resumes = db.query(Resume).filter(Resume.user_id == user.id).all()
    
    if not resumes:
        return {"insights": [], "message": "No resumes found"}
    
    # Extract all skills
    all_skills = []
    for resume in resumes:
        skills = resume_service.extract_skills(resume.content)
        all_skills.extend(skills)
    
    # Get unique skills
    unique_skills = list(set(all_skills))
    
    # Generate market insights
    insights = resume_service._generate_market_insights(unique_skills)
    
    return {
        "user_skills": unique_skills,
        "market_insights": insights,
        "skill_demand_level": resume_service._calculate_skill_demand(unique_skills),
        "recommended_skills": resume_service._recommend_additional_skills(unique_skills)
    }

# Job endpoints
@app.post("/api/jobs")
async def create_job(
    job_data: JobCreate,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    user: User = Depends(rate_limit_check),
    db: Session = Depends(get_db)
):
    # Check for idempotency
    if idempotency_key:
        existing_job = db.query(Job).filter(
            Job.title == job_data.title,
            Job.company == job_data.company
        ).first()
        if existing_job:
            return {"job_id": existing_job.id, "message": "Job already exists"}
    
    # Create job description for embedding
    job_text = f"{job_data.title} at {job_data.company}. {job_data.description}"
    if job_data.requirements:
        job_text += f" Requirements: {', '.join(job_data.requirements)}"
    
    # Create embedding
    embedding = resume_service.create_embedding(job_text)
    
    # Save job
    job = Job(
        title=job_data.title,
        description=job_data.description,
        requirements=json.dumps(job_data.requirements),
        company=job_data.company,
        embedding=json.dumps(embedding)
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    
    return {
        "job_id": job.id,
        "title": job.title,
        "company": job.company,
        "requirements": job_data.requirements,
        "created_at": job.created_at.isoformat()
    }

@app.get("/api/jobs/{job_id}")
async def get_job(
    job_id: int,
    user: User = Depends(rate_limit_check),
    db: Session = Depends(get_db)
):
    job = db.query(Job).filter(Job.id == job_id).first()
    
    if not job:
        raise HTTPException(
            status_code=404,
            detail=create_error_response("JOB_NOT_FOUND", "Job not found")
        )
    
    return {
        "id": job.id,
        "title": job.title,
        "description": job.description,
        "requirements": json.loads(job.requirements),
        "company": job.company,
        "created_at": job.created_at.isoformat()
    }

@app.post("/api/jobs/{job_id}/match")
async def match_job_candidates(
    job_id: int,
    match_data: JobMatch,
    user: User = Depends(rate_limit_check),
    db: Session = Depends(get_db)
):
    # Get job
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=404,
            detail=create_error_response("JOB_NOT_FOUND", "Job not found")
        )
    
    # Get all resumes (for recruiter) or user's resumes
    if user.role == "recruiter":
        resumes = db.query(Resume).all()
    else:
        resumes = db.query(Resume).filter(Resume.user_id == user.id).all()
    
    if not resumes:
        return {
            "job_id": job_id,
            "matches": [],
            "total_candidates_evaluated": 0
        }
    
    # Prepare resume data for matching
    resume_data = []
    for resume in resumes:
        resume_data.append({
            'id': resume.id,
            'content': resume.content,
            'embedding': resume.embedding,
            'skills': json.dumps(resume_service.extract_skills(resume.content))
        })
    
    # Match resumes to job
    job_embedding = json.loads(job.embedding)
    job_requirements = json.loads(job.requirements)
    
    matches = resume_service.match_job_to_resumes(
        job_embedding, job_requirements, resume_data, match_data.top_n
    )
    
    # Format response
    formatted_matches = []
    for match in matches:
        formatted_matches.append({
            "resume_id": match['resume_id'],
            "candidate_name": match['candidate_name'],
            "match_percentage": match['match_percentage'],
            "similarity_score": match['similarity_score'],
            "matching_skills": match['matching_skills'],
            "missing_requirements": match['missing_requirements'],
            "evidence_snippets": match['evidence_snippets']
        })
    
    return {
        "job_id": job_id,
        "job_title": job.title,
        "matches": formatted_matches,
        "total_candidates_evaluated": len(resumes)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
