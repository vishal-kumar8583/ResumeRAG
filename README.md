# ResumeRAG - AI-Powered Resume Search & Job Matching

A modern web application that provides intelligent resume search, parsing, and job matching capabilities using RAG (Retrieval-Augmented Generation) and semantic search.

## 🚀 Features

- **Multi-format Resume Upload**: Support for PDF, DOCX, DOC, TXT, and ZIP files
- **AI-Powered Search**: Natural language queries with semantic similarity matching
- **Smart PII Redaction**: Automatic privacy protection for non-recruiter users
- **Job Matching**: Intelligent candidate-job matching with evidence and missing requirements
- **Modern UI**: Beautiful, responsive interface built with React and Tailwind CSS
- **Robust API**: RESTful endpoints with pagination, rate limiting, and idempotency
- **Role-based Access**: Different permissions for job seekers vs recruiters

## 🏗️ Architecture

### Backend Stack
- **FastAPI**: Modern Python web framework
- **SQLAlchemy**: Database ORM with SQLite/PostgreSQL support
- **SentenceTransformers**: AI embeddings for semantic search
- **PyPDF2 & python-docx**: Document parsing
- **Redis**: Rate limiting and caching
- **JWT**: Authentication and authorization

### Frontend Stack
- **React 18**: Modern React with hooks
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first styling
- **React Query**: Data fetching and caching
- **React Router**: Client-side routing
- **Vite**: Fast build tool

## 📋 API Documentation

### Base URL
```
http://localhost:8000/api
```

### Authentication

All protected endpoints require a Bearer token in the Authorization header:
```
Authorization: Bearer <your_token>
```

### Core Endpoints

#### Authentication

##### Register User
```http
POST /api/auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "password123",
  "role": "user" // or "recruiter"
}
```

Response:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer"
}
```

##### Login User
```http
POST /api/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "password123"
}
```

#### Resume Management

##### Upload Resumes
```http
POST /api/resumes
Authorization: Bearer <token>
Content-Type: multipart/form-data
Idempotency-Key: upload_123_abc

files: [File, File, ...]
```

Response:
```json
{
  "uploaded_resumes": [
    {
      "id": 1,
      "filename": "john_doe_resume.pdf",
      "skills": ["Python", "JavaScript", "React"],
      "word_count": 450
    }
  ],
  "count": 1
}
```

##### Get Resumes (with Pagination)
```http
GET /api/resumes?limit=20&offset=0&q=python
Authorization: Bearer <token>
```

Response:
```json
{
  "items": [
    {
      "id": 1,
      "filename": "resume.pdf",
      "content_preview": "Experienced software developer...",
      "created_at": "2024-01-01T12:00:00Z",
      "word_count": 450,
      "is_pii_redacted": true
    }
  ],
  "total": 5,
  "limit": 20,
  "offset": 0,
  "next_offset": null
}
```

##### Get Specific Resume
```http
GET /api/resumes/{resume_id}
Authorization: Bearer <token>
```

Response:
```json
{
  "id": 1,
  "filename": "resume.pdf",
  "content": "Full resume content...",
  "skills": ["Python", "JavaScript", "React"],
  "experience": [
    {
      "period": "2020-2023",
      "context": "Software Engineer at Tech Corp..."
    }
  ],
  "created_at": "2024-01-01T12:00:00Z",
  "word_count": 450,
  "is_pii_redacted": true
}
```

#### AI Search

##### Ask Questions
```http
POST /api/ask
Authorization: Bearer <token>
Content-Type: application/json

{
  "query": "Find developers with Python and machine learning experience",
  "k": 5
}
```

Response:
```json
{
  "query": "Find developers with Python and machine learning experience",
  "answers": [
    {
      "resume_id": 1,
      "filename": "resume.pdf",
      "similarity_score": 0.85,
      "evidence_snippets": [
        "Experienced Python developer with 5 years in ML",
        "Built machine learning models using TensorFlow"
      ],
      "created_at": "2024-01-01T12:00:00Z"
    }
  ],
  "total_resumes_searched": 10
}
```

#### Job Management

##### Create Job
```http
POST /api/jobs
Authorization: Bearer <token>
Content-Type: application/json
Idempotency-Key: job_123_abc

{
  "title": "Senior Python Developer",
  "description": "Looking for an experienced Python developer...",
  "requirements": ["Python", "Django", "PostgreSQL"],
  "company": "Tech Corp"
}
```

Response:
```json
{
  "job_id": 1,
  "title": "Senior Python Developer",
  "company": "Tech Corp",
  "requirements": ["Python", "Django", "PostgreSQL"],
  "created_at": "2024-01-01T12:00:00Z"
}
```

##### Get Job
```http
GET /api/jobs/{job_id}
Authorization: Bearer <token>
```

##### Match Job to Candidates
```http
POST /api/jobs/{job_id}/match
Authorization: Bearer <token>
Content-Type: application/json

{
  "top_n": 10
}
```

Response:
```json
{
  "job_id": 1,
  "job_title": "Senior Python Developer",
  "matches": [
    {
      "resume_id": 1,
      "candidate_name": "John Doe",
      "match_percentage": 85,
      "similarity_score": 0.85,
      "matching_skills": ["Python", "Django"],
      "missing_requirements": ["PostgreSQL"],
      "evidence_snippets": [
        "5 years of Python development experience",
        "Built web applications using Django framework"
      ]
    }
  ],
  "total_candidates_evaluated": 20
}
```

### Error Responses

All errors follow a consistent format:
```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable message",
    "field": "field_name" // optional
  }
}
```

Common error codes:
- `FIELD_REQUIRED`: Missing required field
- `INVALID_CREDENTIALS`: Authentication failed
- `RATE_LIMIT`: Rate limit exceeded (429 status)
- `UNAUTHORIZED`: Invalid or expired token (401 status)
- `NOT_FOUND`: Resource not found (404 status)

## 🚦 Rate Limiting

- **Limit**: 60 requests per minute per user
- **Response**: 429 status code when exceeded
- **Headers**: Rate limit info in response headers

## 🔐 Test Credentials

### Pre-seeded Users
- **Regular User**: `user@demo.com` / `password123`
- **Recruiter**: `recruiter@demo.com` / `password123`

### User Roles
- **user**: Can upload and search their own resumes
- **recruiter**: Can access all resumes and see unredacted PII

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.8+
- Node.js 16+
- Redis (optional, for rate limiting)
- PostgreSQL (optional, SQLite used by default)

### Backend Setup

1. **Clone and navigate to project**:
   ```bash
   cd hackathon
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

5. **Create uploads directory**:
   ```bash
   mkdir uploads
   ```

6. **Run the application**:
   ```bash
   python main.py
   ```

The API will be available at `http://localhost:8000`

### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start development server**:
   ```bash
   npm run dev
   ```

The frontend will be available at `http://localhost:5173`

### Production Setup

1. **Build frontend**:
   ```bash
   cd frontend
   npm run build
   ```

2. **Configure environment variables**:
   ```bash
   export DATABASE_URL="postgresql://user:pass@localhost/resumerag"
   export REDIS_URL="redis://localhost:6379"
   export SECRET_KEY="your-production-secret-key"
   ```

3. **Run with production server**:
   ```bash
   pip install gunicorn
   gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

## 📊 Example Usage Flow

1. **Register/Login**: Create account or login
2. **Upload Resumes**: Upload PDF/DOCX files or ZIP archives
3. **Search**: Ask natural language questions about candidates
4. **Create Jobs**: Define job requirements and descriptions
5. **Match**: Find best candidates for specific jobs
6. **Review**: Examine candidate profiles and evidence

## 🧪 Testing

### Sample Resume Upload
```bash
curl -X POST "http://localhost:8000/api/resumes" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@sample_resume.pdf"
```

### Sample Search Query
```bash
curl -X POST "http://localhost:8000/api/ask" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find software engineers with React experience",
    "k": 5
  }'
```

## 🎯 Judge Requirements Checklist

- ✅ **API Correctness**: All required endpoints implemented
- ✅ **Pagination**: `?limit=&offset=` → `{items, next_offset}`
- ✅ **Idempotency**: `Idempotency-Key` headers supported
- ✅ **Rate Limits**: 60 req/min/user with 429 responses
- ✅ **Error Format**: Consistent error response structure
- ✅ **CORS**: Open CORS for judging
- ✅ **Auth Endpoints**: Register and login implemented
- ✅ **Resume Upload**: Multipart upload with ZIP support
- ✅ **Search**: Semantic search with evidence snippets
- ✅ **Job Matching**: Evidence and missing requirements
- ✅ **PII Redaction**: Automatic privacy protection
- ✅ **Modern UI**: React frontend with all required pages

## 🏆 Key Features for Judges

1. **Upload 3+ resumes**: Drag & drop interface supports bulk upload
2. **Ask endpoint**: Returns schema-compliant answers with evidence
3. **Job matching**: Deterministic rankings with evidence and gaps
4. **Pagination**: Proper offset-based pagination throughout
5. **Modern UI**: Professional, responsive design
6. **Robustness**: Rate limiting, idempotency, error handling

## 📝 License

This project was created for a hackathon. All rights reserved.

## 🤝 Contributing

This is a hackathon project. For questions or issues, please contact the development team.
