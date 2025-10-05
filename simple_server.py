#!/usr/bin/env python3

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# Create FastAPI app
app = FastAPI(
    title="ResumeRAG API",
    description="AI-Powered Resume Search & Job Matching",
    version="1.0.0"
)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ResumeRAG - AI-Powered Resume Search</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2563eb; margin-bottom: 20px; }
            .feature { background: #f8fafc; padding: 20px; margin: 15px 0; border-radius: 8px; border-left: 4px solid #2563eb; }
            .api-link { background: #1f2937; color: white; padding: 15px; border-radius: 8px; margin: 20px 0; }
            .api-link a { color: #60a5fa; text-decoration: none; }
            .status { background: #dcfce7; color: #166534; padding: 10px; border-radius: 6px; margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 ResumeRAG - AI-Powered Resume Search</h1>
            
            <div class="status">
                ✅ <strong>Server is running successfully!</strong>
            </div>
            
            <div class="feature">
                <h3>🔍 Smart Resume Search</h3>
                <p>Upload resumes and search them using natural language queries powered by AI semantic search.</p>
            </div>
            
            <div class="feature">
                <h3>🤖 Job Matching</h3>
                <p>Create job postings and find the best matching candidates with evidence and missing requirements analysis.</p>
            </div>
            
            <div class="feature">
                <h3>🔒 Privacy Protection</h3>
                <p>Automatic PII redaction for non-recruiter users with role-based access control.</p>
            </div>
            
            <div class="feature">
                <h3>📁 Multi-format Support</h3>
                <p>Upload PDF, DOCX, DOC, TXT files, or ZIP archives containing multiple resumes.</p>
            </div>
            
            <div class="api-link">
                <h3>🔗 API Endpoints</h3>
                <p><strong>Interactive API Documentation:</strong> <a href="/docs" target="_blank">http://localhost:8000/docs</a></p>
                <p><strong>API Health Check:</strong> <a href="/health" target="_blank">http://localhost:8000/health</a></p>
            </div>
            
            <div style="margin-top: 30px; padding: 20px; background: #fef3c7; border-radius: 8px; border-left: 4px solid #f59e0b;">
                <h3>🎯 Demo Instructions</h3>
                <ol>
                    <li>Visit <a href="/docs" style="color: #92400e;">/docs</a> for interactive API testing</li>
                    <li>Register a new account using <code>POST /api/auth/register</code></li>
                    <li>Upload resumes using <code>POST /api/resumes</code></li>
                    <li>Search resumes using <code>POST /api/ask</code></li>
                    <li>Create jobs and match candidates using job endpoints</li>
                </ol>
            </div>
        </div>
    </body>
    </html>
    """

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "ResumeRAG API",
        "version": "1.0.0",
        "message": "Server is running successfully!"
    }

@app.get("/api/status")
async def api_status():
    return {
        "api": "ResumeRAG",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "auth": "/api/auth/register, /api/auth/login",
            "resumes": "/api/resumes, /api/ask",
            "jobs": "/api/jobs, /api/jobs/{id}/match"
        }
    }

def main():
    print("🚀 Starting ResumeRAG Server...")
    print("📡 Server will be available at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("💡 Health Check: http://localhost:8000/health")
    print("\n" + "="*50)
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main()
