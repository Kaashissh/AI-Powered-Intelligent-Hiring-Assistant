# api.py - FastAPI Backend Service

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import json
import io
import PyPDF2
import docx
from resume_matcher import ResumeJobMatcher
from chatbot import ResumeChatbot

# Initialize FastAPI app
app = FastAPI(
    title="AI Resume Matcher API",
    description="AI-powered resume matching service using Hugging Face models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global matcher instance
matcher = None

# Global chatbot instance
chatbot = None

# Simple in-memory store: {session_id: last analysis result}.
# Fine for a demo/single-instance deployment; swap for a real store
# (Redis, DB, etc.) for a production, multi-worker deployment.
_last_results = {}

# Pydantic models for request/response
class ResumeAnalysisRequest(BaseModel):
    resume_text: str
    job_description: str
    company_name: Optional[str] = None
    job_title: Optional[str] = None
    session_id: Optional[str] = "default"

class ResumeAnalysisResponse(BaseModel):
    match_category: str
    overall_score: float
    detailed_scores: dict
    matched_skills: List[str]
    missing_skills: List[str]
    recommendations: List[str]
    success: bool = True
    message: str = "Analysis completed successfully"

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    reply: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str

# Utility functions
def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    """Extract text from PDF bytes"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")

def extract_text_from_docx_bytes(file_bytes: bytes) -> str:
    """Extract text from DOCX bytes"""
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading DOCX: {str(e)}")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the matcher on startup"""
    global matcher, chatbot
    try:
        matcher = ResumeJobMatcher()
        matcher.download_nltk_data()
        matcher.load_models()
        
        # Try to load pre-trained model
        try:
            matcher.load_model('resume_matcher_model.pkl')
            print("✅ Pre-trained model loaded successfully")
        except FileNotFoundError:
            print("⚠️ Pre-trained model not found, using fresh model")

        chatbot = ResumeChatbot()
        print("💬 Chatbot assistant initialized")

        print("🚀 AI Resume Matcher API started successfully")
        
    except Exception as e:
        print(f"❌ Error initializing matcher: {e}")
        raise e

# API Endpoints

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return HealthResponse(
        status="running",
        model_loaded=matcher is not None,
        version="1.0.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if matcher is not None else "unhealthy",
        model_loaded=matcher is not None,
        version="1.0.0"
    )

@app.post("/analyze", response_model=ResumeAnalysisResponse)
async def analyze_resume(request: ResumeAnalysisRequest):
    """Analyze resume against job description"""
    
    if matcher is None:
        raise HTTPException(status_code=503, detail="AI model not loaded")
    
    if not request.resume_text.strip():
        raise HTTPException(status_code=400, detail="Resume text cannot be empty")
    
    if not request.job_description.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty")
    
    try:
        # Perform analysis
        result = matcher.analyze_resume(
            request.resume_text, 
            request.job_description
        )
        
        _last_results[request.session_id or "default"] = result

        return ResumeAnalysisResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/upload", response_model=ResumeAnalysisResponse)
async def analyze_resume_upload(
    job_description: str,
    resume_file: UploadFile = File(...),
    company_name: Optional[str] = None,
    job_title: Optional[str] = None
):
    """Analyze uploaded resume file against job description"""
    
    if matcher is None:
        raise HTTPException(status_code=503, detail="AI model not loaded")
    
    if not job_description.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty")
    
    # Validate file type
    allowed_types = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain']
    if resume_file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail="Unsupported file type. Please upload PDF, DOCX, or TXT files."
        )
    
    try:
        # Read file content
        file_content = await resume_file.read()
        
        # Extract text based on file type
        if resume_file.content_type == 'application/pdf':
            resume_text = extract_text_from_pdf_bytes(file_content)
        elif resume_file.content_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            resume_text = extract_text_from_docx_bytes(file_content)
        else:  # text/plain
            resume_text = file_content.decode('utf-8')
        
        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the uploaded file")
        
        # Perform analysis
        result = matcher.analyze_resume(resume_text, job_description)
        
        return ResumeAnalysisResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

@app.post("/batch-analyze")
async def batch_analyze_resumes(
    job_description: str,
    resume_files: List[UploadFile] = File(...)
):
    """Analyze multiple resumes against a single job description"""
    
    if matcher is None:
        raise HTTPException(status_code=503, detail="AI model not loaded")
    
    if not job_description.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty")
    
    if len(resume_files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")
    
    results = []
    
    for i, resume_file in enumerate(resume_files):
        try:
            # Read and extract text
            file_content = await resume_file.read()
            
            if resume_file.content_type == 'application/pdf':
                resume_text = extract_text_from_pdf_bytes(file_content)
            elif resume_file.content_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                resume_text = extract_text_from_docx_bytes(file_content)
            else:
                resume_text = file_content.decode('utf-8')
            
            # Analyze
            result = matcher.analyze_resume(resume_text, job_description)
            result['filename'] = resume_file.filename
            result['file_index'] = i
            
            results.append(result)
            
        except Exception as e:
            # Add error result for failed files
            results.append({
                'filename': resume_file.filename,
                'file_index': i,
                'success': False,
                'error': str(e),
                'overall_score': 0.0,
                'match_category': 'Error'
            })
    
    # Sort results by score (descending)
    results.sort(key=lambda x: x.get('overall_score', 0), reverse=True)
    
    return {
        'total_files': len(resume_files),
        'successful_analyses': len([r for r in results if r.get('success', True)]),
        'results': results
    }

@app.get("/skills")
async def get_skills_database():
    """Get the list of skills in the database"""
    if matcher is None:
        raise HTTPException(status_code=503, detail="AI model not loaded")
    
    return {
        'total_skills': len(matcher.skills_keywords),
        'skills': sorted(matcher.skills_keywords)
    }

@app.post("/extract-skills")
async def extract_skills_from_text(text: str):
    """Extract skills from given text"""
    if matcher is None:
        raise HTTPException(status_code=503, detail="AI model not loaded")
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        skills = matcher.extract_skills(text)
        return {
            'found_skills': skills,
            'skill_count': len(skills),
            'text_length': len(text)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Skill extraction failed: {str(e)}")

@app.post("/similarity")
async def calculate_similarity(text1: str, text2: str):
    """Calculate similarity between two texts"""
    if matcher is None:
        raise HTTPException(status_code=503, detail="AI model not loaded")
    
    if not text1.strip() or not text2.strip():
        raise HTTPException(status_code=400, detail="Both texts must be non-empty")
    
    try:
        scores = matcher.calculate_similarity_scores(text1, text2)
        return scores
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similarity calculation failed: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Conversational assistant: ask questions about a previous /analyze
    result, grounded via the session_id used for that analysis."""

    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")

    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    context = _last_results.get(request.session_id or "default")
    reply = chatbot.respond(request.message, context=context)
    return ChatResponse(reply=reply)

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "status_code": 404}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "status_code": 500}

# Run server
if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )