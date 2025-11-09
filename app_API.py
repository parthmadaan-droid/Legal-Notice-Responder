from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
import uuid
import os
import tempfile
import shutil
import json
import logging
import base64
import time
from datetime import datetime
import re
from enum import Enum

# Document handling imports
from PIL import Image
import pytesseract
import io

# PDF handling - with fallbacks
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

# OpenAI integration
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('legal_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("OpenAI API key not found in environment variables")
else:
    try:
        # Try new version
        from openai import OpenAI
        client = OpenAI(api_key=openai_api_key, timeout=30.0)
        OPENAI_VERSION = "new"
        logger.info("Using OpenAI v1.0+ client")
    except Exception as e:
        logger.warning(f"New OpenAI client failed: {e}")
        # Fall back to legacy client
        import openai
        openai.api_key = openai_api_key
        client = openai
        OPENAI_VERSION = "old"
        logger.info("Using OpenAI legacy client")

# Constants
TEMP_DIR = tempfile.mkdtemp()
logger.info(f"Created temporary directory: {TEMP_DIR}")

# Model constants
VISION_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4-vision-preview"]
ANALYSIS_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]

# In-memory storage (in production, use a database)
DOCUMENT_STORE = {}  # document_id -> document data
ANALYSIS_STORE = {}  # document_id -> analysis data
NOTICE_STORE = {}    # document_id -> legal notice data

# API app definition
app = FastAPI(
    title="Legal Document Analyzer API",
    description="AI-Powered Legal Document Analysis and Notice Generation API",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class UploadResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    status: str
    message: str
    char_count: int = 0

class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    file_type: str
    char_count: int
    extraction_method: str
    status: str
    uploaded_at: str

class PriorityLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class AnalysisResult(BaseModel):
    document_id: str
    is_legal_document: bool
    document_confidence: str
    document_type: str
    summary: str
    key_parties: List[str]
    key_dates: List[str]
    amounts: List[str]
    deadlines: List[str]
    claims: List[str]
    priority: PriorityLevel
    priority_reasoning: str
    urgency_indicators: List[str]
    recommended_actions: List[str]
    risks: List[str]
    non_legal_reason: Optional[str] = None

class LegalNoticeRequest(BaseModel):
    document_id: str
    model: Optional[str] = None

class LegalNoticeResponse(BaseModel):
    document_id: str
    notice_id: str
    notice_text: str
    model_used: str
    word_count: int
    created_at: str

class NoticeUpdateRequest(BaseModel):
    notice_id: str
    instruction: str
    model: Optional[str] = None

class NoticeUpdateResponse(BaseModel):
    notice_id: str
    updated_notice: str
    model_used: str
    word_count: int
    updated_at: str
    changes_made: bool

class HealthCheck(BaseModel):
    status: str
    openai_version: str
    pdf_libraries: List[str]
    timestamp: str

# Helper Functions
def encode_image_to_base64(image_data):
    """Convert image data to base64"""
    try:
        return base64.b64encode(image_data).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image to base64: {str(e)}")
        return None

async def extract_text_from_image_with_openai(base64_image, image_description="uploaded image"):
    """Extract text from image using OpenAI Vision API with fallback models"""
    logger.info(f"Starting OpenAI vision extraction for {image_description}")
    
    for model in VISION_MODELS:
        try:
            logger.info(f"Trying model: {model}")
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Please extract ALL text from this image. This appears to be a legal document. 
                            Maintain the original formatting as much as possible, including:
                            - Line breaks and spacing
                            - Headers and sections
                            - Dates, amounts, and addresses
                            - Signatures and letterheads
                            
                            If you see any handwritten text, please include it as well.
                            Provide the complete text content without any commentary."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
            
            if OPENAI_VERSION == "new":
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=4000,
                    temperature=0.1
                )
                extracted_text = response.choices[0].message.content.strip()
            else:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    max_tokens=4000,
                    temperature=0.1
                )
                extracted_text = response.choices[0].message.content.strip()
            
            logger.info(f"✅ {model} vision extraction successful: {len(extracted_text)} characters")
            return extracted_text, model
            
        except Exception as e:
            logger.warning(f"❌ {model} failed: {str(e)}")
            continue
    
    logger.error("All vision models failed")
    return "", None

async def extract_text_from_pdf(file_path):
    """Extract text from PDF using multiple methods"""
    logger.info(f"Starting PDF text extraction from: {file_path}")
    
    # Method 1: Try PyMuPDF first
    if PYMUPDF_AVAILABLE:
        try:
            pdf_document = fitz.open(file_path)
            text_content = ""
            images_for_ocr = []
            
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                
                # Extract text
                page_text = page.get_text()
                if page_text.strip():
                    text_content += f"\n--- Page {page_num + 1} ---\n{page_text}"
                    logger.info(f"Extracted text from PDF page {page_num + 1}: {len(page_text)} characters")
                
                # If no text found, get page as image for OCR
                if not page_text.strip():
                    pix = page.get_pixmap()
                    img_path = os.path.join(TEMP_DIR, f"page_{page_num}.png")
                    pix.save(img_path)
                    with open(img_path, "rb") as img_file:
                        img_data = img_file.read()
                    images_for_ocr.append((page_num + 1, img_data))
                    os.remove(img_path)  # Clean up temp file
            
            pdf_document.close()
            
            # If we got text content, return it
            if text_content.strip():
                logger.info(f"Successfully extracted text from PDF: {len(text_content)} characters")
                return text_content, "PyMuPDF"
            
            # If no text, try OCR on images using OpenAI
            if images_for_ocr:
                logger.info(f"No text found in PDF, trying OCR on {len(images_for_ocr)} pages")
                ocr_text = ""
                for page_num, img_data in images_for_ocr[:3]:  # Limit to first 3 pages for API costs
                    base64_image = base64.b64encode(img_data).decode('utf-8')
                    page_text, model = await extract_text_from_image_with_openai(base64_image, f"PDF page {page_num}")
                    if page_text:
                        ocr_text += f"\n--- Page {page_num} (OCR) ---\n{page_text}"
                
                if ocr_text.strip():
                    return ocr_text, f"PyMuPDF + OpenAI Vision ({model})"
        
        except Exception as e:
            logger.error(f"Error with PyMuPDF extraction: {str(e)}")
    
    # Method 2: Fallback to PyPDF2
    if PYPDF2_AVAILABLE:
        try:
            pdf_reader = PyPDF2.PdfReader(file_path)
            text = ""
            for i, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text.strip():
                    text += f"\n--- Page {i + 1} ---\n{page_text}"
            
            if text.strip():
                logger.info(f"PyPDF2 extraction successful: {len(text)} characters")
                return text, "PyPDF2"
                
        except Exception as e:
            logger.error(f"Error with PyPDF2 extraction: {str(e)}")
    
    logger.error("All PDF extraction methods failed")
    return "", "Failed"

async def extract_text_from_image_file(file_path):
    """Extract text from image file using OpenAI Vision"""
    logger.info(f"Processing image file: {file_path}")
    
    try:
        # Read image and convert to base64
        with open(file_path, "rb") as img_file:
            img_data = img_file.read()
        
        base64_image = base64.b64encode(img_data).decode('utf-8')
        
        # Use OpenAI Vision
        extracted_text, model = await extract_text_from_image_with_openai(base64_image, os.path.basename(file_path))
        
        if extracted_text.strip():
            logger.info(f"✅ OpenAI vision successful with {model}: {len(extracted_text)} characters")
            return extracted_text, f"OpenAI Vision ({model})"
        
        # Fallback to traditional OCR if OpenAI fails
        logger.info("OpenAI vision failed, trying traditional OCR")
        try:
            image = Image.open(file_path)
            extracted_text = pytesseract.image_to_string(image)
            logger.info(f"Traditional OCR result: {len(extracted_text)} characters")
            return extracted_text, "Tesseract OCR"
        except Exception as ocr_e:
            logger.error(f"Traditional OCR also failed: {str(ocr_e)}")
        
        return "", "Failed"
        
    except Exception as e:
        logger.error(f"Error processing image file: {str(e)}")
        return "", "Failed"

async def analyze_document_with_openai(document_text, document_id):
    """Analyze document using OpenAI"""
    logger.info(f"Starting document analysis for {document_id}, text length: {len(document_text)} characters")
    
    # Try to use the model with best accuracy first, then fall back to others
    for model in ANALYSIS_MODELS:
        try:
            logger.info(f"Trying model: {model} for analysis")
            
            prompt = f"""
            Analyze the following document and provide a structured analysis in JSON format.

            Document Text:
            {document_text[:4000]}... (truncated for length)

            First, determine if this is a legal document (contract, notice, legal letter, court document, etc.) or a non-legal document (essay, article, personal letter, etc.).

            Please provide the analysis in the following JSON structure:
            {{
                "is_legal_document": true/false,
                "document_confidence": "High/Medium/Low confidence in document type identification",
                "summary": "Brief summary of the document",
                "document_type": "Type of document (legal or non-legal)",
                "key_parties": ["Party 1", "Party 2"],
                "key_dates": ["Date 1: Description", "Date 2: Description"],
                "amounts": ["Amount 1: Description", "Amount 2: Description"],
                "deadlines": ["Deadline 1: Description"],
                "claims": ["Claim 1", "Claim 2"],
                "priority": "High/Medium/Low",
                "priority_reasoning": "Explanation for priority level",
                "urgency_indicators": ["Indicator 1", "Indicator 2"],
                "recommended_actions": ["Action 1", "Action 2"],
                "risks": ["Risk 1", "Risk 2"],
                "non_legal_reason": "If not a legal document, explain what type of document this appears to be"
            }}

            If this is NOT a legal document, still fill out the fields but focus on explaining what type of document it actually is.
            Provide only the JSON response, no additional text.
            """
            
            if OPENAI_VERSION == "new":
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a legal document analysis expert. Analyze documents thoroughly and provide structured insights."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
                result = response.choices[0].message.content.strip()
            else:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a legal document analysis expert. Analyze documents thoroughly and provide structured insights."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
                result = response.choices[0].message.content.strip()
            
            # Clean up the response to ensure it's valid JSON
            if result.startswith('```json'):
                result = result[7:]
                if result.endswith('```'):
                    result = result[:-3]
            elif result.startswith('```'):
                result = result[3:]
                if result.endswith('```'):
                    result = result[:-3]
            
            # Parse JSON
            parsed_result = json.loads(result)
            logger.info(f"Document analysis completed successfully with {model}")
            
            # Add document_id and model used
            parsed_result["document_id"] = document_id
            parsed_result["model_used"] = model
            parsed_result["analysis_timestamp"] = datetime.now().isoformat()
            
            return parsed_result, model
            
        except json.JSONDecodeError as json_e:
            logger.error(f"JSON parsing error with {model}: {str(json_e)}")
            logger.error(f"Raw response from {model}: {result[:200]}...")
            continue
        except Exception as e:
            logger.error(f"Error with {model} for analysis: {str(e)}")
            continue
    
    # If all models fail
    logger.error(f"All models failed for document analysis: {document_id}")
    raise HTTPException(status_code=500, detail="Failed to analyze document with all available models")

async def generate_legal_notice(analysis_result, document_text, model=None):
    """Generate legal notice based on analysis with model selection"""
    logger.info("Starting legal notice generation")
    document_id = analysis_result.get("document_id", "unknown")
    
    models_to_try = [model] if model else ANALYSIS_MODELS
    
    prompt = f"""
    Based on the following legal document analysis, generate a professional legal notice/reply:

    Original Document Analysis:
    {json.dumps(analysis_result, indent=2)}

    Original Document Text (excerpt):
    {document_text[:2000]}...

    Generate a formal legal notice that:
    1. Addresses the claims professionally
    2. States the client's position clearly
    3. Includes relevant dates and references
    4. Uses appropriate legal language
    5. Is properly formatted as a legal document

    Generate only the legal notice content, properly formatted.
    """
    
    for model_name in models_to_try:
        try:
            logger.info(f"Sending request to {model_name} for legal notice generation")
            
            if OPENAI_VERSION == "new":
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a legal expert specializing in drafting professional legal notices and replies. Create formal, well-structured legal documents."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.4
                )
                result = response.choices[0].message.content.strip()
            else:
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a legal expert specializing in drafting professional legal notices and replies. Create formal, well-structured legal documents."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.4
                )
                result = response.choices[0].message.content.strip()
            
            logger.info(f"✅ Legal notice generated successfully with {model_name}, length: {len(result)} characters")
            return result, model_name
            
        except Exception as e:
            logger.error(f"Error with {model_name} for legal notice generation: {str(e)}")
            continue
    
    # If all models fail
    logger.error(f"All models failed for legal notice generation for document: {document_id}")
    raise HTTPException(status_code=500, detail="Failed to generate legal notice with all available models")

async def update_legal_notice(current_notice, user_instruction, analysis_context, model=None):
    """Update legal notice based on user instruction with model selection"""
    logger.info(f"Updating legal notice with instruction: {user_instruction}")
    
    models_to_try = [model] if model else ANALYSIS_MODELS
    
    prompt = f"""
    Current Legal Notice:
    {current_notice}

    Original Case Context:
    {json.dumps(analysis_context, indent=2)}

    User Instruction: {user_instruction}

    Please update the legal notice based on the user's instruction. 
    Maintain professional legal language and proper formatting.
    Return ONLY the complete updated legal notice, no additional commentary.
    """
    
    for model_name in models_to_try:
        try:
            logger.info(f"Sending update request to {model_name}")
            
            if OPENAI_VERSION == "new":
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a legal document editor. Update legal notices based on user instructions while maintaining professional standards."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
                result = response.choices[0].message.content.strip()
            else:
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a legal document editor. Update legal notices based on user instructions while maintaining professional standards."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
                result = response.choices[0].message.content.strip()
            
            logger.info(f"✅ Legal notice updated successfully with {model_name}")
            
            # Check if changes were made
            changes_made = result != current_notice
            
            return result, model_name, changes_made
            
        except Exception as e:
            logger.error(f"Error with {model_name} for notice update: {str(e)}")
            continue
    
    # If all models fail
    logger.error(f"All models failed for legal notice update")
    raise HTTPException(status_code=500, detail="Failed to update legal notice with all available models")

async def process_document(document_id: str, file_path: str, filename: str, file_type: str):
    """Background task to process uploaded document"""
    logger.info(f"Processing document {document_id}: {filename}")
    
    try:
        # Extract text based on file type
        extracted_text = ""
        extraction_method = "Unknown"
        
        if "pdf" in file_type.lower():
            extracted_text, extraction_method = await extract_text_from_pdf(file_path)
        elif file_type.startswith('image/'):
            extracted_text, extraction_method = await extract_text_from_image_file(file_path)
        elif "text/plain" in file_type.lower():
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                extracted_text = f.read()
                extraction_method = "Text File"
        
        # Update document store with extracted text
        if extracted_text and extracted_text.strip():
            DOCUMENT_STORE[document_id]["text"] = extracted_text
            DOCUMENT_STORE[document_id]["extraction_method"] = extraction_method
            DOCUMENT_STORE[document_id]["char_count"] = len(extracted_text)
            DOCUMENT_STORE[document_id]["status"] = "processed"
            DOCUMENT_STORE[document_id]["processed_at"] = datetime.now().isoformat()
            
            logger.info(f"Document {document_id} processed successfully: {len(extracted_text)} characters with {extraction_method}")
        else:
            DOCUMENT_STORE[document_id]["status"] = "failed"
            DOCUMENT_STORE[document_id]["error"] = "Could not extract text from document"
            logger.error(f"Failed to extract text from document {document_id}")
        
        # Clean up temp file
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Removed temporary file: {file_path}")
            
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {str(e)}")
        DOCUMENT_STORE[document_id]["status"] = "failed"
        DOCUMENT_STORE[document_id]["error"] = str(e)
        
        # Clean up temp file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as file_e:
                logger.error(f"Error removing temp file: {str(file_e)}")

# API Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Legal Document Analyzer API",
        "version": "1.0.0",
        "description": "AI-Powered Legal Document Analysis and Notice Generation API",
        "endpoints": {
            "/upload": "Upload a document for processing",
            "/documents/{document_id}": "Get document info",
            "/analyze/{document_id}": "Analyze a document",
            "/generate-notice": "Generate a legal notice",
            "/update-notice": "Update a legal notice with specific instructions"
        }
    }

@app.get("/health", response_model=HealthCheck, tags=["General"])
async def health_check():
    """Health check endpoint"""
    pdf_libs = []
    if PYMUPDF_AVAILABLE:
        pdf_libs.append("PyMuPDF")
    if PYPDF2_AVAILABLE:
        pdf_libs.append("PyPDF2")
    
    return {
        "status": "ok",
        "openai_version": OPENAI_VERSION,
        "pdf_libraries": pdf_libs,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload a document for processing
    
    Accepts PDF, image, or text files and extracts text for further analysis.
    The extraction process runs in the background.
    """
    try:
        # Validate file type
        file_type = file.content_type
        supported_types = ["application/pdf", "image/jpeg", "image/png", "image/jpg", "text/plain"]
        
        if not any(supported in file_type.lower() for supported in supported_types):
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_type}. Supported types: PDF, JPG, PNG, TXT"
            )
        
        # Create unique document ID
        document_id = str(uuid.uuid4())
        filename = file.filename
        
        # Save file to temp directory
        file_path = os.path.join(TEMP_DIR, f"{document_id}_{filename}")
        
        with open(file_path, "wb") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
        
        logger.info(f"File saved to {file_path}")
        
        # Store document info
        DOCUMENT_STORE[document_id] = {
            "document_id": document_id,
            "filename": filename,
            "file_type": file_type,
            "file_path": file_path,
            "status": "processing",
            "uploaded_at": datetime.now().isoformat()
        }
        
        # Process document in background
        background_tasks.add_task(process_document, document_id, file_path, filename, file_type)
        
        return {
            "document_id": document_id,
            "filename": filename,
            "file_type": file_type,
            "status": "processing",
            "message": "Document uploaded and processing started",
            "char_count": 0
        }
        
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

@app.get("/documents/{document_id}", response_model=DocumentInfo, tags=["Documents"])
async def get_document_info(document_id: str):
    """
    Get information about an uploaded document
    
    Returns status, extraction method, and character count for the document.
    """
    if document_id not in DOCUMENT_STORE:
        raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")
    
    doc_info = DOCUMENT_STORE[document_id]
    
    return {
        "document_id": document_id,
        "filename": doc_info.get("filename", "Unknown"),
        "file_type": doc_info.get("file_type", "Unknown"),
        "char_count": doc_info.get("char_count", 0),
        "extraction_method": doc_info.get("extraction_method", "Unknown"),
        "status": doc_info.get("status", "Unknown"),
        "uploaded_at": doc_info.get("uploaded_at", datetime.now().isoformat())
    }

@app.get("/documents", tags=["Documents"])
async def list_documents():
    """List all uploaded documents"""
    documents = []
    
    for doc_id, doc_info in DOCUMENT_STORE.items():
        documents.append({
            "document_id": doc_id,
            "filename": doc_info.get("filename", "Unknown"),
            "status": doc_info.get("status", "Unknown"),
            "char_count": doc_info.get("char_count", 0),
            "uploaded_at": doc_info.get("uploaded_at", "Unknown")
        })
    
    return {"documents": documents, "count": len(documents)}

@app.post("/analyze/{document_id}", response_model=AnalysisResult, tags=["Analysis"])
async def analyze_document(document_id: str, model: Optional[str] = Query(None, description="Model to use for analysis")):
    """
    Analyze a document to extract key information
    
    Processes the document to identify if it's a legal document, extract parties,
    dates, amounts, and assess priority level.
    """
    if document_id not in DOCUMENT_STORE:
        raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")
    
    doc_info = DOCUMENT_STORE[document_id]
    
    # Check if document has been processed
    if doc_info.get("status") != "processed":
        raise HTTPException(
            status_code=400, 
            detail=f"Document not ready for analysis. Current status: {doc_info.get('status', 'Unknown')}"
        )
    
    # Check if we have extracted text
    if "text" not in doc_info or not doc_info["text"].strip():
        raise HTTPException(status_code=400, detail="No text content found in document")
    
    # Check if already analyzed
    if document_id in ANALYSIS_STORE:
        logger.info(f"Using cached analysis for document {document_id}")
        return ANALYSIS_STORE[document_id]
    
    # Analyze document
    try:
        document_text = doc_info["text"]
        
        # If specific model requested, validate it
        if model and model not in ANALYSIS_MODELS:
            logger.warning(f"Requested model {model} not in allowed list, falling back to defaults")
            model = None
        
        analysis, model_used = await analyze_document_with_openai(document_text, document_id)
        
        # Store analysis result
        ANALYSIS_STORE[document_id] = analysis
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing document: {str(e)}")

@app.post("/generate-notice", response_model=LegalNoticeResponse, tags=["Legal Notice"])
async def generate_notice(request: LegalNoticeRequest):
    """
    Generate a legal notice based on document analysis
    
    Creates a professional legal response based on the analyzed document.
    """
    document_id = request.document_id
    
    # Check if document exists and has been analyzed
    if document_id not in DOCUMENT_STORE:
        raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")
    
    if document_id not in ANALYSIS_STORE:
        raise HTTPException(status_code=400, detail="Document has not been analyzed yet. Please analyze first.")
    
    # Get document text and analysis
    doc_info = DOCUMENT_STORE[document_id]
    document_text = doc_info["text"]
    analysis = ANALYSIS_STORE[document_id]
    
    # Check if it's a legal document
    if not analysis.get("is_legal_document", True):
        raise HTTPException(
            status_code=400, 
            detail="This is not a legal document. Legal notice generation is only available for legal documents."
        )
    
    # If specific model requested, validate it
    model = request.model
    if model and model not in ANALYSIS_MODELS:
        logger.warning(f"Requested model {model} not in allowed list, falling back to defaults")
        model = None
    
    # Generate legal notice
    try:
        notice_text, model_used = await generate_legal_notice(analysis, document_text, model)
        
        # Create notice ID
        notice_id = str(uuid.uuid4())
        
        # Store notice
        NOTICE_STORE[notice_id] = {
            "notice_id": notice_id,
            "document_id": document_id,
            "notice_text": notice_text,
            "model_used": model_used,
            "word_count": len(notice_text.split()),
            "created_at": datetime.now().isoformat(),
            "update_history": []  # For tracking changes
        }
        
        return {
            "document_id": document_id,
            "notice_id": notice_id,
            "notice_text": notice_text,
            "model_used": model_used,
            "word_count": len(notice_text.split()),
            "created_at": NOTICE_STORE[notice_id]["created_at"]
        }
        
    except Exception as e:
        logger.error(f"Error generating legal notice for document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating legal notice: {str(e)}")

@app.post("/update-notice", response_model=NoticeUpdateResponse, tags=["Legal Notice"])
async def update_notice(request: NoticeUpdateRequest):
    """
    Update a legal notice based on specific instructions
    
    Modifies the generated legal notice according to user instructions.
    """
    notice_id = request.notice_id
    instruction = request.instruction
    
    # Check if notice exists
    if notice_id not in NOTICE_STORE:
        raise HTTPException(status_code=404, detail=f"Notice not found: {notice_id}")
    
    # Get current notice and related document/analysis
    notice_info = NOTICE_STORE[notice_id]
    document_id = notice_info["document_id"]
    
    if document_id not in ANALYSIS_STORE:
        raise HTTPException(status_code=400, detail="Associated analysis not found")
    
    analysis = ANALYSIS_STORE[document_id]
    current_notice = notice_info["notice_text"]
    
    # If specific model requested, validate it
    model = request.model
    if model and model not in ANALYSIS_MODELS:
        logger.warning(f"Requested model {model} not in allowed list, falling back to defaults")
        model = None
    
    # Update notice
    try:
        updated_notice, model_used, changes_made = await update_legal_notice(
            current_notice,
            instruction,
            analysis,
            model
        )
        
        # Update notice store
        if changes_made:
            # Track update history
            update_entry = {
                "timestamp": datetime.now().isoformat(),
                "instruction": instruction,
                "model_used": model_used,
            }
            notice_info["update_history"].append(update_entry)
            notice_info["notice_text"] = updated_notice
            notice_info["updated_at"] = datetime.now().isoformat()
            notice_info["word_count"] = len(updated_notice.split())
        
        return {
            "notice_id": notice_id,
            "updated_notice": updated_notice,
            "model_used": model_used,
            "word_count": len(updated_notice.split()),
            "updated_at": datetime.now().isoformat(),
            "changes_made": changes_made
        }
        
    except Exception as e:
        logger.error(f"Error updating legal notice {notice_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating legal notice: {str(e)}")

@app.get("/notices/{notice_id}", tags=["Legal Notice"])
async def get_notice(notice_id: str):
    """Get a legal notice by ID"""
    if notice_id not in NOTICE_STORE:
        raise HTTPException(status_code=404, detail=f"Notice not found: {notice_id}")
    
    notice_info = NOTICE_STORE[notice_id]
    
    return {
        "notice_id": notice_id,
        "document_id": notice_info["document_id"],
        "notice_text": notice_info["notice_text"],
        "word_count": notice_info["word_count"],
        "created_at": notice_info["created_at"],
        "updated_at": notice_info.get("updated_at", notice_info["created_at"]),
        "update_count": len(notice_info.get("update_history", []))
    }

@app.get("/notices", tags=["Legal Notice"])
async def list_notices():
    """List all generated legal notices"""
    notices = []
    
    for notice_id, notice_info in NOTICE_STORE.items():
        notices.append({
            "notice_id": notice_id,
            "document_id": notice_info["document_id"],
            "word_count": notice_info["word_count"],
            "created_at": notice_info["created_at"],
            "updated_at": notice_info.get("updated_at", notice_info["created_at"]),
            "update_count": len(notice_info.get("update_history", []))
        })
    
    return {"notices": notices, "count": len(notices)}

# Cleanup on shutdown
@app.on_event("shutdown")
def cleanup():
    """Clean up temporary files on shutdown"""
    try:
        shutil.rmtree(TEMP_DIR)
        logger.info(f"Removed temporary directory: {TEMP_DIR}")
    except Exception as e:
        logger.error(f"Error cleaning up temp directory: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)