import streamlit as st
import openai
import os
from dotenv import load_dotenv
import PyPDF2
import io
from PIL import Image
import pytesseract
import json
from datetime import datetime, timedelta
import re
import logging
import base64
import fitz  # PyMuPDF for better PDF handling

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('legal_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure OpenAI - Handle both old and new versions
# try:
#     # Try new version (v1.0+) first
#     from openai import OpenAI
#     client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
#     OPENAI_VERSION = "new"
#     logger.info("Using OpenAI v1.0+ client")
# except ImportError:
#     # Fall back to old version
#     import openai
#     openai.api_key = os.getenv('OPENAI_API_KEY')
#     OPENAI_VERSION = "old" 
#     logger.info("Using OpenAI legacy client")
try:
    # Try new version with minimal parameters to avoid proxy issues
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        timeout=30.0  # Only essential parameters
    )
    OPENAI_VERSION = "new"
    logger.info("Using OpenAI v1.0+ client")
except Exception as e:
    logger.warning(f"New OpenAI client failed: {e}")
    try:
        # Fall back to legacy client
        import openai
        openai.api_key = os.getenv('OPENAI_API_KEY')
        client = openai
        OPENAI_VERSION = "old" 
        logger.info("Using OpenAI legacy client")
    except Exception as e2:
        logger.error(f"All OpenAI initialization failed: {e2}")
        st.error("❌ Could not initialize OpenAI client. Please check your API key and try again.")
        st.stop()

# Model constants for backward compatibility
VISION_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4-vision-preview"]
ANALYSIS_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]

# Verify OpenAI API key
if not os.getenv('OPENAI_API_KEY'):
    st.error("❌ OpenAI API key not found! Please add OPENAI_API_KEY to your .env file.")
    st.stop()

# Test OpenAI connection
def test_openai_connection():
    """Test OpenAI API connection"""
    try:
        if OPENAI_VERSION == "new":
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
        else:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
        return True, "Connection successful"
    except Exception as e:
        return False, str(e)

logger.info("Application started - OpenAI API key configured")

# Page configuration
st.set_page_config(
    page_title="Legal Document Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 10px 10px;
    }
    
    .priority-high {
        background-color: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .priority-high h3 {
        color: #dc2626 !important;
        margin-top: 0;
    }
    
    .priority-high p {
        color: #374151 !important;
        margin-bottom: 0;
    }
    
    .priority-medium {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;  
        margin: 1rem 0;
    }
    
    .priority-medium h3 {
        color: #d97706 !important;
        margin-top: 0;
    }
    
    .priority-medium p {
        color: #374151 !important;
        margin-bottom: 0;
    }
    
    .priority-low {
        background-color: #f0fdf4;
        border-left: 4px solid #22c55e;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .priority-low h3 {
        color: #16a34a !important;
        margin-top: 0;
    }
    
    .priority-low p {
        color: #374151 !important;
        margin-bottom: 0;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
    }
    
    .chat-message strong {
        color: #374151 !important;
    }
    
    .legal-notice {
        background-color: #ffffff;
        border: 2px solid #e5e7eb;
        border-radius: 8px;
        padding: 2rem;
        font-family: 'Times New Roman', serif;
        line-height: 1.6;
        min-height: 600px;
        color: #000000;
    }
    
    .legal-notice pre {
        white-space: pre-wrap;
        font-family: 'Times New Roman', serif;
        color: #000000 !important;
        font-size: 14px;
        margin: 0;
        padding: 0;
    }
    
    .action-button {
        background: linear-gradient(90deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        cursor: pointer;
        margin: 0.5rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'stage' not in st.session_state:
    st.session_state.stage = 'upload'
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'legal_notice' not in st.session_state:
    st.session_state.legal_notice = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'document_text' not in st.session_state:
    st.session_state.document_text = ""

def encode_image_to_base64(image_file):
    """Convert image to base64 for OpenAI API"""
    try:
        image_file.seek(0)
        image_data = image_file.read()
        return base64.b64encode(image_data).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image to base64: {str(e)}")
        return None

def extract_text_from_pdf_with_openai(pdf_file):
    """Extract text from PDF using multiple methods"""
    logger.info("Starting PDF text extraction")
    
    # Method 1: Try PyMuPDF first (better than PyPDF2)
    try:
        pdf_file.seek(0)
        pdf_data = pdf_file.read()
        pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
        
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
                img_data = pix.pil_tobytes(format="PNG")
                images_for_ocr.append((page_num + 1, img_data))
        
        pdf_document.close()
        
        # If we got text content, return it
        if text_content.strip():
            logger.info(f"Successfully extracted text from PDF: {len(text_content)} characters")
            return text_content
        
        # If no text, try OCR on images using OpenAI
        if images_for_ocr:
            logger.info(f"No text found in PDF, trying OCR on {len(images_for_ocr)} pages")
            ocr_text = ""
            for page_num, img_data in images_for_ocr[:3]:  # Limit to first 3 pages for API costs
                base64_image = base64.b64encode(img_data).decode('utf-8')
                page_text = extract_text_from_image_with_openai(base64_image, f"PDF page {page_num}")
                if page_text:
                    ocr_text += f"\n--- Page {page_num} (OCR) ---\n{page_text}"
            
            if ocr_text.strip():
                return ocr_text
        
    except Exception as e:
        logger.error(f"Error with PyMuPDF extraction: {str(e)}")
    
    # Method 2: Fallback to PyPDF2
    try:
        pdf_file.seek(0)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text.strip():
                text += f"\n--- Page {i + 1} ---\n{page_text}"
        
        if text.strip():
            logger.info(f"PyPDF2 extraction successful: {len(text)} characters")
            return text
            
    except Exception as e:
        logger.error(f"Error with PyPDF2 extraction: {str(e)}")
    
    logger.error("All PDF extraction methods failed")
    return ""

def extract_text_from_image_with_openai(base64_image, image_description="uploaded image"):
    """Extract text from image using OpenAI Vision API with fallback models"""
    logger.info(f"Starting OpenAI vision extraction for {image_description}")
    
    # Try models in order of preference
    models_to_try = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4-vision-preview"]
    
    for model in models_to_try:
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

def extract_text_from_image_file(image_file):
    """Extract text from image file using OpenAI Vision"""
    logger.info(f"Processing image file: {image_file.name}")
    
    try:
        # Convert to base64
        base64_image = encode_image_to_base64(image_file)
        if not base64_image:
            logger.error("Failed to encode image to base64")
            return ""
        
        # Use OpenAI Vision
        extracted_text, used_model = extract_text_from_image_with_openai(base64_image, image_file.name)
        
        if extracted_text.strip():
            logger.info(f"✅ OpenAI vision successful with {used_model}: {len(extracted_text)} characters")
            return extracted_text
        
        # Fallback to traditional OCR if OpenAI fails
        logger.info("OpenAI vision failed, trying traditional OCR")
        try:
            image_file.seek(0)
            image = Image.open(image_file)
            extracted_text = pytesseract.image_to_string(image)
            logger.info(f"Traditional OCR result: {len(extracted_text)} characters")
            return extracted_text
        except Exception as ocr_e:
            logger.error(f"Traditional OCR also failed: {str(ocr_e)}")
        
        return ""
        
    except Exception as e:
        logger.error(f"Error processing image file: {str(e)}")
        return ""

def analyze_document_with_openai(document_text):
    """Analyze document using OpenAI"""
    logger.info(f"Starting document analysis, text length: {len(document_text)} characters")
    
    prompt = f"""
    Analyze the following document and provide a structured analysis in JSON format.

    Document Text:
    {document_text}

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
    
    try:
        logger.info("Sending document to OpenAI for analysis")
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a legal document analysis expert. Analyze documents thoroughly and provide structured insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        result = response.choices[0].message.content.strip()
        logger.info("Received response from OpenAI, parsing JSON")
        
        # Clean up the response to ensure it's valid JSON
        if result.startswith('```json'):
            result = result[7:-3]
        elif result.startswith('```'):
            result = result[3:-3]
            
        parsed_result = json.loads(result)
        logger.info("Document analysis completed successfully")
        return parsed_result
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        logger.error(f"Raw response: {result}")
        st.error("Error parsing analysis results. Please try again.")
        return None
    except Exception as e:
        logger.error(f"Error analyzing document: {str(e)}")
        st.error(f"Error analyzing document: {str(e)}")
        return None

def generate_legal_notice(analysis_result, document_text, selected_model=None):
    """Generate legal notice based on analysis with model selection"""
    logger.info("Starting legal notice generation")
    
    models_to_try = [selected_model] if selected_model else ANALYSIS_MODELS
    
    prompt = f"""
    Based on the following legal document analysis, generate a professional legal notice/reply:

    Original Document Analysis:
    {json.dumps(analysis_result, indent=2)}

    Original Document Text:
    {document_text[:2000]}...

    Generate a formal legal notice that:
    1. Addresses the claims professionally
    2. States the client's position clearly
    3. Includes relevant dates and references
    4. Uses appropriate legal language
    5. Is properly formatted as a legal document

    Generate only the legal notice content, properly formatted.
    """
    
    for model in models_to_try:
        try:
            logger.info(f"Sending request to {model} for legal notice generation")
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a legal expert specializing in drafting professional legal notices and replies. Create formal, well-structured legal documents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4
            )
            
            result = response.choices[0].message.content.strip()
            logger.info(f"✅ Legal notice generated successfully with {model}, length: {len(result)} characters")
            return result
            
        except Exception as e:
            logger.error(f"Error with {model} for legal notice generation: {str(e)}")
            continue
    
    logger.error("All models failed for legal notice generation")
    st.error("❌ Failed to generate legal notice with all available models.")
    return ""

def update_legal_notice(current_notice, user_instruction, analysis_context, selected_model=None):
    """Update legal notice based on user instruction with model selection"""
    logger.info(f"Updating legal notice with instruction: {user_instruction}")
    
    models_to_try = [selected_model] if selected_model else ANALYSIS_MODELS
    
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
    
    for model in models_to_try:
        try:
            logger.info(f"Sending update request to {model}")
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a legal document editor. Update legal notices based on user instructions while maintaining professional standards."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            result = response.choices[0].message.content.strip()
            logger.info(f"✅ Legal notice updated successfully with {model}")
            return result
            
        except Exception as e:
            logger.error(f"Error with {model} for notice update: {str(e)}")
            continue
    
    logger.error("All models failed for notice update")
    st.error("❌ Failed to update legal notice with all available models.")
    return current_notice

# Main header
st.markdown("""
<div class="main-header">
    <h1>⚖️ Legal Document Analyzer</h1>
    <p>AI-Powered Legal Document Analysis & Notice Drafting</p>
</div>
""", unsafe_allow_html=True)

# Stage 1: Document Upload
if st.session_state.stage == 'upload':
    st.markdown("## 📄 Upload Your Legal Document")
    st.markdown("Upload your legal document for AI-powered analysis and priority assessment.")
    
    # Simple API connection test
    with st.expander("🔧 API Connection Test", expanded=False):
        st.info(f"**OpenAI Library Version:** {OPENAI_VERSION.title()} ({'v1.0+' if OPENAI_VERSION == 'new' else 'Legacy'})")
        
        if st.button("🧪 Test OpenAI Connection"):
            with st.spinner("Testing OpenAI API connection..."):
                success, message = test_openai_connection()
                if success:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ Connection failed: {message}")
                    st.markdown("""
                    **Troubleshooting:**
                    - Check your OpenAI API key in the .env file
                    - Ensure you have credits in your OpenAI account
                    - Try upgrading OpenAI library: `pip install openai>=1.3.0`
                    """)
        
        if OPENAI_VERSION == "old":
            st.warning("⚠️ You're using an older OpenAI library. Consider upgrading: `pip install openai>=1.3.0`")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'png', 'jpg', 'jpeg', 'txt'],
        help="Supported formats: PDF, Images (PNG, JPG, JPEG), Text files"
    )
    
    if uploaded_file is not None:
        st.info(f"📁 Processing: {uploaded_file.name} ({uploaded_file.size} bytes)")
        logger.info(f"File uploaded: {uploaded_file.name}, size: {uploaded_file.size}, type: {uploaded_file.type}")
        
        with st.spinner("🔍 Extracting text from document..."):
            document_text = ""
            
            try:
                # Extract text based on file type
                if uploaded_file.type == "application/pdf":
                    st.info("📄 Processing PDF document...")
                    document_text = extract_text_from_pdf_with_openai(uploaded_file)
                    
                elif uploaded_file.type.startswith('image/'):
                    st.info("🖼️ Processing image with AI vision...")
                    document_text = extract_text_from_image_file(uploaded_file)
                    
                elif uploaded_file.type == "text/plain":
                    st.info("📝 Processing text file...")
                    document_text = str(uploaded_file.read(), "utf-8")
                    logger.info(f"Text file processed: {len(document_text)} characters")
                    
                else:
                    st.error(f"❌ Unsupported file type: {uploaded_file.type}")
                    logger.error(f"Unsupported file type: {uploaded_file.type}")
                    st.stop()
                
                # Validate extracted text
                if document_text and document_text.strip():
                    st.session_state.document_text = document_text.strip()
                    
                    # Show preview of extracted text
                    with st.expander("📖 Preview of Extracted Text", expanded=False):
                        st.text_area(
                            "Extracted Content:",
                            value=document_text[:1000] + ("..." if len(document_text) > 1000 else ""),
                            height=200,
                            disabled=True
                        )
                    
                    st.success(f"✅ Text extracted successfully! ({len(document_text)} characters)")
                    logger.info(f"Text extraction successful: {len(document_text)} characters")
                    
                    if st.button("🔍 Analyze Document", type="primary"):
                        if len(document_text.strip()) < 50:
                            st.warning("⚠️ The extracted text seems very short. This might not be enough for a comprehensive analysis.")
                        
                        with st.spinner("🤖 Analyzing document with AI..."):
                            analysis_result = analyze_document_with_openai(document_text)
                            if analysis_result:
                                st.session_state.analysis_result = analysis_result
                                st.session_state.stage = 'analysis'
                                logger.info("Analysis completed, moving to analysis stage")
                                st.rerun()
                            else:
                                st.error("❌ Failed to analyze document. Please check the logs and try again.")
                                logger.error("Document analysis failed")
                
                else:
                    error_msg = "❌ Could not extract any text from the document."
                    
                    if uploaded_file.type == "application/pdf":
                        error_msg += "\n\n**Possible solutions:**\n"
                        error_msg += "- The PDF might be password protected\n"
                        error_msg += "- The PDF might contain only images (try converting to image first)\n"
                        error_msg += "- The PDF might be corrupted\n"
                        error_msg += "- Try a different PDF file\n"
                    elif uploaded_file.type.startswith('image/'):
                        error_msg += "\n\n**Possible solutions:**\n"
                        error_msg += "- Ensure the image is clear and text is readable\n"
                        error_msg += "- Try a higher resolution image\n"
                        error_msg += "- Ensure the image contains text content\n"
                        error_msg += "- Try a different image format\n"
                    
                    error_msg += "\n\n**Note:** The app uses multiple AI models automatically for best results."
                    
                    st.error(error_msg)
                    logger.error(f"No text extracted from {uploaded_file.name}")
                    
                    # Debug information
                    with st.expander("🔧 Debug Information"):
                        st.write(f"**File name:** {uploaded_file.name}")
                        st.write(f"**File type:** {uploaded_file.type}")
                        st.write(f"**File size:** {uploaded_file.size} bytes")
                        st.write(f"**Text length:** {len(document_text)} characters")
                        if document_text:
                            st.write(f"**Raw text preview:** `{repr(document_text[:200])}`")
                        
                        # Model information
                        st.markdown("**🤖 AI Models Used:**")
                        st.write("- Vision: gpt-4o → gpt-4o-mini → gpt-4-turbo → gpt-4-vision-preview (automatic fallback)")
                        st.write("- Analysis: gpt-4o-mini → gpt-4o → gpt-4-turbo → gpt-3.5-turbo (automatic fallback)")
                        st.write(f"- OpenAI Library: {OPENAI_VERSION.title()} version")
                        st.write("- Check logs for specific model performance")
            
            except Exception as e:
                st.error(f"❌ Error processing document: {str(e)}")
                logger.error(f"Document processing error: {str(e)}", exc_info=True)
                
                with st.expander("🔧 Error Details"):
                    st.code(str(e))
                    st.write("**Please try:**")
                    st.write("- A different file format")
                    st.write("- Checking your internet connection")
                    st.write("- Verifying your OpenAI API key")
    
    else:
        st.markdown("""
        ### 📋 Supported File Formats:
        - **PDF files** (.pdf) - Processed with AI-powered extraction
        - **Images** (.png, .jpg, .jpeg) - Uses AI vision for text extraction
        - **Text files** (.txt) - Direct text processing
        
        ### 💡 Tips for Best Results:
        - Ensure images are clear and text is readable
        - PDFs with selectable text work best
        - File size should be reasonable (< 10MB recommended)
        """)
        
        # Show current status
        if st.session_state.get('document_text'):
            st.info(f"📄 Document already loaded ({len(st.session_state.document_text)} characters)")
            if st.button("🔍 Analyze Current Document", type="primary"):
                with st.spinner("🤖 Analyzing document with AI..."):
                    analysis_result = analyze_document_with_openai(st.session_state.document_text)
                    if analysis_result:
                        st.session_state.analysis_result = analysis_result
                        st.session_state.stage = 'analysis'
                        st.rerun()

# Stage 2: Analysis Results
elif st.session_state.stage == 'analysis':
    analysis = st.session_state.analysis_result
    
    st.markdown("## 📊 Document Analysis Complete")
    
    # Check if it's a legal document
    is_legal = analysis.get('is_legal_document', True)
    
    if not is_legal:
        # Warning for non-legal documents
        st.error("""
        ⚠️ **This does not appear to be a legal document**
        
        Based on the analysis, this document appears to be: **{}**
        
        **Reason:** {}
        """.format(
            analysis.get('document_type', 'Non-legal document'),
            analysis.get('non_legal_reason', 'This document does not contain legal claims, notices, or formal legal language typically found in legal documents.')
        ))
        
        st.info("""
        **💡 For legal document analysis, please upload:**
        - Legal notices or demand letters
        - Contracts or agreements
        - Court documents
        - Legal correspondence
        - Bills, invoices with legal claims
        - Any document requiring a legal response
        """)
        
        # Still show basic analysis but limit legal features
        st.markdown("### 📋 Basic Document Summary")
        st.write(analysis.get('summary', 'No summary available'))
        
        # Show restart option
        if st.button("🔄 Upload Legal Document", type="primary"):
            # Reset session state
            for key in ['analysis_result', 'legal_notice', 'chat_history', 'document_text']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.stage = 'upload'
            st.rerun()
        
        st.stop()  # Don't show legal analysis features for non-legal documents
    
    # Priority Display (only for legal documents)
    priority = analysis.get('priority', 'Medium').lower()
    priority_class = f"priority-{priority}"
    confidence = analysis.get('document_confidence', 'Medium')
    
    st.markdown(f"""
    <div class="{priority_class}">
        <h3>🚨 Priority: {analysis.get('priority', 'Medium').upper()}</h3>
        <p><strong>Document Type:</strong> {analysis.get('document_type', 'Legal Document')}</p>
        <p><strong>Reasoning:</strong> {analysis.get('priority_reasoning', 'Standard legal matter')}</p>
        <p><strong>Analysis Confidence:</strong> {confidence}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Analysis Details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Summary")
        st.write(analysis.get('summary', 'No summary available'))
        
        st.markdown("### 👥 Key Parties")
        parties = analysis.get('key_parties', [])
        if parties:
            for party in parties:
                st.write(f"• {party}")
        else:
            st.write("No key parties identified")
        
        st.markdown("### 💰 Amounts")
        amounts = analysis.get('amounts', [])
        if amounts:
            for amount in amounts:
                st.write(f"• {amount}")
        else:
            st.write("No amounts specified")
    
    with col2:
        st.markdown("### 📅 Important Dates")
        dates = analysis.get('key_dates', [])
        if dates:
            for date in dates:
                st.write(f"• {date}")
        else:
            st.write("No specific dates found")
        
        st.markdown("### ⚠️ Risks")
        risks = analysis.get('risks', [])
        if risks:
            for risk in risks:
                st.write(f"• {risk}")
        else:
            st.write("No significant risks identified")
        
        st.markdown("### 🎯 Recommended Actions")
        actions = analysis.get('recommended_actions', [])
        if actions:
            for action in actions:
                st.write(f"• {action}")
        else:
            st.write("No specific actions recommended")
    
    # Action Buttons
    st.markdown("### 🚀 Your Next Steps")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📝 Draft Legal Notice", type="primary"):
            with st.spinner("Generating legal notice..."):
                legal_notice = generate_legal_notice(analysis, st.session_state.document_text)
                if legal_notice:
                    st.session_state.legal_notice = legal_notice
                    st.session_state.stage = 'drafting'
                    st.rerun()
    
    with col2:
        if st.button("👨‍💼 Consult Lawyer"):
            st.info("We recommend consulting with a qualified lawyer for this matter.")
    
    with col3:
        if st.button("🔄 Upload New Document"):
            # Reset session state
            for key in ['analysis_result', 'legal_notice', 'chat_history', 'document_text']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.stage = 'upload'
            st.rerun()

# Stage 3: Legal Notice Drafting
elif st.session_state.stage == 'drafting':
    st.markdown("## ✍️ Legal Notice Drafting & Editing")
    
    # Two-column layout for real-time editing
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📄 Legal Notice (Live Document)")
        
        # Display the legal notice in a styled container with better formatting
        st.markdown(f"""
        <div class="legal-notice">
            <pre>{st.session_state.legal_notice}</pre>
        </div>
        """, unsafe_allow_html=True)
        
        # Download and copy options
        st.markdown("---")
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        
        with col_dl1:
            st.download_button(
                label="📥 Download Notice",
                data=st.session_state.legal_notice,
                file_name=f"legal_notice_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col_dl2:
            if st.button("📋 Copy to Clipboard", use_container_width=True):
                st.code(st.session_state.legal_notice, language="text")
                st.success("📋 Notice text displayed above - you can copy it manually!")
        
        with col_dl3:
            word_count = len(st.session_state.legal_notice.split())
            st.metric("Word Count", word_count)
    
    with col2:
        st.markdown("### 💬 Edit Instructions")
        st.markdown("Give instructions to modify your legal notice in real-time!")
        
        # Helpful suggestions
        with st.expander("💡 Suggestions to Improve Your Draft", expanded=True):
            st.markdown("""
            **To make this reply stronger, consider adding:**
            
            🔹 **Why payment is not due**: 
            - 'I already paid on [date] via [method]'
            - 'The product was faulty/defective'
            - 'Services were not provided as agreed'
            - 'The contract was breached by your client'
            
            🔹 **Specific details**:
            - 'Please respond within 15 days'
            - 'I have receipts/documentation to support this claim'
            - 'The amount claimed is incorrect - it should be [amount]'
            
            🔹 **Legal strengthening**:
            - 'Failure to respond may result in legal action'
            - 'We reserve all rights under the law'
            - 'This matter can be resolved amicably'
            
            **Example instructions you can give:**
            - "Add that I paid ₹15,000 on May 15th via bank transfer"
            - "Mention the TV was defective and stopped working after 2 days"
            - "Include a 15-day deadline for response"
            - "Add that I have email proof of the complaint"
            """)
        
        # Common scenarios
        st.markdown("#### 🎯 Common Scenarios")
        scenario_buttons = [
            ("💳 Already Paid", "Add that the payment has already been made with specific date and method"),
            ("🔧 Defective Product", "Mention that the product was faulty or not working as expected"),
            ("📅 Add Deadline", "Include a specific deadline (15 days) for response"),
            ("📋 Have Evidence", "State that supporting documentation/evidence is available")
        ]
        
        cols = st.columns(2)
        for i, (label, instruction) in enumerate(scenario_buttons):
            with cols[i % 2]:
                if st.button(label, key=f"scenario_{i}", use_container_width=True):
                    # Add this instruction to the text area by updating session state
                    st.session_state.suggested_instruction = instruction
                    st.info(f"💡 Try this instruction: '{instruction}'")
        
        # Chat interface
        chat_container = st.container()
        
        # Display chat history
        with chat_container:
            for i, (instruction, response) in enumerate(st.session_state.chat_history):
                st.markdown(f"""
                <div class="chat-message">
                    <strong>You:</strong> {instruction}<br>
                    <strong>Assistant:</strong> {response}
                </div>
                """, unsafe_allow_html=True)
        
        # Chat input
        st.markdown("#### ✍️ Custom Instructions")
        user_instruction = st.text_area(
            "Enter your editing instruction:",
            placeholder="e.g., 'Add a 15-day deadline', 'Mention that payment was already made on July 15th', 'Include that the product was defective'...",
            key="chat_input",
            height=100,
            help="Be specific about what you want to change or add to make the legal notice stronger."
        )
        
        if st.button("✨ Update Notice", type="primary") and user_instruction:
            logger.info(f"User instruction received: {user_instruction}")
            with st.spinner("🤖 Updating legal notice..."):
                try:
                    updated_notice = update_legal_notice(
                        st.session_state.legal_notice,
                        user_instruction,
                        st.session_state.analysis_result
                    )
                    
                    if updated_notice and updated_notice != st.session_state.legal_notice:
                        st.session_state.legal_notice = updated_notice
                        st.session_state.chat_history.append((
                            user_instruction,
                            "✅ Legal notice updated successfully! Check the changes on the left."
                        ))
                        logger.info("Legal notice updated successfully via chat")
                    else:
                        st.session_state.chat_history.append((
                            user_instruction,
                            "⚠️ No changes were made. Please try a different instruction or be more specific."
                        ))
                        logger.warning("No changes made to legal notice")
                    
                    st.rerun()
                    
                except Exception as e:
                    error_msg = f"❌ Error updating notice: {str(e)}"
                    st.session_state.chat_history.append((
                        user_instruction,
                        error_msg
                    ))
                    logger.error(f"Error updating legal notice: {str(e)}")
                    st.rerun()
        
        # Quick action buttons
        st.markdown("### ⚡ Quick Actions")
        
        # Categorized quick actions
        improvement_actions = [
            ("Add 15-Day Deadline", "Add a 15-day deadline for response and make it urgent"),
            ("State Payment Made", "Add that the payment has already been made with reference to date/method"),
            ("Mention Product Faulty", "Add that the product/service was faulty or not as described"),
            ("Include Evidence", "Mention that supporting documentation is available"),
            ("Add Settlement Offer", "Include willingness to discuss and resolve the matter amicably"),
            ("Make More Formal", "Use more formal legal language and professional tone"),
            ("Add Legal Consequences", "Include potential legal consequences for non-compliance"),
            ("Soften Tone", "Make the tone more diplomatic while maintaining firmness")
        ]
        
        # Display quick actions in two columns
        for i in range(0, len(improvement_actions), 2):
            col_a, col_b = st.columns(2)
            
            with col_a:
                if i < len(improvement_actions):
                    label, instruction = improvement_actions[i]
                    if st.button(label, key=f"quick_{i}", use_container_width=True):
                        logger.info(f"Quick action triggered: {label}")
                        with st.spinner(f"Applying '{label}'..."):
                            try:
                                updated_notice = update_legal_notice(
                                    st.session_state.legal_notice,
                                    instruction,
                                    st.session_state.analysis_result
                                )
                                
                                if updated_notice and updated_notice != st.session_state.legal_notice:
                                    st.session_state.legal_notice = updated_notice
                                    st.session_state.chat_history.append((
                                        instruction,
                                        f"✅ Applied '{label}' to your legal notice!"
                                    ))
                                    logger.info(f"Quick action '{label}' applied successfully")
                                else:
                                    st.session_state.chat_history.append((
                                        instruction,
                                        f"⚠️ '{label}' could not be applied. The notice may already have this characteristic."
                                    ))
                                    logger.warning(f"Quick action '{label}' made no changes")
                                
                                st.rerun()
                                
                            except Exception as e:
                                error_msg = f"❌ Error applying '{label}': {str(e)}"
                                st.session_state.chat_history.append((
                                    instruction,
                                    error_msg
                                ))
                                logger.error(f"Error in quick action '{label}': {str(e)}")
                                st.rerun()
            
            with col_b:
                if i + 1 < len(improvement_actions):
                    label, instruction = improvement_actions[i + 1]
                    if st.button(label, key=f"quick_{i+1}", use_container_width=True):
                        logger.info(f"Quick action triggered: {label}")
                        with st.spinner(f"Applying '{label}'..."):
                            try:
                                updated_notice = update_legal_notice(
                                    st.session_state.legal_notice,
                                    instruction,
                                    st.session_state.analysis_result
                                )
                                
                                if updated_notice and updated_notice != st.session_state.legal_notice:
                                    st.session_state.legal_notice = updated_notice
                                    st.session_state.chat_history.append((
                                        instruction,
                                        f"✅ Applied '{label}' to your legal notice!"
                                    ))
                                    logger.info(f"Quick action '{label}' applied successfully")
                                else:
                                    st.session_state.chat_history.append((
                                        instruction,
                                        f"⚠️ '{label}' could not be applied. The notice may already have this characteristic."
                                    ))
                                    logger.warning(f"Quick action '{label}' made no changes")
                                
                                st.rerun()
                                
                            except Exception as e:
                                error_msg = f"❌ Error applying '{label}': {str(e)}"
                                st.session_state.chat_history.append((
                                    instruction,
                                    error_msg
                                ))
                                logger.error(f"Error in quick action '{label}': {str(e)}")
                                st.rerun()
    
    # Navigation buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("⬅️ Back to Analysis"):
            st.session_state.stage = 'analysis'
            st.rerun()
    
    with col2:
        if st.button("🔄 Start Over"):
            # Reset everything
            for key in list(st.session_state.keys()):
                if key != 'stage':
                    del st.session_state[key]
            st.session_state.stage = 'upload'
            st.rerun()
    
    with col3:
        if st.button("📧 Email Notice"):
            st.info("Email functionality would be implemented here.")

# Footer with debug information
st.markdown("---")

# Debug section (only show if there are errors)
if st.checkbox("🔧 Show Debug Information", value=False):
    st.markdown("### 📋 Application Logs")
    
    try:
        with open('legal_analyzer.log', 'r') as log_file:
            log_content = log_file.read()
            if log_content:
                # Show only last 50 lines
                log_lines = log_content.split('\n')
                recent_logs = '\n'.join(log_lines[-50:]) if len(log_lines) > 50 else log_content
                st.text_area("Recent Log Entries:", value=recent_logs, height=300)
            else:
                st.info("No log entries found.")
    except FileNotFoundError:
        st.info("Log file not found.")
    except Exception as e:
        st.error(f"Error reading log file: {str(e)}")
    
    # Session state debug
    st.markdown("### 🔍 Session State")
    if st.checkbox("Show Session State"):
        for key, value in st.session_state.items():
            if isinstance(value, str) and len(value) > 200:
                st.write(f"**{key}:** {value[:200]}... ({len(value)} chars)")
            else:
                st.write(f"**{key}:** {value}")

st.markdown("""
<div style="text-align: center; color: #6b7280; padding: 2rem;">
    <p>⚖️ Legal Document Analyzer | Powered by AI | For informational purposes only</p>
    <p><small>Always consult with a qualified lawyer for legal advice</small></p>
    <p><small>Logs are saved to 'legal_analyzer.log' for debugging</small></p>
</div>
""", unsafe_allow_html=True)