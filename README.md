# Legal Document Analyzer

A Streamlit application that analyzes legal documents, provides summaries, and helps draft responses.

## Features

- Upload and process legal documents (PDF, DOCX, TXT, images)
- Extract text using OCR for scanned documents
- Analyze document content with OpenAI's AI models
- Generate plain-English summaries
- Categorize documents by priority
- Recommend appropriate actions
- Draft and customize response letters
- Chat-based assistance for refining responses

## Setup Instructions

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install Tesseract OCR for image processing:
   - On Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
   - On macOS: `brew install tesseract`
   - On Windows: Download and install from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

4. Create a `.env` file in the project directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

5. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Upload a legal document through the file uploader
2. Enable OCR if processing scanned documents
3. Click "Analyze Document" to process the content
4. Review the summary, key details, and recommended actions
5. Use the provided template or chat interface to draft a response
6. Download or print your final response

## Dependencies

- Streamlit: Web application framework
- OpenAI: AI models for document analysis
- PyPDF2: PDF text extraction
- python-docx: DOCX processing
- pytesseract: OCR for scanned documents
- pdf2image: PDF to image conversion for OCR
- python-dotenv: Environment variable management

## Production Deployment

For production deployment:

1. Set up a proper logging system
2. Configure secure environment variable handling
3. Deploy to a platform like Streamlit Cloud, Heroku, or AWS

## License

MIT License