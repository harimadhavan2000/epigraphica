from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
from pdf_enhancer import PDFEnhancer
import asyncio
import tempfile
import uuid
import logging
from datetime import datetime

# Define the response model
class ProcessResponse(BaseModel):
    success: bool
    output_id: Optional[str] = None
    error: Optional[str] = None

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Return the upload form HTML"""
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>PDF Enhancer</title>
            <link rel="stylesheet" href="/static/styles.css">
        </head>
        <body>
            <div class="container">
                <h1>PDF Enhancer</h1>
                <form id="uploadForm" class="upload-form">
                    <input type="file" name="file" accept=".pdf" required>
                    <input type="number" name="start_page" placeholder="Start Page" value="0" min="0">
                    <input type="number" name="max_pages" placeholder="Max Pages" value="20" min="1">
                    <input type="text" name="api_key" placeholder="Gemini API Key">
                    <button type="submit">Process PDF</button>
                </form>
                <div id="status" class="status"></div>
            </div>
            <script src="/static/script.js"></script>
        </body>
    </html>
    """

@app.post("/process", response_model=ProcessResponse)
async def process_pdf(
    file: UploadFile = File(...),
    start_page: int = Form(0),
    max_pages: int = Form(20),
    api_key: str = Form(None)
):
    """Process uploaded PDF file"""
    temp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(temp_dir, f"{uuid.uuid4()}.pdf")
    
    # Setup logging for web service
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    web_logger = logging.getLogger("web_service")
    web_logger.setLevel(logging.INFO)
    
    log_file = os.path.join(log_dir, f'web_service_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    web_logger.addHandler(handler)
    
    try:
        web_logger.info(f"Received file: {file.filename}, size: {file.size} bytes")
        web_logger.info(f"Parameters: start_page={start_page}, max_pages={max_pages}")
        
        # Save uploaded file
        content = await file.read()
        with open(pdf_path, "wb") as buffer:
            buffer.write(content)
        web_logger.info(f"Saved uploaded file to: {pdf_path}")
        
        # Process PDF
        enhancer = PDFEnhancer(api_key or os.getenv('GOOGLE_API_KEY'))
        output_dir = os.path.join("static", "output")
        os.makedirs(output_dir, exist_ok=True)
        
        output_id = str(uuid.uuid4())
        output_path = os.path.join(output_dir, f"{output_id}.html")
        
        web_logger.info(f"Starting PDF processing with output_id: {output_id}")
        await enhancer.process_pdf(pdf_path, output_path, start_page, max_pages)
        web_logger.info(f"PDF processing completed successfully")
        
        return ProcessResponse(success=True, output_id=output_id)
        
    except Exception as e:
        web_logger.error(f"Error processing request: {str(e)}")
        return ProcessResponse(success=False, error=str(e))
    finally:
        # Cleanup
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)
            web_logger.info("Cleaned up temporary PDF file")
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
            web_logger.info("Cleaned up temporary directory")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 