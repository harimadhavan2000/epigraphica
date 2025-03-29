import os
from typing import List, Optional
import google.generativeai as genai
from google.genai import types
from PyPDF2 import PdfReader
import argparse
from pdf2image import convert_from_path
from PIL import Image
import tempfile
import shutil
import logging
from datetime import datetime
import json
from tqdm import tqdm
import io
import base64

DEFAULT_API_KEY = ""

class PDFEnhancer:
    def __init__(self, api_key: str, log_dir: str = "logs"):
        # Initialize Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.temp_dir = None
        
        # Setup logging
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = os.path.join(log_dir, f'pdf_enhancer_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Create detailed content log
        self.content_log_file = os.path.join(log_dir, f'content_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        self.content_log = []

    def _setup_temp_dir(self):
        """Create temporary directory for image storage"""
        self.temp_dir = tempfile.mkdtemp()
        self.logger.info(f"Created temporary directory: {self.temp_dir}")
        return self.temp_dir

    def _cleanup_temp_dir(self):
        """Clean up temporary directory"""
        if self.temp_dir:
            shutil.rmtree(self.temp_dir)

    def _convert_pdf_to_images(self, pdf_path: str, start_page: int = 0, max_pages: int = 20, dpi: int = 300) -> List[str]:
        """Convert PDF pages to high-resolution images."""
        image_paths = []
        try:
            # Check if file exists
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            # Test poppler installation first
            poppler_ok, poppler_msg = self._test_poppler_installation()
            if not poppler_ok:
                raise Exception(poppler_msg)
            
            # Extract poppler path from the successful test
            poppler_path = poppler_msg.split(": ")[1] if ": " in poppler_msg else None
            
            # Get total pages in PDF
            with open(pdf_path, 'rb') as pdf_file:
                pdf = PdfReader(pdf_file)
                total_pages = len(pdf.pages)
            
            # Validate page range
            if start_page < 0:
                start_page = 0
            if start_page >= total_pages:
                raise ValueError(f"Start page {start_page} exceeds PDF length of {total_pages} pages")
            
            end_page = min(start_page + max_pages, total_pages)
            
            self.logger.info(f"Converting PDF pages {start_page+1} to {end_page} with DPI={dpi}")
            self.logger.info(f"Using poppler path: {poppler_path}")
            
            # Convert specific pages to images
            images = convert_from_path(
                pdf_path,
                dpi=dpi,
                poppler_path=poppler_path,
                fmt='jpeg',
                thread_count=4,
                first_page=start_page + 1,  # poppler uses 1-based page numbers
                last_page=end_page
            )
            
            # Save images
            for i, image in enumerate(images, start=1):
                image_path = os.path.join(self.temp_dir, f'page_{start_page + i}.jpg')
                image.save(image_path, 'JPEG', quality=95)
                image_paths.append(image_path)
                self.logger.info(f"Saved page {start_page + i} as image: {image_path}")
            
            return image_paths
            
        except Exception as e:
            error_msg = f"Error converting PDF to images: {str(e)}"
            self.logger.error(error_msg)
            if os.name == 'nt':
                self.logger.error("On Windows, ensure poppler is installed and in PATH")
            raise Exception(error_msg)

    def _batch_images(self, image_paths: List[str], batch_size: int = 5) -> List[List[str]]:
        """Group images into batches for processing"""
        return [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]

    async def _process_image_batch(self, image_paths: List[str], is_historical: bool = False) -> List[dict]:
        """Process a batch of images with Gemini."""
        self.logger.info(f"Processing batch of {len(image_paths)} images")
        results = []
        
        for image_path in image_paths:
            try:
                self.logger.info(f"Processing image: {os.path.basename(image_path)}")
                
                with Image.open(image_path) as img:
                    # Log image details
                    self._log_content("image_processing", {
                        "image_path": image_path,
                        "size": img.size,
                        "mode": img.mode
                    })
                    
                    response = await self._generate_content(image_path, is_historical)
                    verification = await self._verify_ocr_quality(image_path, response)
                    
                    # Log results
                    self._log_content("ocr_results", {
                        "image_path": image_path,
                        "text_length": len(response["original_text"]) if response else 0,
                        "verification_status": verification.get("status", "unknown"),
                        "issues_found": verification.get("issues", [])
                    })
                    
                    results.append({
                        'page_num': response["page_num"],
                        'original_text': response["original_text"],
                        'footnotes': response["footnotes"],
                        'summary': response["summary"],
                        'verification': verification
                    })
                    
            except Exception as e:
                self.logger.error(f"Error processing image {image_path}: {str(e)}")
                results.append({
                    'page_num': int(os.path.basename(image_path).split('_')[1].split('.')[0]),
                    'original_text': f"Error: {str(e)}",
                    'footnotes': "",
                    'summary': "Failed to process image",
                    'verification': {"status": "error", "issues": [str(e)]}
                })
        
        return results

    def _get_instruction(self, is_historical: bool) -> str:
        """Get appropriate instruction based on document type."""
        if is_historical:
            return """Extract text from these historical document images...
                   [Your historical document instruction here]"""
        else:
            return """Extract ALL text content from these document pages...
                   [Your complex document instruction here]"""

    def extract_text_from_pdf(self, pdf_path: str, start_page: int = 0, max_pages: int = 20) -> List[str]:
        """Extract text from PDF file and create summaries for each page."""
        try:
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
            end_page = min(start_page + max_pages, total_pages)
            
            pages_text = []
            for page_num in range(start_page, end_page):
                page = reader.pages[page_num]
                raw_text = page.extract_text()
                
                # Split text into paragraphs
                paragraphs = [p.strip() for p in raw_text.split('\n\n') if p.strip()]
                
                # Create summary for each meaningful chunk of text
                summaries = []
                current_chunk = []
                current_length = 0
                
                for para in paragraphs:
                    if current_length + len(para) > 1000:  # Chunk size limit
                        if current_chunk:
                            summary = self._generate_summary(' '.join(current_chunk))
                            if summary:
                                summaries.append(summary)
                        current_chunk = [para]
                        current_length = len(para)
                    else:
                        current_chunk.append(para)
                        current_length += len(para)
                
                # Process remaining chunk
                if current_chunk:
                    summary = self._generate_summary(' '.join(current_chunk))
                    if summary:
                        summaries.append(summary)
                
                pages_text.append('\n\n'.join(summaries))
            
            return pages_text
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")

    async def _generate_summary(self, text: str) -> str:
        """Generate summary using Gemini model."""
        if not text.strip():
            return ""
        
        try:
            response = self.model.generate_content(
                """Please provide a concise summary of the following text. 
                Focus on key points and maintain factual accuracy. 
                Format the output with clear structure and bullet points where appropriate:

                {text}""".format(text=text)
            )
            return response.text.strip()
        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")
            return "Failed to generate summary"

    async def enhance_text(self, text: str) -> str:
        """Use Gemini to enhance text readability."""
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text="""Please improve the following text's formatting and readability while preserving all information. 
                        Fix any OCR artifacts, adjust spacing, and organize the content logically:
                        
                        {text}""".format(text=text)
                    ),
                ],
            ),
        ]
        
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
        )

        try:
            response_text = ""
            for chunk in self.model.generate_content_stream(
                contents=contents,
                config=generate_content_config,
            ):
                response_text += chunk.text

            return response_text.strip()
        except Exception as e:
            print(f"Error generating content: {str(e)}")
            return f"Error processing text: {str(e)}"

    async def process_pdf(self, pdf_path: str, output_path: str, start_page: int = 0, 
                         max_pages: int = 20, is_historical: bool = False):
        """Process PDF using image-based approach."""
        self.logger.info(f"Starting PDF processing: {pdf_path}")
        self.logger.info(f"Parameters: start_page={start_page}, max_pages={max_pages}, is_historical={is_historical}")
        
        try:
            self._setup_temp_dir()
            self.logger.info(f"Temporary directory created: {self.temp_dir}")
            
            # Convert PDF to images with page range
            image_paths = self._convert_pdf_to_images(
                pdf_path, 
                start_page=start_page,
                max_pages=max_pages,
                dpi=300
            )
            
            if not image_paths:
                raise Exception("Failed to convert PDF to images")
            
            batches = self._batch_images(image_paths)
            self.logger.info(f"Created {len(batches)} batches for processing")
            
            all_results = []
            for i, batch in enumerate(batches, 1):
                self.logger.info(f"Processing batch {i}/{len(batches)}")
                results = await self._process_image_batch(batch, is_historical)
                all_results.extend(results)
                
                # Log batch statistics
                self._log_content("batch_statistics", {
                    "batch_number": i,
                    "processed_images": len(results),
                    "successful_extractions": len([r for r in results if r.get('original_text')])
                })
            
            if len(batches) > 1:
                self.logger.info("Harmonizing content across batches")
                all_results = await self._harmonize_content(all_results)
            
            self.logger.info(f"Generating HTML output: {output_path}")
            await self._generate_html(all_results, output_path)
            
            # Log final statistics
            self._log_content("final_statistics", {
                "total_pages_processed": len(image_paths),
                "total_batches": len(batches),
                "total_text_extracted": sum(len(r.get('original_text', '')) for r in all_results),
                "output_file_size": os.path.getsize(output_path)
            })
            
        except Exception as e:
            self.logger.error(f"Error during PDF processing: {str(e)}")
            raise
        finally:
            self.logger.info("Cleaning up temporary files")
            self._cleanup_temp_dir()

    async def _generate_html(self, processed_pages: list, output_path: str):
        """Generate HTML output file."""
        html_template = """<!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Enhanced PDF Text</title>
            <style>
                body { 
                    margin: 0;
                    padding: 20px;
                    font-family: Arial, sans-serif;
                }
                .page { 
                    display: none;
                    margin-bottom: 40px;
                    height: 100vh;
                }
                .page.active {
                    display: flex;
                }
                .main-content {
                    width: 70%;
                    height: 100vh;
                    overflow-y: auto;
                    padding: 20px;
                    display: flex;
                    flex-direction: column;
                }
                .text-content {
                    flex: 2;
                    overflow-y: auto;
                    padding: 20px;
                    background: #fff;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }
                .footnotes {
                    flex: 1;
                    padding: 20px;
                    background: #f9f9f9;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    font-size: 0.9em;
                    overflow-y: auto;
                }
                .side-content {
                    width: 30%;
                    height: 100vh;
                    overflow-y: auto;
                    padding: 20px;
                    background: #f5f5f5;
                    border-left: 1px solid #ddd;
                }
                .navigation {
                    position: fixed;
                    bottom: 20px;
                    right: 20px;
                    display: flex;
                    gap: 10px;
                }
                .nav-button {
                    padding: 10px 20px;
                    background: #007bff;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                }
                .page-header {
                    padding: 10px;
                    background: #f5f5f5;
                    margin-bottom: 15px;
                    border-radius: 3px;
                }
                .footnotes-header {
                    font-weight: bold;
                    margin-bottom: 10px;
                    color: #666;
                }
            </style>
        </head>
        <body>"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
            
            for page in processed_pages:
                f.write(f'''
                    <div class="page" id="page_{page['page_num']}">
                        <div class="main-content">
                            <div class="page-header">Page {page["page_num"]}</div>
                            <div class="text-content">{page["original_text"]}</div>
                            <div class="footnotes">
                                <div class="footnotes-header">Footnotes</div>
                                {page.get("footnotes", "")}
                            </div>
                        </div>
                        <div class="side-content">
                            <div class="summary">{page["summary"]}</div>
                        </div>
                    </div>
                ''')
            
            f.write('''
                <div class="navigation">
                    <button class="nav-button" onclick="previousPage()">←</button>
                    <button class="nav-button" onclick="nextPage()">→</button>
                </div>
                <script>
                    let currentPage = 0;
                    const pages = document.querySelectorAll('.page');
                    
                    function showPage(pageNum) {
                        pages[currentPage].classList.remove('active');
                        currentPage = pageNum;
                        pages[currentPage].classList.add('active');
                    }
                    
                    function nextPage() {
                        if (currentPage < pages.length - 1) {
                            showPage(currentPage + 1);
                        }
                    }
                    
                    function previousPage() {
                        if (currentPage > 0) {
                            showPage(currentPage - 1);
                        }
                    }
                    
                    document.addEventListener('keydown', (e) => {
                        if (e.key === 'ArrowLeft') previousPage();
                        if (e.key === 'ArrowRight') nextPage();
                    });
                    
                    // Show first page initially
                    showPage(0);
                </script>
            </body>
            </html>
            ''')

    def _log_content(self, stage: str, content: dict):
        """Log detailed content information"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            **content
        }
        self.content_log.append(entry)
        
        # Write to JSON file
        with open(self.content_log_file, 'w') as f:
            json.dump(self.content_log, f, indent=2)

    def _test_poppler_installation(self) -> tuple[bool, str]:
        """Test if poppler is correctly installed and accessible."""
        try:
            # Windows-specific poppler path check
            if os.name == 'nt':
                possible_paths = [
                    r"C:\Program Files\poppler\Library\bin",
                    r"C:\Program Files\poppler\bin",
                    r"C:\poppler\bin",
                    r"C:\Users\munch\poppler\Library\bin",
                    r"C:\Users\munch\poppler\poppler-24.08.0\Library\bin",
                    # Add current directory check
                    os.path.join(os.getcwd(), "poppler", "Library", "bin"),
                    os.path.join(os.path.expanduser("~"), "poppler", "Library", "bin")
                ]
                
                self.logger.info("Checking possible poppler paths:")
                for path in possible_paths:
                    self.logger.info(f"Checking path: {path}")
                    if os.path.exists(path):
                        self.logger.info(f"Path exists: {path}")
                        pdftoppm_path = os.path.join(path, "pdftoppm.exe")
                        if os.path.exists(pdftoppm_path):
                            self.logger.info(f"Found pdftoppm.exe at: {pdftoppm_path}")
                            return True, f"Poppler found at: {path}"
                        else:
                            self.logger.warning(f"Path exists but pdftoppm.exe not found in: {path}")
                    else:
                        self.logger.warning(f"Path does not exist: {path}")
                
                # If we get here, no valid path was found
                error_msg = (
                    "Poppler not found. Please:\n"
                    "1. Download from: https://github.com/oschwartz10612/poppler-windows/releases/\n"
                    "2. Extract to C:\\Users\\munch\\poppler\n"
                    "3. Ensure the bin directory contains pdftoppm.exe"
                )
                self.logger.error(error_msg)
                return False, error_msg
            
            else:  # Non-Windows systems
                import subprocess
                try:
                    result = subprocess.run(['pdftoppm', '-v'], capture_output=True, text=True)
                    version = result.stderr.strip()
                    return True, f"Poppler is installed: {version}"
                except FileNotFoundError:
                    return False, "Poppler (pdftoppm) not found in system PATH"
            
        except Exception as e:
            error_msg = f"Error testing poppler installation: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg

    async def _generate_content(self, image_path: str, is_historical: bool = False) -> dict:
        """Generate content analysis from image using Gemini Vision."""
        try:
            self.logger.info(f"Analyzing image: {os.path.basename(image_path)}")
            
            # Read image as bytes
            with Image.open(image_path) as img:
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG')
                img_bytes = img_byte_arr.getvalue()
                
                # Updated prompt with footnote handling
                prompt = """
                Extract the text from this image with the following rules:
                1. Focus on the main body text first
                2. Separate footnotes or references with a clear "---FOOTNOTES---" marker
                3. Ignore page numbers
                4. Preserve paragraph structure
                5. Do not add any commentary or analysis
                6. Do not include phrases like "The text reads:" or "The document contains:"
                
                Format the output as:
                [Main Text]
                
                ---FOOTNOTES---
                [Footnotes/References if any]
                """
                
                # Generate content using Gemini Vision
                with tqdm(total=1, desc="Generating content") as pbar:
                    response = self.model.generate_content([
                        prompt,
                        {'mime_type': 'image/jpeg', 'data': base64.b64encode(img_bytes).decode('utf-8')}
                    ])
                    pbar.update(1)
                
                # Process text and separate main content from footnotes
                text_parts = response.text.strip().split('---FOOTNOTES---')
                main_text = text_parts[0].strip()
                footnotes = text_parts[1].strip() if len(text_parts) > 1 else ""
                
                # Generate summary first
                summary = await self._generate_summary(main_text) if main_text else "No text to summarize"
                
                return {
                    "original_text": main_text,
                    "footnotes": footnotes,
                    "summary": summary,
                    "page_num": int(os.path.basename(image_path).split('_')[1].split('.')[0])
                }
                
        except Exception as e:
            error_msg = f"Error processing image {image_path}: {str(e)}"
            self.logger.error(error_msg)
            return {
                "original_text": f"Error: {str(e)}",
                "footnotes": "",
                "summary": "Failed to process image",
                "page_num": int(os.path.basename(image_path).split('_')[1].split('.')[0])
            }

    async def _verify_ocr_quality(self, image_path: str, text_content: dict) -> dict:
        """Verify OCR quality and identify potential issues."""
        try:
            self.logger.info(f"Verifying OCR quality for: {os.path.basename(image_path)}")
            
            original_text = text_content.get('original_text', '')
            if not original_text:
                return {
                    "status": "failed",
                    "issues": ["No text content found"],
                    "confidence": 0.0
                }
            
            # Basic quality checks with progress bar
            issues = []
            confidence = 1.0
            
            with tqdm(total=4, desc="Verifying OCR quality") as pbar:
                # Check 1: Minimum text length
                if len(original_text) < 50:
                    issues.append("Very short text content - possible OCR failure")
                    confidence *= 0.5
                pbar.update(1)
                
                # Check 2: Common OCR artifacts
                artifacts = ['|', '[?]', '{}', '[]', '( )', '...', '???']
                found_artifacts = [a for a in artifacts if a in original_text]
                if found_artifacts:
                    issues.append(f"Found OCR artifacts: {', '.join(found_artifacts)}")
                    confidence *= 0.8
                pbar.update(1)
                
                # Check 3: Reasonable word length
                words = original_text.split()
                avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
                if avg_word_length > 15:
                    issues.append("Unusually long words detected - possible OCR errors")
                    confidence *= 0.7
                pbar.update(1)
                
                # Check 4: Paragraph structure
                if '\n\n' not in original_text and len(original_text) > 200:
                    issues.append("No paragraph breaks detected - possible formatting issues")
                    confidence *= 0.9
                pbar.update(1)
            
            # Log verification results
            self._log_content("ocr_verification", {
                "image": os.path.basename(image_path),
                "confidence": confidence,
                "issues_found": len(issues),
                "issues": issues
            })
            
            return {
                "status": "passed" if confidence > 0.7 and not issues else "warning",
                "issues": issues,
                "confidence": confidence
            }
            
        except Exception as e:
            error_msg = f"Error verifying OCR quality: {str(e)}"
            self.logger.error(error_msg)
            return {
                "status": "error",
                "issues": [error_msg],
                "confidence": 0.0
            }

    async def _harmonize_content(self, results: List[dict]) -> List[dict]:
        """Harmonize content across all processed pages."""
        try:
            self.logger.info("Harmonizing content across pages")
            
            # Sort results by page number
            results.sort(key=lambda x: x['page_num'])
            
            # Check for any missing pages
            page_numbers = [r['page_num'] for r in results]
            expected_pages = range(min(page_numbers), max(page_numbers) + 1)
            missing_pages = set(expected_pages) - set(page_numbers)
            
            if missing_pages:
                self.logger.warning(f"Missing pages detected: {missing_pages}")
                
            # Add any missing pages as empty entries
            for page_num in missing_pages:
                results.append({
                    'page_num': page_num,
                    'original_text': "Page processing failed",
                    'summary': "No content available",
                    'footnotes': ""
                })
            
            # Re-sort after adding missing pages
            results.sort(key=lambda x: x['page_num'])
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during content harmonization: {str(e)}")
            return results

def main():
    parser = argparse.ArgumentParser(description='Enhance PDF readability using Gemini AI')
    parser.add_argument('pdf_path', help='Path to input PDF file')
    parser.add_argument('output_path', help='Path to output text file')
    parser.add_argument('--start-page', type=int, default=0, help='Starting page number (0-based)')
    parser.add_argument('--max-pages', type=int, default=20, help='Maximum number of pages to process')
    parser.add_argument('--api-key', help='Google API Key (or set GOOGLE_API_KEY environment variable)')
    
    args = parser.parse_args()
    
    # Try to get API key from command line args or environment variable
    api_key = args.api_key or os.environ.get('GOOGLE_API_KEY', DEFAULT_API_KEY)
    
    if not api_key:
        raise ValueError("API key must be provided either via --api-key argument or GOOGLE_API_KEY environment variable")
    
    enhancer = PDFEnhancer(api_key)
    
    import asyncio
    asyncio.run(enhancer.process_pdf(
        args.pdf_path,
        args.output_path,
        args.start_page,
        args.max_pages
    ))

if __name__ == "__main__":
    main() 