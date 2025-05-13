from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import chromadb
from chromadb.config import Settings
import PyPDF2
import docx
import pandas as pd
from pptx import Presentation
from bs4 import BeautifulSoup
import json
import striprtf
import os
from typing import List, Dict, Any, Tuple
from datetime import datetime
import shutil
from pathlib import Path
import logging
import mimetypes
from PIL import Image, ImageEnhance
import pytesseract
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from tqdm import tqdm
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
import tiktoken

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize ChromaDB
chroma_client = chromadb.Client(Settings(
    persist_directory="chroma_db",
    anonymized_telemetry=False
))

# Create collections if they don't exist
try:
    collection = chroma_client.get_collection("documents")
except:
    collection = chroma_client.create_collection("documents")

# Create upload directories
UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed")
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

# Initialize thread pool for async operations
executor = ThreadPoolExecutor()

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

def estimate_token_count(text: str) -> int:
    """Estimate the number of tokens in a text string."""
    return len(tokenizer.encode(text))

def chunk_texts_for_openai(texts: List[str], max_tokens_per_batch: int = 280000) -> List[List[str]]:
    """Split texts into batches that fit within OpenAI's token limits."""
    batches = []
    current_batch = []
    current_tokens = 0

    for text in texts:
        tokens = estimate_token_count(text)
        
        # Skip empty or very short texts
        if not text.strip() or tokens < 10:
            continue
            
        # Skip texts that are too long for a single embedding
        if tokens > 8192:
            logger.warning(f"Skipping text chunk of {tokens} tokens (exceeds 8192 limit)")
            continue
            
        if current_tokens + tokens > max_tokens_per_batch:
            if current_batch:  # Only append if we have items
                batches.append(current_batch)
            current_batch = []
            current_tokens = 0
            
        current_batch.append(text)
        current_tokens += tokens

    if current_batch:  # Don't forget the last batch
        batches.append(current_batch)

    return batches

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings from OpenAI API with retry logic and batching."""
    try:
        all_embeddings = []
        batches = chunk_texts_for_openai(texts)
        
        if not batches:
            logger.error("No valid text chunks to process")
            raise ValueError("No valid text chunks to process")
            
        logger.info(f"Processing {len(batches)} batches of text chunks")
        
        for i, batch in enumerate(batches, 1):
            logger.info(f"Processing batch {i}/{len(batches)} with {len(batch)} chunks")
            response = await openai_client.embeddings.create(
                input=batch,
                model="text-embedding-ada-002"
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
        if not all_embeddings:
            logger.error("No embeddings generated. Check input document content or model behavior.")
            raise ValueError("Empty embeddings list — check document content or embedding model.")
            
        return all_embeddings
    except Exception as e:
        logger.error(f"Error getting embeddings: {str(e)}")
        raise

def extract_text_from_pdf(file_path: str) -> Dict[str, Any]:
    """Extract text and metadata from PDF files."""
    start_time = time.time()
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            metadata = {
                "title": "",
                "author": "",
                "subject": "",
                "pageCount": len(reader.pages),
                "fileSize": os.path.getsize(file_path)
            }
            
            # Safely get metadata
            if reader.metadata:
                metadata.update({
                    "title": str(reader.metadata.get('/Title', '')),
                    "author": str(reader.metadata.get('/Author', '')),
                    "subject": str(reader.metadata.get('/Subject', ''))
                })
            
            # Process pages in chunks to avoid memory issues
            CHUNK_SIZE = 10
            for i in range(0, len(reader.pages), CHUNK_SIZE):
                chunk_pages = reader.pages[i:i + CHUNK_SIZE]
                for page in chunk_pages:
                    try:
                        text += page.extract_text() + "\n"
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {i}: {str(e)}")
                        continue
        
        if not text.strip():
            logger.warning("No text extracted from PDF")
            text = "[No text content found in PDF]"
            
        logger.info(f"PDF processing time: {time.time() - start_time:.2f}s")
        return {"text": text, "metadata": metadata}
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise

def extract_text_from_docx(file_path: str) -> Dict[str, Any]:
    """Extract text and metadata from DOCX files."""
    start_time = time.time()
    doc = docx.Document(file_path)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    metadata = {
        "title": doc.core_properties.title,
        "author": doc.core_properties.author,
        "fileSize": os.path.getsize(file_path)
    }
    logger.info(f"DOCX processing time: {time.time() - start_time:.2f}s")
    return {"text": text, "metadata": metadata}

def extract_text_from_txt(file_path: str) -> Dict[str, Any]:
    """Extract text from plain text files."""
    start_time = time.time()
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    metadata = {
        "fileSize": os.path.getsize(file_path)
    }
    logger.info(f"TXT processing time: {time.time() - start_time:.2f}s")
    return {"text": text, "metadata": metadata}

def extract_text_from_csv(file_path: str) -> Dict[str, Any]:
    """Extract text and metadata from CSV files."""
    start_time = time.time()
    try:
        # Read CSV in chunks to handle large files
        chunks = []
        metadata = {
            "columns": "",
            "rows": 0,
            "chunks": 0,
            "fileSize": os.path.getsize(file_path)
        }
        
        # Read CSV in chunks of 1000 rows
        chunk_size = 1000
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            # Process each row in the chunk
            for _, row in chunk.iterrows():
                # Only include non-empty values
                values = [str(val) for val in row if pd.notna(val)]
                if values:
                    # Join values with a meaningful separator
                    text = " | ".join(values)
                    chunks.append(text)
            
            # Update row count
            metadata["rows"] += len(chunk)
            
            # Break if we've reached the maximum number of chunks
            if len(chunks) >= 1000:
                logger.warning("Reached maximum chunk limit (1000)")
                break
        
        # Get column names from first chunk
        first_chunk = pd.read_csv(file_path, nrows=1)
        metadata["columns"] = str(first_chunk.columns.tolist())
        
        # Join all chunks with newlines
        text = "\n".join(chunks)
        metadata["chunks"] = len(chunks)
        
        logger.info(f"CSV processing time: {time.time() - start_time:.2f}s")
        return {"text": text, "metadata": metadata}
    except Exception as e:
        logger.error(f"Error processing CSV: {str(e)}")
        raise

def extract_text_from_excel(file_path: str) -> Dict[str, Any]:
    """Extract text and metadata from Excel files."""
    start_time = time.time()
    df = pd.read_excel(file_path)
    # Limit preview to first 100 rows for performance
    preview = df.head(100).to_string(index=False)
    metadata = {
        "columns": df.columns.tolist(),
        "rows": len(df),
        "previewRows": 100,
        "sheets": len(df.sheet_names) if hasattr(df, 'sheet_names') else 1,
        "fileSize": os.path.getsize(file_path)
    }
    logger.info(f"Excel processing time: {time.time() - start_time:.2f}s")
    return {"text": preview, "metadata": metadata}

def extract_text_from_pptx(file_path: str) -> Dict[str, Any]:
    """Extract text and metadata from PowerPoint files."""
    start_time = time.time()
    prs = Presentation(file_path)
    text = ""
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text += shape.text + "\n"
    metadata = {
        "slides": len(prs.slides),
        "fileSize": os.path.getsize(file_path)
    }
    logger.info(f"PPTX processing time: {time.time() - start_time:.2f}s")
    return {"text": text, "metadata": metadata}

def extract_text_from_md(file_path: str) -> Dict[str, Any]:
    """Extract text from Markdown files."""
    start_time = time.time()
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    metadata = {
        "fileSize": os.path.getsize(file_path)
    }
    logger.info(f"MD processing time: {time.time() - start_time:.2f}s")
    return {"text": text, "metadata": metadata}

def extract_text_from_html(file_path: str) -> Dict[str, Any]:
    """Extract text from HTML files."""
    start_time = time.time()
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file.read(), 'html.parser')
        text = soup.get_text(separator='\n', strip=True)
    metadata = {
        "fileSize": os.path.getsize(file_path)
    }
    logger.info(f"HTML processing time: {time.time() - start_time:.2f}s")
    return {"text": text, "metadata": metadata}

def extract_text_from_json(file_path: str) -> Dict[str, Any]:
    """Extract text from JSON files."""
    start_time = time.time()
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        text = json.dumps(data, indent=2)
    metadata = {
        "fileSize": os.path.getsize(file_path)
    }
    logger.info(f"JSON processing time: {time.time() - start_time:.2f}s")
    return {"text": text, "metadata": metadata}

def extract_text_from_rtf(file_path: str) -> Dict[str, Any]:
    """Extract text from RTF files."""
    start_time = time.time()
    with open(file_path, 'r', encoding='utf-8') as file:
        rtf_text = file.read()
        text = striprtf.rtf_to_text(rtf_text)
    metadata = {
        "fileSize": os.path.getsize(file_path)
    }
    logger.info(f"RTF processing time: {time.time() - start_time:.2f}s")
    return {"text": text, "metadata": metadata}

def extract_text_from_image(file_path: str) -> Dict[str, Any]:
    """Extract text from image files using OCR."""
    start_time = time.time()
    try:
        logger.info(f"Running OCR on image: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")
            
        # Open and preprocess image
        img = Image.open(file_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize large images to improve OCR performance
        max_size = 2000
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Enhance image for better OCR
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)  # Increase contrast for better OCR
        
        # Perform OCR with improved settings
        custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
        try:
            text = pytesseract.image_to_string(img, config=custom_config)
        except Exception as ocr_error:
            logger.error(f"OCR failed: {ocr_error}")
            text = "[OCR failed to extract text]"
        
        if not text.strip():
            logger.warning(f"No text extracted from image: {file_path}")
            text = "[No text content found in image]"
            
        metadata = {
            "format": img.format,
            "mode": img.mode,
            "size": str(img.size),
            "fileSize": os.path.getsize(file_path)
        }
        
        logger.info(f"Image processing time: {time.time() - start_time:.2f}s")
        return {"text": text, "metadata": metadata}
        
    except Exception as e:
        logger.error(f"Error processing image {file_path}: {str(e)}", exc_info=True)
        raise

def extract_text_fallback(file_path: str) -> Dict[str, Any]:
    """Fallback text extraction for unknown file types."""
    start_time = time.time()
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    except UnicodeDecodeError:
        text = "[Binary file content not extractable]"
    metadata = {
        "fileSize": os.path.getsize(file_path)
    }
    logger.info(f"Fallback processing time: {time.time() - start_time:.2f}s")
    return {"text": text, "metadata": metadata}

# Map file extensions to their handlers
EXTENSION_HANDLERS = {
    '.pdf': extract_text_from_pdf,
    '.docx': extract_text_from_docx,
    '.txt': extract_text_from_txt,
    '.csv': extract_text_from_csv,
    '.xlsx': extract_text_from_excel,
    '.xls': extract_text_from_excel,
    '.pptx': extract_text_from_pptx,
    '.ppt': extract_text_from_pptx,
    '.md': extract_text_from_md,
    '.html': extract_text_from_html,
    '.htm': extract_text_from_html,
    '.json': extract_text_from_json,
    '.rtf': extract_text_from_rtf,
    '.png': extract_text_from_image,
    '.jpg': extract_text_from_image,
    '.jpeg': extract_text_from_image,
    '.tiff': extract_text_from_image,
    '.bmp': extract_text_from_image,
}

def split_text_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks with token limit awareness."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_chunk.append(word)
        current_size += len(word) + 1  # +1 for space
        
        if current_size >= chunk_size:
            chunk_text = " ".join(current_chunk)
            # Only add chunks that aren't too long
            if estimate_token_count(chunk_text) <= 8192:
                chunks.append(chunk_text)
            # Keep overlap words
            current_chunk = current_chunk[-overlap:] if overlap > 0 else []
            current_size = sum(len(word) + 1 for word in current_chunk)
    
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        if estimate_token_count(chunk_text) <= 8192:
            chunks.append(chunk_text)
    
    return chunks

def process_file_sync(file_path: str, file_extension: str) -> Dict[str, Any]:
    """Process a file based on its extension."""
    handler = EXTENSION_HANDLERS.get(file_extension)
    if not handler:
        raise ValueError(f"Unsupported file extension: {file_extension}")
    
    logger.info(f"Using handler: {handler.__name__} for extension: {file_extension}")
    try:
        return handler(file_path)
    except Exception as e:
        logger.error(f"Handler {handler.__name__} failed: {e}")
        raise

@app.post("/process-document")
async def process_document(file: UploadFile = File(...)):
    """Process uploaded document and store in ChromaDB."""
    try:
        # Create a temporary file
        temp_file = UPLOAD_DIR / file.filename
        with open(temp_file, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Get file extension with dot prefix
        file_extension = f".{file.filename.split('.')[-1].lower()}"
        logger.info(f"Received file: {file.filename}, resolved extension: {file_extension}, content_type: {file.content_type}")

        # Robust extension handling
        if file_extension not in EXTENSION_HANDLERS:
            guessed_ext = mimetypes.guess_extension(file.content_type)
            logger.info(f"Guessed extension from MIME type: {guessed_ext}")
            if guessed_ext in EXTENSION_HANDLERS:
                file_extension = guessed_ext
            else:
                logger.error(f"Unsupported or unrecognized file extension: {file_extension} (guessed: {guessed_ext})")
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                raise HTTPException(status_code=400, detail=f"Unsupported or unrecognized file extension: {file_extension}")

        # Process file
        result = await asyncio.get_event_loop().run_in_executor(
            executor, process_file_sync, str(temp_file), file_extension
        )
        
        # Split text into chunks
        chunks = split_text_into_chunks(result["text"])
        
        if not chunks:
            raise ValueError("No text content extracted from document")
            
        # Generate embeddings
        embeddings = await get_embeddings(chunks)
        
        # Prepare metadata for each chunk
        metadatas = [{
            "source": str(file.filename),
            "title": str(result["metadata"].get("title", "")),
            "author": str(result["metadata"].get("author", "")),
            "fileSize": str(result["metadata"].get("fileSize", 0)),
            "processedAt": datetime.now().isoformat()
        } for _ in chunks]
        
        # Add to ChromaDB
        collection.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
            ids=[f"{file.filename}_{i}" for i in range(len(chunks))]
        )
        
        # Save a summary JSON for later retrieval
        summary = {
            "name": file.filename,
            "metadata": result["metadata"],
            "chunks": chunks
        }
        
        summary_path = PROCESSED_DIR / f"{file.filename}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # Clean up
        os.remove(temp_file)
        
        return {
            "message": "Document processed successfully",
            "chunks": len(chunks),
            "metadata": result["metadata"]
        }
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def search_documents(query: str, limit: int = 5):
    """Search through processed documents."""
    try:
        # Generate query embedding using OpenAI
        query_embedding = await get_embeddings([query])
        
        # Search in ChromaDB
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=limit
        )
        
        return {
            "results": [
                {
                    "text": doc,
                    "metadata": meta
                }
                for doc, meta in zip(results["documents"][0], results["metadatas"][0])
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/document/{filename}")
async def get_processed_document(filename: str):
    """Retrieve a processed document by filename."""
    try:
        summary_path = PROCESSED_DIR / f"{filename}.json"
        if not summary_path.exists():
            raise HTTPException(status_code=404, detail="Document not found")

        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception as e:
        logger.error(f"Error retrieving document {filename}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 