from fastapi import FastAPI, UploadFile, File, HTTPException
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
from typing import List, Dict, Any
from datetime import datetime
import shutil
from pathlib import Path
import logging
from PIL import Image, ImageEnhance
import pytesseract
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
import tiktoken
import mimetypes

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
tokenizer = tiktoken.get_encoding("cl100k_base")

# ChromaDB
chroma_client = chromadb.Client(Settings(
    persist_directory="chroma_db",
    anonymized_telemetry=False
))
try:
    collection = chroma_client.get_collection("documents")
except:
    collection = chroma_client.create_collection("documents")

UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed")
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)
executor = ThreadPoolExecutor()

def estimate_token_count(text: str) -> int:
    return len(tokenizer.encode(text))

# --- FIXED TOKEN-BASED CHUNKER ---
def split_text_into_token_chunks(
    text: str, chunk_size: int = 500, overlap: int = 100
) -> List[str]:
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += chunk_size - overlap  # Slide window
    return chunks

# --- ROBUST TOKEN-LIMITED BATCHING ---
def batch_chunks_by_tokens(
    chunks: List[str], max_tokens_per_batch: int = 290_000, max_chunks_per_batch: int = 2048
) -> List[List[str]]:
    batches = []
    batch = []
    tokens_in_batch = 0
    for chunk in chunks:
        chunk_tokens = estimate_token_count(chunk)
        if chunk_tokens > 8192:
            logger.warning(f"Skipping chunk of {chunk_tokens} tokens (>8192)")
            continue
        if tokens_in_batch + chunk_tokens > max_tokens_per_batch or len(batch) >= max_chunks_per_batch:
            if batch:
                logger.info(f"Batch size: {len(batch)} chunks, tokens: {tokens_in_batch}")
                batches.append(batch)
            batch = []
            tokens_in_batch = 0
        batch.append(chunk)
        tokens_in_batch += chunk_tokens
    if batch:
        logger.info(f"Batch size: {len(batch)} chunks, tokens: {tokens_in_batch}")
        batches.append(batch)
    return batches

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def get_embeddings(chunks: List[str]) -> List[List[float]]:
    try:
        all_embeddings = []
        batches = batch_chunks_by_tokens(chunks)
        logger.info(f"Processing {len(batches)} batches of text chunks")
        for i, batch in enumerate(batches, 1):
            batch_token_count = sum(estimate_token_count(t) for t in batch)
            logger.info(f"→ Batch {i}: {len(batch)} chunks, {batch_token_count} tokens")
            assert batch_token_count <= 290_000, f"Batch {i} is too big: {batch_token_count} tokens"
            response = await openai_client.embeddings.create(
                input=batch,
                model="text-embedding-ada-002"
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        if not all_embeddings:
            logger.error("No embeddings generated.")
            raise ValueError("Empty embeddings list.")
        return all_embeddings
    except Exception as e:
        logger.error(f"Error getting embeddings: {str(e)}")
        raise

# --------- FILE EXTRACTORS (unchanged) ---------
def extract_text_from_pdf(file_path: str) -> Dict[str, Any]:
    start_time = time.time()
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
        if reader.metadata:
            metadata.update({
                "title": str(reader.metadata.get('/Title', '')),
                "author": str(reader.metadata.get('/Author', '')),
                "subject": str(reader.metadata.get('/Subject', ''))
            })
        for i, page in enumerate(reader.pages):
            try:
                text += page.extract_text() + "\n"
            except Exception as e:
                logger.warning(f"Error extracting text from page {i}: {str(e)}")
                continue
    if not text.strip():
        text = "[No text content found in PDF]"
    logger.info(f"PDF processing time: {time.time() - start_time:.2f}s")
    return {"text": text, "metadata": metadata}

def extract_text_from_docx(file_path: str) -> Dict[str, Any]:
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
    start_time = time.time()
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    metadata = {
        "fileSize": os.path.getsize(file_path)
    }
    logger.info(f"TXT processing time: {time.time() - start_time:.2f}s")
    return {"text": text, "metadata": metadata}

# ... (other extractors unchanged, see your original file)
# You can keep your other extractors as-is for CSV, Excel, PPTX, MD, HTML, JSON, RTF, Image

EXTENSION_HANDLERS = {
    '.pdf': extract_text_from_pdf,
    '.docx': extract_text_from_docx,
    '.txt': extract_text_from_txt,
    # ... etc
}

def process_file_sync(file_path: str, file_extension: str) -> Dict[str, Any]:
    handler = EXTENSION_HANDLERS.get(file_extension)
    if not handler:
        raise ValueError(f"Unsupported file extension: {file_extension}")
    logger.info(f"Using handler: {handler.__name__} for extension: {file_extension}")
    return handler(file_path)

@app.post("/process-document")
async def process_document(file: UploadFile = File(...)):
    temp_file = UPLOAD_DIR / file.filename
    try:
        with open(temp_file, "wb") as buffer:
            buffer.write(await file.read())
        file_extension = f".{file.filename.split('.')[-1].lower()}"
        logger.info(f"Received file: {file.filename}, resolved extension: {file_extension}, content_type: {file.content_type}")
        if file_extension not in EXTENSION_HANDLERS:
            guessed_ext = mimetypes.guess_extension(file.content_type)
            logger.info(f"Guessed extension from MIME type: {guessed_ext}")
            if guessed_ext in EXTENSION_HANDLERS:
                file_extension = guessed_ext
            else:
                os.remove(temp_file)
                raise HTTPException(status_code=400, detail=f"Unsupported file extension: {file_extension}")
        result = await asyncio.get_event_loop().run_in_executor(
            executor, process_file_sync, str(temp_file), file_extension
        )
        # --- CHUNK TEXT SAFELY ---
        chunks = split_text_into_token_chunks(result["text"])
        if not chunks:
            raise ValueError("No text content extracted from document")
        logger.info(f"Chunk stats: {len(chunks)} chunks, max tokens in chunk: {max(estimate_token_count(c) for c in chunks)}")
        embeddings = await get_embeddings(chunks)
        metadatas = [{
            "source": str(file.filename),
            "title": str(result["metadata"].get("title", "")),
            "author": str(result["metadata"].get("author", "")),
            "fileSize": str(result["metadata"].get("fileSize", 0)),
            "processedAt": datetime.now().isoformat()
        } for _ in chunks]
        collection.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
            ids=[f"{file.filename}_{i}" for i in range(len(chunks))]
        )
        summary = {
            "name": file.filename,
            "metadata": result["metadata"],
            "chunks": chunks
        }
        summary_path = PROCESSED_DIR / f"{file.filename}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
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
    try:
        query_embedding = await get_embeddings([query])
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
