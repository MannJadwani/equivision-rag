import os
import io
import uuid
import math
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pdfplumber
import numpy as np
import pandas as pd
import faiss

from dotenv import load_dotenv
from openai import OpenAI

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    HTMLResponse,
    StreamingResponse
)

from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------
# LOGGING
# ---------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("faiss-rag")

# ---------------------------------------------------
# ENV
# ---------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

DEFAULT_EMBED_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small"
)

DEFAULT_CHAT_MODEL = os.getenv(
    "CHAT_MODEL",
    "gpt-4o-mini"
)

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found")

oa_client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------------------------------
# FASTAPI
# ---------------------------------------------------

app = FastAPI(title="FAISS PDF Extraction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# QUESTIONS
# ---------------------------------------------------

HARDCODED_QUESTIONS = [
    "Who is the arranger/lead manager?",
]

# ---------------------------------------------------
# SCHEMAS
# ---------------------------------------------------

class QAItem(BaseModel):
    question: str
    answer: str


class FileResult(BaseModel):
    filename: str
    qas: List[QAItem]


# ---------------------------------------------------
# PDF CHUNKING
# ---------------------------------------------------

def extract_pdf_text_chunks(
    pdf_bytes: bytes,
    max_tokens: int = 800,
    overlap: int = 150
) -> List[Dict[str, Any]]:

    approx_chars = max_tokens * 4
    approx_overlap = overlap * 4

    chunks = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:

        logger.info(f"Pages: {len(pdf.pages)}")

        for pageno, page in enumerate(pdf.pages, start=1):

            text = (page.extract_text() or "").strip()

            if not text:
                continue

            start = 0
            n = len(text)

            while start < n:

                end = min(start + approx_chars, n)

                snippet = text[start:end].strip()

                if snippet:
                    chunks.append({
                        "page": pageno,
                        "text": snippet
                    })

                if end >= n:
                    break

                start = max(0, end - approx_overlap)

    logger.info(f"Total chunks: {len(chunks)}")

    return chunks

# ---------------------------------------------------
# EMBEDDINGS
# ---------------------------------------------------

def embed_texts(
    texts: List[str],
    model: str
) -> List[List[float]]:

    all_embeddings = []

    batch_size = 100

    total = len(texts)

    total_batches = math.ceil(total / batch_size)

    for i in range(0, total, batch_size):

        batch = texts[i:i + batch_size]

        batch_no = i // batch_size + 1

        logger.info(
            f"Embedding batch "
            f"{batch_no}/{total_batches}"
        )

        response = oa_client.embeddings.create(
            model=model,
            input=batch
        )

        all_embeddings.extend(
            [d.embedding for d in response.data]
        )

    return all_embeddings

# ---------------------------------------------------
# FAISS
# ---------------------------------------------------

def create_faiss_index(
    embeddings: List[List[float]]
):

    vectors = np.array(
        embeddings,
        dtype=np.float32
    )

    dimension = vectors.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(vectors)

    return index


def search_faiss(
    index,
    query_embedding,
    chunks,
    top_k=6
):

    query_vector = np.array(
        [query_embedding],
        dtype=np.float32
    )

    distances, indices = index.search(
        query_vector,
        top_k
    )

    results = []

    for idx, distance in zip(
        indices[0],
        distances[0]
    ):

        if idx == -1:
            continue

        chunk = chunks[idx]

        results.append({
            "id": str(idx),
            "score": float(distance),
            "text": chunk["text"],
            "page": chunk["page"]
        })

    return results

# ---------------------------------------------------
# PROMPT
# ---------------------------------------------------

def build_prompt(
    query,
    contexts
):

    context_text = "\n\n".join([
        f"(Page {c['page']}) {c['text']}"
        for c in contexts
    ])

    return f"""
You are an information extraction assistant.

Rules:
- Return ONLY the exact answer
- No explanations
- No extra text
- If answer not found return exactly:
Not found

Question:
{query}

Context:
{context_text}

Answer:
"""

# ---------------------------------------------------
# ANSWER GENERATION
# ---------------------------------------------------

def generate_answer(
    faiss_index,
    chunks,
    question,
    embedding_model,
    chat_model,
    top_k=6
):

    try:

        q_embedding = oa_client.embeddings.create(
            model=embedding_model,
            input=[question]
        ).data[0].embedding

    except Exception as e:

        logger.error(f"Embedding error: {e}")

        return "Embedding Error"

    matches = search_faiss(
        faiss_index,
        q_embedding,
        chunks,
        top_k=top_k
    )

    if not matches:
        return "Not found"

    prompt = build_prompt(
        question,
        matches
    )

    try:

        response = oa_client.chat.completions.create(
            model=chat_model,
            messages=[
                {
                    "role": "system",
                    "content":
                    "You are a concise extraction assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2
        )

        return response.choices[0].message.content.strip()

    except Exception as e:

        logger.error(f"Chat error: {e}")

        return "Chat Error"

# ---------------------------------------------------
# PROCESS SINGLE PDF
# ---------------------------------------------------

def process_single_pdf(
    pdf_bytes,
    filename,
    embedding_model,
    chat_model
):

    logger.info(f"Processing: {filename}")

    chunks = extract_pdf_text_chunks(pdf_bytes)

    if not chunks:

        return {
            "filename": filename,
            "qas": []
        }

    texts = [c["text"] for c in chunks]

    embeddings = embed_texts(
        texts,
        embedding_model
    )

    faiss_index = create_faiss_index(
        embeddings
    )

    qas = []

    for question in HARDCODED_QUESTIONS:

        answer = generate_answer(
            faiss_index,
            chunks,
            question,
            embedding_model,
            chat_model
        )

        qas.append({
            "question": question,
            "answer": answer
        })

    return {
        "filename": filename,
        "qas": qas
    }

# ---------------------------------------------------
# INGEST + ANSWER
# ---------------------------------------------------

@app.post("/ingest")
async def ingest_pdfs(
    pdfs: List[UploadFile] = File(...)
):

    results = []

    for pdf in pdfs:

        if not pdf.filename.lower().endswith(".pdf"):
            continue

        pdf_bytes = await pdf.read()

        result = process_single_pdf(
            pdf_bytes,
            pdf.filename,
            DEFAULT_EMBED_MODEL,
            DEFAULT_CHAT_MODEL
        )

        results.append(result)

    return {
        "total_files": len(results),
        "results": results
    }

# ---------------------------------------------------
# EXPORT EXCEL
# ---------------------------------------------------

@app.post("/export-excel")
async def export_excel(
    pdfs: List[UploadFile] = File(...)
):

    excel_rows = []

    for pdf in pdfs:

        if not pdf.filename.lower().endswith(".pdf"):
            continue

        logger.info(f"Excel processing: {pdf.filename}")

        pdf_bytes = await pdf.read()

        result = process_single_pdf(
            pdf_bytes,
            pdf.filename,
            DEFAULT_EMBED_MODEL,
            DEFAULT_CHAT_MODEL
        )

        row = {
            "filename": result["filename"]
        }

        for qa in result["qas"]:

            row[qa["question"]] = qa["answer"]

        excel_rows.append(row)

    df = pd.DataFrame(excel_rows)

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            index=False
        )

    output.seek(0)

    return StreamingResponse(
        output,
        media_type=(
            "application/vnd.openxmlformats-officedocument"
            ".spreadsheetml.sheet"
        ),
        headers={
            "Content-Disposition":
            "attachment; filename=answers.xlsx"
        }
    )

# ---------------------------------------------------
# HEALTH
# ---------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok"
    }

# ---------------------------------------------------
# FRONTEND
# ---------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def serve_frontend():

    file_path = Path(__file__).parent / "rag.html"

    if not file_path.exists():
        return """
        <h1>FAISS PDF Extraction API</h1>
        <p>Backend running successfully.</p>
        """

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        return f.read()