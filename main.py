import os
import io
import uuid
import math
from typing import List, Dict, Any, Optional
import time
import logging
import pdfplumber
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi import HTTPException
from pathlib import Path
from fastapi.responses import HTMLResponse
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
load_dotenv()

# ------------ LOGGING ------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag")

# -------- ENV --------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
print(OPENAI_API_KEY)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
DEFAULT_INDEX_NAME = os.getenv("PINECONE_INDEX", "rag-index")

DEFAULT_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
DEFAULT_CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-mini")

EMBED_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}

EMBED_ALIASES = {
    "small": "text-embedding-3-small",
    "large": "text-embedding-3-large",
    "text-embedding-3-small": "text-embedding-3-small",
    "text-embedding-3-large": "text-embedding-3-large",
}

if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in environment.")
if not PINECONE_API_KEY:
    raise RuntimeError("Set PINECONE_API_KEY in environment.")

oa_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

app = FastAPI(title="Retrieval augmented generation")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------ SCHEMAS ------------
class QueryRequest(BaseModel):
    query: str = Field(..., examples=["What is the listing date on the IPO?"])
    top_k: int = Field(6, ge=1, le=50)
    index_name: Optional[str] = Field(None, examples=[DEFAULT_INDEX_NAME])
    embedding_model: Optional[str] = Field(
        None,
        description="Use 'text-embedding-3-small' | 'text-embedding-3-large' (or aliases 'small'|'large').",
        examples=[DEFAULT_EMBED_MODEL],
    )
    chat_model: Optional[str] = Field(None, examples=[DEFAULT_CHAT_MODEL])

class IngestResponse(BaseModel):
    index_name: str
    doc_id: str
    chunks: int
    embedding_model: str
    dimension: int

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

# batch models
class BatchQueryItem(BaseModel):
    query: str
    top_k: int = 6
    index_name: Optional[str] = None
    embedding_model: Optional[str] = None
    chat_model: Optional[str] = None

class BatchQueryResponse(BaseModel):
    results: List[QueryResponse]


# ------------ UTILS ------------
def normalize_embed_model(maybe: Optional[str]) -> str:
    if not maybe:
        return DEFAULT_EMBED_MODEL
    m = str(maybe).strip().lower()
    if m in {"string", "none", "null", ""}:
        return DEFAULT_EMBED_MODEL
    return EMBED_ALIASES.get(m, DEFAULT_EMBED_MODEL if m not in EMBED_DIMS else m)


def ensure_index(index_name: str, embedding_model: str):
    if embedding_model not in EMBED_DIMS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported embedding model '{embedding_model}'. Use one of: {list(EMBED_DIMS.keys())}",
        )

    required_dim = EMBED_DIMS[embedding_model]
    existing = {idx["name"] for idx in pc.list_indexes()}
    if index_name in existing:
        desc = pc.describe_index(index_name)
        try:
            actual_dim = int(getattr(desc, "dimension"))
        except Exception:
            try:
                actual_dim = int(desc.to_dict().get("dimension"))
            except Exception:
                actual_dim = None

        if actual_dim is not None and actual_dim != required_dim:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Index '{index_name}' exists with dimension {actual_dim}, "
                    f"but embedding model '{embedding_model}' requires {required_dim}."
                ),
            )
        logger.info(f"Index '{index_name}' already exists and is compatible.")
    else:
        logger.info(f"Creating index '{index_name}' with dim {required_dim}...")
        pc.create_index(
            name=index_name,
            dimension=required_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        logger.info(f"Index '{index_name}' created.")
    return pc.Index(index_name), required_dim


def extract_pdf_text_chunks(pdf_bytes: bytes,
                            max_tokens: int = 800,
                            overlap: int = 150) -> List[Dict[str, Any]]:
    approx_chars = max_tokens * 4
    approx_overlap = overlap * 4
    chunks: List[Dict[str, Any]] = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        logger.info(f"PDF opened. Total pages: {len(pdf.pages)}")
        for pageno, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                words = page.extract_words() or []
                if words:
                    text = " ".join(w.get("text", "") for w in words).strip()

            if not text:
                continue

            start = 0
            n = len(text)
            while start < n:
                end = min(start + approx_chars, n)
                snippet = text[start:end].strip()
                if snippet:
                    chunks.append({"page": pageno, "text": snippet})
                    # log every 100 chunks
                    if len(chunks) % 100 == 0:
                        logger.info(f"chunking... {len(chunks)}")
                if end >= n:
                    break
                start = max(0, end - approx_overlap)

    logger.info(f"chunking done. total chunks: {len(chunks)}")
    return chunks


def embed_texts(texts: List[str], model: str) -> List[List[float]]:
    out: List[List[float]] = []
    batch_size = 128
    total = len(texts)
    total_batches = math.ceil(total / batch_size)
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        batch_no = i // batch_size + 1
        logger.info(f"embedding batch {batch_no}/{total_batches} (size {len(batch)})...")
        resp = oa_client.embeddings.create(model=model, input=batch)
        out.extend([d.embedding for d in resp.data])
    logger.info("embedding done.")
    return out


def upsert_chunks_to_pinecone(index, doc_id: str, chunks: List[Dict[str, Any]],
                              embeddings: List[List[float]]):
    vectors = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": f"{doc_id}-{i}",
            "values": emb,
            "metadata": {
                "doc_id": doc_id,
                "page": chunk["page"],
                "text": chunk["text"][:4000],
            }
        })

    batch = 100
    total = len(vectors)
    total_batches = math.ceil(total / batch)
    for i in range(0, total, batch):
        batch_no = i // batch + 1
        logger.info(f"upserting vectors {batch_no}/{total_batches} ...")
        index.upsert(vectors=vectors[i:i + batch])
    logger.info("upsert done.")


def search(index, query_embedding: List[float], top_k: int = 6):
    res = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_values=False,
        include_metadata=True,
    )
    matches = getattr(res, "matches", []) or []
    return [
        {
            "id": m.id,
            "score": getattr(m, "score", None),
            "text": (m.metadata or {}).get("text", ""),
            "page": (m.metadata or {}).get("page"),
            "doc_id": (m.metadata or {}).get("doc_id"),
        }
        for m in matches
    ]


def build_prompt(query: str, contexts: List[Dict[str, Any]]) -> str:
    bullets = []
    for c in contexts:
        page = c.get("page")
        text = c.get("text", "")
        bullets.append(f"(p.{page}) {text}")
    context_block = "\n\n".join(bullets[:10])
    return (
        "Answer strictly from the provided context. If the answer is not present, say you don't have enough information.\n\n"
        f"Question:\n{query}\n\n"
        f"Context:\n{context_block}\n\n"
        "Answer concisely and cite page numbers inline like [p.X]."
    )


def safe_clear_index(index, namespace: str = "") -> None:
    try:
        stats = index.describe_index_stats()
        ns_total = (
            stats.get("namespaces", {})
                 .get(namespace or "", {})
                 .get("vector_count", 0)
        )
        if ns_total and ns_total > 0:
            logger.info(f"clearing namespace '{namespace}' with {ns_total} vectors...")
            index.delete(delete_all=True, namespace=namespace or "")
            logger.info("namespace cleared.")
        else:
            logger.info("namespace empty, nothing to clear.")
    except Exception as e:
        msg = str(e)
        if ("Namespace not found" in msg) or ("404" in msg):
            logger.warning("namespace not found, continuing.")
            return
        raise HTTPException(status_code=500, detail=f"Failed to clear index: {e}")


# ------------ ENDPOINTS ------------
@app.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(
    pdf: UploadFile = File(...),
    index_name: str = Form(DEFAULT_INDEX_NAME),
    embedding_model: str = Form(DEFAULT_EMBED_MODEL),
):
    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file.")

    embedding_model = normalize_embed_model(embedding_model)
    logger.info(f"/ingest called. index={index_name}, embed_model={embedding_model}")

    index, dim = ensure_index(index_name, embedding_model)

    safe_clear_index(index, namespace="")

    content = await pdf.read()
    chunks = extract_pdf_text_chunks(content, max_tokens=800, overlap=150)
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No text extracted. Likely a scanned PDF. Run OCR (e.g., Tesseract) before ingestion."
        )

    doc_id = str(uuid.uuid4())
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts, model=embedding_model)
    upsert_chunks_to_pinecone(index, doc_id, chunks, embeddings)

    logger.info(f"ingest complete. doc_id={doc_id}, chunks={len(chunks)}")

    return IngestResponse(
        index_name=index_name,
        doc_id=doc_id,
        chunks=len(chunks),
        embedding_model=embedding_model,
        dimension=dim,
    )


@app.post("/query", response_model=QueryResponse)
async def query_knowledge(req: QueryRequest):
    index_name = (req.index_name or DEFAULT_INDEX_NAME).strip() or DEFAULT_INDEX_NAME
    embedding_model = normalize_embed_model(req.embedding_model)
    chat_model = (req.chat_model or DEFAULT_CHAT_MODEL).strip() or DEFAULT_CHAT_MODEL

    logger.info(f"/query called. q='{req.query[:60]}...' index={index_name}")

    index, _ = ensure_index(index_name, embedding_model)

    try:
        q_emb = oa_client.embeddings.create(model=embedding_model, input=[req.query]).data[0].embedding
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding error: {e}")

    try:
        matches = search(index, q_emb, top_k=req.top_k)
    except Exception as e:
        logger.error(f"Pinecone query error: {e}")
        raise HTTPException(status_code=500, detail=f"Pinecone query error: {e}")

    if not matches:
        logger.info("no matches found.")
        return QueryResponse(
            answer="I don't have enough information to answer from the knowledge base.",
            sources=[]
        )

    prompt = build_prompt(req.query, matches)

    try:
        chat = oa_client.chat.completions.create(
            model=chat_model,
            messages=[
                {"role": "system", "content": "You are a concise expert assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        answer = chat.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat error: {e}")

    sources = [{"id": m["id"], "page": m["page"], "score": m["score"], "doc_id": m["doc_id"]} for m in matches]
    logger.info("query answered.")
    return QueryResponse(answer=answer, sources=sources)


@app.post("/query/batch", response_model=BatchQueryResponse)
async def batch_query(payload: List[BatchQueryItem]):
    logger.info(f"/query/batch called. total_queries={len(payload)}")
    results: List[QueryResponse] = []

    for idx, item in enumerate(payload, start=1):
        logger.info(f"processing query {idx}/{len(payload)}: '{item.query[:50]}...'")
        index_name = (item.index_name or DEFAULT_INDEX_NAME).strip() or DEFAULT_INDEX_NAME
        embedding_model = normalize_embed_model(item.embedding_model)
        chat_model = (item.chat_model or DEFAULT_CHAT_MODEL).strip() or DEFAULT_CHAT_MODEL

        try:
            index, _ = ensure_index(index_name, embedding_model)

            q_emb = oa_client.embeddings.create(model=embedding_model, input=[item.query]).data[0].embedding
            matches = search(index, q_emb, top_k=item.top_k)

            if not matches:
                logger.info(f"query {idx}: no matches found.")
                results.append(
                    QueryResponse(answer="I don't have enough information to answer from the knowledge base.", sources=[])
                )
                continue

            prompt = build_prompt(item.query, matches)

            chat = oa_client.chat.completions.create(
                model=chat_model,
                messages=[
                    {"role": "system", "content": "You are a concise expert assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
            )
            answer = chat.choices[0].message.content.strip()
            sources = [
                {"id": m["id"], "page": m["page"], "score": m["score"], "doc_id": m["doc_id"]}
                for m in matches
            ]
            results.append(QueryResponse(answer=answer, sources=sources))
            logger.info(f"query {idx} answered.")
        except Exception as e:
            logger.error(f"Error in query {idx}: {e}")
            results.append(
                QueryResponse(
                    answer=f"Error processing this query: {e}",
                    sources=[]
                )
            )

    logger.info("batch query complete.")
    return BatchQueryResponse(results=results)


@app.get("/debug")
def debug(index_name: str = DEFAULT_INDEX_NAME):
    if index_name not in {idx["name"] for idx in pc.list_indexes()}:
        return {"index": index_name, "exists": False}

    desc = pc.describe_index(index_name)
    idx = pc.Index(index_name)
    stats = idx.describe_index_stats()
    total = int(stats.get("total_vector_count", 0))

    sample_query = "test"
    try:
        q_emb = oa_client.embeddings.create(model=DEFAULT_EMBED_MODEL, input=[sample_query]).data[0].embedding
        test_matches = search(idx, q_emb, top_k=1)
    except Exception as e:
        test_matches = [{"error": str(e)}]

    return {
        "index": index_name,
        "exists": True,
        "dimension": int(getattr(desc, "dimension", -1)),
        "metric": getattr(desc, "metric", None),
        "total_vectors": total,
        "test_query_matches": test_matches,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    file_path = Path(__file__).parent / "rag.html"
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
