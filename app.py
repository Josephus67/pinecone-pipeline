from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pypdf import PdfReader
from pinecone import Pinecone
import os
import re
import time
import uuid
from typing import Optional
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI(title="Pinecone Integrated API — DISAL AI Tutor")

PINECONE_API_KEY  = os.getenv("PINECONE_API_KEY")
PINECONE_HOST     = os.getenv("PINECONE_HOST")
PINECONE_INDEX    = "disal"

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not set in environment.")

pc = Pinecone(api_key=PINECONE_API_KEY)

# CORS
origins = [
    "http://localhost:3000",      
    "http://localhost:5173",     
    "https://ai-tutor-admin-1.onrender.com", # Your production frontend
]

# 2. Add the CORS middleware to your FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,           # Allows specific origins
    allow_credentials=True,          # Allows cookies/auth headers
    allow_methods=["*"],             # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],             # Allows all custom headers
)



# ─────────────────────────────────────────────────────────────────────────────
# EDUCATION-AWARE CHUNKER
# ─────────────────────────────────────────────────────────────────────────────

# Heading patterns found in lecture notes / textbooks
HEADING_PATTERNS = [
    r"^(Chapter|Section|Unit|Module|Lecture|Topic|Part)\s+\d+",   # Chapter 1, Section 2
    r"^\d+\.\d*\s+[A-Z]",                                          # 1.1 Introduction
    r"^[A-Z][A-Z\s]{4,}$",                                         # ALL CAPS HEADING
    r"^(Introduction|Overview|Summary|Conclusion|Background|Objectives?|References?)",
]
HEADING_RE = re.compile("|".join(HEADING_PATTERNS), re.MULTILINE | re.IGNORECASE)


def extract_text_from_pdf(file) -> tuple[str, list[dict]]:
    """
    Extract text from PDF page by page.
    Returns full text and a list of page dicts {page_num, text}.
    """
    reader = PdfReader(file)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page_num": i + 1, "text": text})
    full_text = "\n".join(p["text"] for p in pages)
    return full_text, pages


def detect_section_boundaries(text: str) -> list[int]:
    """
    Find character positions of section headings in the text.
    These become natural split points for semantic chunking.
    """
    boundaries = [0]
    for match in HEADING_RE.finditer(text):
        pos = match.start()
        # Only add if meaningfully far from previous boundary (avoid micro-sections)
        if pos - boundaries[-1] > 200:
            boundaries.append(pos)
    boundaries.append(len(text))
    return boundaries


def education_aware_chunk(
    full_text: str,
    teaching_chunk_size: int = 1800,
    qa_chunk_size: int = 500,
    overlap: int = 100,
) -> list[dict]:
    """
    Produces two types of chunks from the same source text:

    - "teaching" chunks (1800 chars): large, concept-complete segments for
      lesson delivery. Sized so the AI can speak ~2-3 minutes per chunk.

    - "qa" chunks (500 chars): small, precise segments for Q&A retrieval.
      Sized for accurate needle-in-haystack search.

    Both types respect section boundaries so concepts are never split mid-topic.
    Each chunk carries rich metadata for filtering and context.
    """
    boundaries = detect_section_boundaries(full_text)
    sections = []

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end   = boundaries[i + 1]
        section_text = full_text[start:end].strip()
        if len(section_text) < 50:   # skip near-empty sections
            continue

        # Try to infer section title from the first non-empty line
        first_line = next(
            (ln.strip() for ln in section_text.splitlines() if ln.strip()),
            f"Section {i + 1}"
        )
        section_title = first_line[:120]   # cap at 120 chars

        sections.append({
            "index": i,
            "title": section_title,
            "text":  section_text,
        })

    chunks = []
    base_id = str(uuid.uuid4())
    chunk_counter = 0

    for section in sections:
        section_text  = section["text"]
        section_title = section["title"]
        section_idx   = section["index"]

        # ── Teaching chunks (large) ──────────────────────────────
        teaching_chunks = split_with_overlap(section_text, teaching_chunk_size, overlap)
        for j, tc in enumerate(teaching_chunks):
            chunks.append({
                "id":              f"{base_id}_teach_{chunk_counter}",
                "text":      tc,
                "chunk_type":      "teaching",
                "section_title":   section_title,
                "section_index":   section_idx,
                "chunk_index":     j,
                "char_count":      len(tc),
            })
            chunk_counter += 1

        # ── Q&A chunks (small) ───────────────────────────────────
        qa_chunks = split_with_overlap(section_text, qa_chunk_size, overlap)
        for j, qc in enumerate(qa_chunks):
            chunks.append({
                "id":              f"{base_id}_qa_{chunk_counter}",
                "text":      qc,
                "chunk_type":      "qa",
                "section_title":   section_title,
                "section_index":   section_idx,
                "chunk_index":     j,
                "char_count":      len(qc),
            })
            chunk_counter += 1

    return chunks


def split_with_overlap(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Sentence-boundary-aware splitter with overlap.
    Prefers splitting at sentence ends (.) rather than mid-sentence.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        # Walk back to find the nearest sentence boundary
        boundary = end
        for i in range(end, max(start + chunk_size // 2, start), -1):
            if text[i] in ".!?\n":
                boundary = i + 1
                break

        chunk = text[start:boundary].strip()
        if chunk:
            chunks.append(chunk)

        # Next chunk starts with overlap
        start = max(boundary - overlap, start + 1)

    return [c for c in chunks if len(c) > 30]   # drop tiny fragments


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/ingest-pdf")
async def ingest_pdf(
    namespace:           str        = Form(...),
    course_title:        str        = Form(...),   # for display/logging only
    file:                UploadFile = File(...),
    clear_existing:      bool       = Form(False),    # wipe old vectors for this namespace first
):
    """
    Ingest a PDF with education-aware chunking.
    Stores two chunk types per section: 'teaching' (large) and 'qa' (small).

    - namespace: course title (must match exactly what's used in /retrieve)
    - clear_existing: set True when re-uploading updated course materials
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        full_text, pages = extract_text_from_pdf(file.file)

        if not full_text.strip():
            raise HTTPException(status_code=400, detail="No extractable text found in the PDF.")

        index = pc.Index(host=PINECONE_HOST or PINECONE_INDEX)

        # Optionally clear old vectors for this namespace before re-ingesting
        if clear_existing:
            try:
                index.delete(delete_all=True, namespace=namespace)
                time.sleep(1)   # let deletion propagate
            except Exception as e:
                print(f"Warning: could not clear namespace '{namespace}': {e}")

        # Chunk the text
        chunks = education_aware_chunk(full_text)

        if not chunks:
            raise HTTPException(status_code=422, detail="Could not extract any usable content from the PDF.")

        # Build Pinecone records
        records = [
            {
                "id":              chunk["id"],
                "text":      chunk["text"],
                "chunk_type":      chunk["chunk_type"],
                "section_title":   chunk["section_title"],
                "section_index":   chunk["section_index"],
                "chunk_index":     chunk["chunk_index"],
                "char_count":      chunk["char_count"],
                "source_filename": file.filename,
                "page_count":      len(pages),
            }
            for chunk in chunks
        ]

        # Upsert in batches of 96
        batch_size = 96
        batches_upserted = 0
        for i in range(0, len(records), batch_size):
            index.upsert_records(
                records=records[i : i + batch_size],
                namespace=namespace,
            )
            batches_upserted += 1

        teaching_count = sum(1 for c in chunks if c["chunk_type"] == "teaching")
        qa_count       = sum(1 for c in chunks if c["chunk_type"] == "qa")
        sections_found = len(set(c["section_index"] for c in chunks))

        return {
            "message":          "Ingestion successful",
            "namespace":        namespace,
            "course_title":     course_title,
            "filename":         file.filename,
            "pages_processed":  len(pages),
            "sections_found":   sections_found,
            "teaching_chunks":  teaching_count,
            "qa_chunks":        qa_count,
            "total_chunks":     len(chunks),
            "batches_upserted": batches_upserted,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        file.file.close()


@app.get("/retrieve")
async def retrieve(
    namespace:  str,
    query:      str,
    top_k:      int = 4,
    chunk_type: Optional[str] = None,   # "teaching" | "qa" | None (both)
):
    """
    Retrieve relevant chunks from Pinecone.

    - chunk_type=qa       for precise Q&A retrieval (top_k=4 default)
    - chunk_type=teaching for lesson content delivery (top_k=6 recommended)
    - chunk_type omitted  returns both types (used for lesson plan generation)

    The backend gateway calls this with top_k=20 for lesson plan generation.
    """
    try:
        index = pc.Index(host=PINECONE_HOST)

        # Build filter if chunk_type requested
        query_params = {
            "inputs": {"text": query},
            "top_k":  top_k,
        }
        if chunk_type in ("teaching", "qa"):
            query_params["filter"] = {"chunk_type": {"$eq": chunk_type}}

        results = index.search(
            namespace=namespace,
            query=query_params,
        )

        raw = results.to_dict() if hasattr(results, "to_dict") else dict(results)
        hits = raw.get("results", {}).get("result", {}).get("hits", [])

        # Return structured hit list for easier consumption by the backend
        clean_hits = [
            {
                "text":          hit.get("fields", {}).get("text", ""),
                "chunk_type":    hit.get("fields", {}).get("chunk_type", ""),
                "section_title": hit.get("fields", {}).get("section_title", ""),
                "score":         round(hit.get("_score", 0), 4),
            }
            for hit in hits
            if hit.get("fields", {}).get("chunk_text")
        ]

        return {
            "query":      query,
            "namespace":  namespace,
            "chunk_type": chunk_type,
            "count":      len(clean_hits),
            "hits":       clean_hits,
            # Keep raw for backward compatibility with existing backend gateway code
            "results":    raw,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/namespace/{namespace}")
async def clear_namespace(namespace: str):
    """
    Delete all vectors for a namespace (course).
    Used when a lecturer deletes or fully replaces course content.
    """
    try:
        index = pc.Index(host=PINECONE_HOST)
        index.delete(delete_all=True, namespace=namespace)
        return {"message": f"Namespace '{namespace}' cleared successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/namespace/{namespace}/stats")
async def namespace_stats(namespace: str):
    """
    Returns chunk counts per type for a namespace.
    Useful for the admin app to confirm ingestion worked correctly.
    """
    try:
        index = pc.Index(host=PINECONE_HOST)
        stats = index.describe_index_stats()
        ns_stats = stats.get("namespaces", {}).get(namespace, {})
        return {
            "namespace":    namespace,
            "vector_count": ns_stats.get("vector_count", 0),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def health():
    return {"status": "healthy", "service": "DISAL Pinecone Pipeline"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
