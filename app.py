from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader
from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import json
import time
import uuid
from datetime import datetime, timezone
from dotenv import load_dotenv
from groq import Groq
from pymongo import MongoClient
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI(title="Pinecone Integrated API")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "integrated-disal"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MONGO_URI = os.getenv("MONGO_URI")
# DB name defaults to the one embedded in the connection string (e.g. .../DISAL).
MONGODB_DB = os.getenv("MONGODB_DB")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "lessons")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables. Please check your .env file.")

_pinecone_client = None

def get_pinecone_client():
    global _pinecone_client
    if _pinecone_client is None:
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not found in environment variables. Please check your .env file.")
        _pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
    return _pinecone_client

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

# Lazily-initialized clients (built on first use so missing keys only break the
# lesson-plan endpoint, not the whole app / other endpoints).
_groq_client = None
_mongo_client = None


def get_groq_client():
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY not set in environment.")
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


def get_lessons_collection():
    global _mongo_client
    if _mongo_client is None:
        if not MONGO_URI:
            raise HTTPException(status_code=500, detail="MONGO_URI not set in environment.")
        _mongo_client = MongoClient(MONGO_URI)
    db = _mongo_client[MONGODB_DB] if MONGODB_DB else _mongo_client.get_default_database()
    return db[MONGODB_COLLECTION]


@app.post("/ingest-pdf")
async def ingest_pdf(
    namespace: str = Form(...),
    file: UploadFile = File(...)
):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    try:
        # Read the incoming PDF file in-memory
        reader = PdfReader(file.file)
        text_content = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_content.append(text)
                
        full_text = "\n".join(text_content)
        
        if not full_text.strip():
            raise HTTPException(status_code=400, detail="No extractable text found in the PDF.")

        # Text chunking
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
        chunks = text_splitter.split_text(full_text)
        
        # Get the index
        pc = get_pinecone_client()
        index = pc.Index(PINECONE_INDEX_NAME)
        
        records = []
        base_id = str(uuid.uuid4())
        for i, chunk in enumerate(chunks):
            records.append({
                "id": f"{base_id}_chunk_{i}",
                "chunk_text": chunk
            })
            
        # Batch and upsert (e.g. 96 chunks per batch)
        batch_size = 96
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            index.upsert_records(records=batch, namespace=namespace)
            
        return {
            "message": "Ingestion successful",
            "index_name": PINECONE_INDEX_NAME,
            "namespace": namespace,
            "processed_chunks": len(records),
            "filename": file.filename
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup: close the uploaded file
        file.file.close()


class RetrievalRequest(BaseModel):
    namespace: str
    query: str
    top_k: int = 4

@app.get("/retrieve")
async def retrieve(namespace: str, query: str, top_k: int = 4):
    try:
        pc = get_pinecone_client()
        index = pc.Index(host=os.getenv("PINECONE_HOST"))

        results = index.search(
            namespace=namespace, 
            query={
                "inputs": {"text": query}, 
                "top_k": top_k
            }
        )
        
        return {
            "query": query,
            "index_name": PINECONE_INDEX_NAME,
            "namespace": namespace,
            # Pinecone returns a custom object that FastAPI struggles to serialize to JSON by default.
            # Converting it securely to a dictionary prevents the 500 Internal Server Error.
            "results": results.to_dict() if hasattr(results, "to_dict") else dict(results)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Lesson Plan Endpoint
class LessonPlanRequest(BaseModel):
    namespace: str


def fetch_namespace_text(namespace: str) -> str:
    """Pull every chunk stored under a namespace and concatenate it.

    Pinecone has no 'dump all vectors' call, so we page through the namespace's
    IDs with index.list() and fetch the stored chunk_text for each.
    """
    pc = get_pinecone_client()
    index = pc.Index(host=os.getenv("PINECONE_HOST"))

    ids = []
    for id_batch in index.list(namespace=namespace):
        ids.extend(id_batch)

    if not ids:
        raise HTTPException(
            status_code=404,
            detail=f"No content found in namespace '{namespace}'.",
        )

    chunks = []
    for i in range(0, len(ids), 100):
        batch = ids[i:i + 100]
        fetched = index.fetch(ids=batch, namespace=namespace)
        vectors = getattr(fetched, "vectors", None) or fetched.get("vectors", {})
        for vec in vectors.values():
            metadata = getattr(vec, "metadata", None)
            if metadata is None and isinstance(vec, dict):
                metadata = vec.get("metadata", {})
            metadata = metadata or {}
            text = metadata.get("chunk_text")
            if text:
                chunks.append(text)

    full_text = "\n\n".join(chunks)
    if not full_text.strip():
        raise HTTPException(
            status_code=404,
            detail=f"Namespace '{namespace}' has records but no chunk_text to summarize.",
        )
    return full_text


# LESSON_PLAN_SYSTEM_PROMPT = (
#     "You are an expert instructional designer. Given course material, you produce "
#     "a structured lesson plan as JSON. Respond with ONLY a JSON object of this exact shape:\n"
#     "{\n"
#     '  "topics": [\n'
#     "    {\n"
#     '      "title": "string - a concise topic title",\n'
#     '      "chunks": ["string", "string", "string"],\n'
#     '      "quiz": [\n'
#     "        {\n"
#     '          "question": "string",\n'
#     '          "options": ["A. ...", "B. ...", "C. ...", "D. ..."],\n'
#     '          "answer": "A"\n'
#     "        }\n"
#     "      ]\n"
#     "    }\n"
#     "  ]\n"
#     "}\n"
#     "Rules: Produce between 3 and 6 topics that together cover the material. "
#     "Each topic must have exactly 3 'chunks' (clear explanatory paragraphs grounded in the "
#     "provided material) and exactly 3 quiz questions. Each quiz question must have exactly 4 "
#     "options prefixed 'A. ', 'B. ', 'C. ', 'D. ', and 'answer' must be the single letter "
#     "(A, B, C, or D) of the correct option. Base everything strictly on the provided material."
# )


# def generate_lesson_plan(course_text: str) -> dict:
#     """Call Groq to turn raw course text into the lessonPlan topics structure."""
#     client = get_groq_client()
#     completion = client.chat.completions.create(
#         model=GROQ_MODEL,
#         response_format={"type": "json_object"},
#         messages=[
#             {"role": "system", "content": LESSON_PLAN_SYSTEM_PROMPT},
#             {"role": "user", "content": f"Course material:\n\n{course_text}"},
#         ],
#     )
#     raw = completion.choices[0].message.content
#     try:
#         parsed = json.loads(raw)
#     except json.JSONDecodeError:
#         raise HTTPException(status_code=502, detail="Groq returned invalid JSON.")

#     topics = parsed.get("topics")
#     if not isinstance(topics, list) or not topics:
#         raise HTTPException(status_code=502, detail="Groq response had no topics.")
#     return topics


# @app.post("/lesson-plan")
# async def lesson_plan(request: LessonPlanRequest):
#     try:
#         course_text = fetch_namespace_text(request.namespace)
#         topics = generate_lesson_plan(course_text)

#         lesson_plan_doc = {
#             "lessonPlan": {
#                 "topics": topics,
#                 "generatedAt": datetime.now(timezone.utc),
#             }
#         }

#         collection = get_lessons_collection()
#         result = collection.insert_one(lesson_plan_doc)

#         return {
#             "message": "Lesson plan generated and stored.",
#             "id": str(result.inserted_id),
#             "namespace": request.namespace,
#             "topic_count": len(topics),
#             "lessonPlan": {
#                 "topics": topics,
#                 "generatedAt": lesson_plan_doc["lessonPlan"]["generatedAt"].isoformat(),
#             },
#         }

#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting uvicorn directly on port {port}...", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port)
