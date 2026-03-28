from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader
from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import time
import uuid
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Pinecone Integrated API")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "integrated-disal"

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables. Please check your .env file.")

# Initialize global Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)


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


@app.get('/')
def health():
    return {"Health status" : "pinecone is healthy"}

if __name__ == "__main__":
    import uvicorn
    # Render sets the PORT environment variable dynamically
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
