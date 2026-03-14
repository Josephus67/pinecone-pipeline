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

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables. Please check your .env file.")

# Initialize global Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)


@app.post("/ingest-pdf")
async def ingest_pdf(
    index_name: str = Form(...),
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
        
        # Check if index exists, create if not
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
        
        if index_name not in existing_indexes:
            print(f"Creating new index: {index_name} with integrated embeddings")
            try:
                pc.create_index_for_model(
                    name=index_name,
                    cloud="aws",
                    region="us-east-1",
                    embed={
                        "model": "llama-text-embed-v2",
                        "field_map": {"text": "chunk_text"}
                    }
                )
                # Wait for index to be ready
                while not pc.describe_index(index_name).status["ready"]:
                    time.sleep(1)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error creating Pinecone index: {str(e)}")
        
        # Upsert Data to Pinecone
        index = pc.Index(index_name)
        
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
            index.upsert_records(records=batch, namespace="__default__")
            
        return {
            "message": "Ingestion successful",
            "index_name": index_name,
            "processed_chunks": len(records),
            "filename": file.filename
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup: close the uploaded file
        file.file.close()


class RetrievalRequest(BaseModel):
    index_name: str
    query: str
    top_k: int = 4

@app.post("/retrieve")
async def retrieve(request: RetrievalRequest):
    try:
        # Verify the index exists
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
        if request.index_name not in existing_indexes:
            raise HTTPException(status_code=404, detail=f"Index '{request.index_name}' not found.")
            
        index = pc.Index(request.index_name)
        
        # Integrated Inference API Search
        results = index.search(
            namespace="__default__", 
            query={
                "inputs": {"text": request.query}, 
                "top_k": request.top_k
            }
        )
        
        return {
            "query": request.query,
            "index_name": request.index_name,
            "results": results.get("result", results)  # Normalizing the output based on Pinecone dict response
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    # Render sets the PORT environment variable dynamically
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run("app:app", host="0.0.0.0", port=port)

