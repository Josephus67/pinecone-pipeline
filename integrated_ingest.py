import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_NAME = "integrated-disal" 

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables. Please check your .env file.")

def ingest_docs():
    # Initialize Pinecone Client
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Check if index exists, create if not
    print("Checking Pinecone index...")
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        print(f"Creating new index: {INDEX_NAME} with integrated embeddings")
        try:
            pc.create_index_for_model(
                name=INDEX_NAME,
                cloud="aws",
                region="us-east-1",
                embed={
                    "model": "llama-text-embed-v2",
                    "field_map": {"text": "chunk_text"}
                }
            )
            # Wait for index to be ready
            while not pc.describe_index(INDEX_NAME).status["ready"]:
                time.sleep(1)
            print(f"Index {INDEX_NAME} created successfully.")
        except Exception as e:
            print(f"Error creating index: {e}")
            raise
    else:
        print(f"Index {INDEX_NAME} already exists.")

    # Get the index
    index = pc.Index(INDEX_NAME)

    # Load documents from the static directory
    print("Loading documents from static/...")
    # using glob to find all .txt files
    # Note: DirectoryLoader can be finicky without unstructured, but TextLoader helps.
    loader = DirectoryLoader("./static", glob="**/*.txt", loader_cls=TextLoader)
    try:
        documents = loader.load()
        print(f"Loaded {len(documents)} documents.")
    except Exception as e:
        print(f"Error loading documents: {e}")
        return

    if not documents:
        print("No documents found in static/ folder.")
        return

    # Split documents into chunks
    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
    splits = text_splitter.split_documents(documents)
    print(f"Created {len(splits)} chunks.")

    # Prepare records for upsert
    print("Preparing records for upsert...")
    records = []
    for i, split in enumerate(splits):
        records.append({
            "id": f"chunk_{i}",
            "chunk_text": split.page_content
        })

    # Upsert to Pinecone in batches
    print(f"Upserting {len(records)} records to index '{INDEX_NAME}' in batches...")
    batch_size = 96
    try:
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            index.upsert_records(records=batch, namespace="__default__")
            print(f"Upserted batch {i // batch_size + 1} of {(len(records) + batch_size - 1) // batch_size}")
        print("Ingestion complete! Vector database is ready.")
    except Exception as e:
        print(f"Error upserting into Pinecone: {e}")

if __name__ == "__main__":
    ingest_docs()