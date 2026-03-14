from pinecone import Pinecone
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# To get the unique host for an index, 
# see https://docs.pinecone.io/guides/manage-data/target-an-index
index = pc.Index(host=os.getenv("PINECONE_HOST"))

results = index.search(
    namespace="__default__", 
    query={
        "inputs": {"text": "What is supervised machine learning?"}, 
        "top_k": 1
    },
    #fields=["category", "chunk_text"]
)

print(results)