import os
import shutil
import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader

import os

# 1. Prevent PyTorch from spawning too many CPU threads and deadlocking
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import shutil
import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 2. Explicitly force 'cpu' so it doesn't hang trying to probe broken GPU drivers
# 3. Lower the batch size so it processes chunks safely without overloading RAM
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    device="cuda",
    embed_batch_size=10
)

# ... [the rest of your ingest_and_archive function remains exactly the same] ...

# Use a fast, free local model for vector embeddings instead of an API
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

def ingest_and_archive(
    upload_dir: str = "./uploads", 
    archive_dir: str = "./archive", 
    collection_name: str = "adhoc_documents"
):
    for directory in [upload_dir, archive_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            
    print(f"Scanning '{upload_dir}' for PDFs...")
    try:
        # Delete commented out code later 
        #documents = SimpleDirectoryReader(input_dir=upload_dir, required_exts=[".pdf"]).load_data()
        
        print("Scanning './uploads' for PDFs using PyMuPDF...")

        # Force LlamaIndex to use PyMuPDF for all PDFs
        parser = PyMuPDFReader()
        file_extractor = {".pdf": parser}

        documents = SimpleDirectoryReader(
            input_dir="./uploads",
            file_extractor=file_extractor
        ).load_data()

    except ValueError:
        print(f"No new PDFs found in '{upload_dir}'.")
        return
        
    print(f"Extracted {len(documents)} chunks. Running local embeddings...")

    print("Connecting to local database...")
    db_path = "./chroma_db"
    chroma_client = chromadb.PersistentClient(path=db_path)
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    print("Database connected successfully!")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Embedding chunks and saving to database...")
    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context,
        show_progress=True # <--- This will show a loading bar for the chunks
    )
    print("Embedding complete!")
    
    # Archive
    processed_files = set(doc.metadata["file_path"] for doc in documents)
    print("\nArchiving processed files:")
    for file_path in processed_files:
        filename = os.path.basename(file_path)
        shutil.move(file_path, os.path.join(archive_dir, filename))
        print(f"Moved -> {filename}")
    
    print("\n✅ Ingestion complete! Database updated.")

if __name__ == "__main__":
    ingest_and_archive()