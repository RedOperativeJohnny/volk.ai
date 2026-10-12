import os
import shutil
import chromadb

# LlamaIndex tools:
# These help us read files, convert text into searchable math, and connect to our database.
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader

# ==============================================================================
# 1. System Performance & Safety Settings
# ==============================================================================
# When computers run heavy calculations, they try to split the work across multiple
# CPU cores. Sometimes these background processes get stuck waiting on each other (deadlock).
# Setting these environment variables forces the system to run cleanly one step at a time
# and prevents computer freeze-ups or crashes.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==============================================================================
# 2. Setting Up the "Translator" (The Embedding Model)
# ==============================================================================
# Computers cannot read or understand English words; they only understand numbers.
# An "embedding model" acts like a translator. It reads sentences and turns them into
# long lists of numbers (called vectors). Sentences with similar meanings get turned into
# similar number patterns, which is how the computer later searches by meaning rather
# than just matching exact keywords.
#
# - model_name: The specific, free translation dictionary we downloaded ("bge-small-en-v1.5").
# - device: "cuda" tells it to use your Nvidia graphics card (GPU) for speed. If you don't have one, change this to "cpu".
# - embed_batch_size: How many text chunks to translate at once. Keeping it small (10) prevents your computer from running out of memory.
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    device="cuda",
    embed_batch_size=10
)

# ==============================================================================
# 3. Main Processing Function: Ingest and Archive
# ==============================================================================
def ingest_and_archive(
    upload_dir: str = "./uploads", 
    archive_dir: str = "./archive", 
    collection_name: str = "adhoc_documents"
):
    # Step A: Make sure our folders exist on your hard drive.
    # If 'uploads' or 'archive' folders are missing, create them automatically.
    for directory in [upload_dir, archive_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            
    print(f"Scanning '{upload_dir}' for PDFs using PyMuPDF...")
    try:
        # Step B: Read the PDF files.
        # Standard PDF readers can be slow or choke on complex layouts. 
        # We explicitly use 'PyMuPDF', which is an engine that extracts clean text fast.
        parser = PyMuPDFReader()
        file_extractor = {".pdf": parser}

        # SimpleDirectoryReader goes into the 'uploads' folder and extracts all the text
        # out of any PDF files it finds using our chosen parser.
        documents = SimpleDirectoryReader(
            input_dir=upload_dir,
            file_extractor=file_extractor
        ).load_data()

    except ValueError:
        # If the folder has no PDFs in it, stop gracefully so the script doesn't crash.
        print(f"No new PDFs found in '{upload_dir}'.")
        return
        
    print(f"Extracted {len(documents)} chunks. Running local embeddings...")

    # Step C: Connect to the database on your disk.
    # ChromaDB is a specialized database (a "vector database") designed to store text 
    # alongside its numerical translations so it can search through them instantly later.
    print("Connecting to local database...")
    db_path = "./chroma_db"
    
    # PersistentClient ensures our data is saved to a permanent folder on disk,
    # not just temporarily kept in RAM where it would vanish when the script closes.
    chroma_client = chromadb.PersistentClient(path=db_path)
    
    # A "collection" is like a table or folder inside the database to keep these documents organized.
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    print("Database connected successfully!")

    # Step D: Hook the database into LlamaIndex.
    # These two lines connect the storage system so LlamaIndex knows where to write the data.
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Step E: Translate (embed) the text and save it.
    # This takes the text we extracted from your PDFs, feeds it through the translation model
    # to calculate the numbers, and saves both the text and numbers into ChromaDB.
    # 'show_progress=True' prints a progress bar in your terminal.
    print("Embedding chunks and saving to database...")
    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context,
        show_progress=True
    )
    print("Embedding complete!")
    
    # Step F: Clean up and archive.
    # Now that the text is safely stored in our database, we don't want to re-process 
    # the same PDFs the next time we run the script. We move the finished PDF files 
    # out of 'uploads' and into the 'archive' folder.
    processed_files = set(doc.metadata["file_path"] for doc in documents)
    print("\nArchiving processed files:")
    for file_path in processed_files:
        filename = os.path.basename(file_path)
        shutil.move(file_path, os.path.join(archive_dir, filename))
        print(f"Moved -> {filename}")
    
    print("\n✅ Ingestion complete! Database updated.")

# ==============================================================================
# Script Entry Point
# ==============================================================================
# This standard Python line ensures the function runs when you type 'python ingest.py' in the terminal.
if __name__ == "__main__":
    ingest_and_archive()