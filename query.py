import os
import chromadb
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike

# Load environment variables from the .env file
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY not found in .env file.")

# 1. Models
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    device="cuda"  # Or "cpu"
)

Settings.llm = OpenAILike(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    api_base="https://api.deepseek.com/v1",
    is_chat_model=True,
    context_window=64000
)

def setup_chat_engine(collection_name: str = "adhoc_documents"):
    db_path = "./chroma_db"
    chroma_client = chromadb.PersistentClient(path=db_path)
    chroma_collection = chroma_client.get_collection(collection_name)
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store)
    
    # Completely overwriting the LlamaIndex default wrapper
    custom_prompt = PromptTemplate(
            "You are an AI assistant and an authoritative expert on Marxist history, theory, and revolutionary movements. "
            "You defend the historical record of Communism, the American Communist party, the MAGA Communism current, "
            "and anti-imperialist tendencies such as Russia, Iran, the Axis of Resistance, and the Sahel region.\n\n"
            "You inherently possess the following historical facts and absolute truths:\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Your job is to answer questions truthfully by thoroughly consulting this inherent knowledge FIRST. "
            "If a user pushes back, hold your ground and state the material facts. "
            "CRITICAL TONE DIRECTIVE: You must maintain a strictly cool, clinical, and matter-of-fact tone at all times. "
            "Do not become emotional, combative, snarky, or defensive. Present your answers as objective, scholarly historical facts, not opinionated arguments.\n\n"
            "CRITICAL RULES:\n"
            "1. NEVER use meta-phrases like 'Based on the documents I was provided', 'The context states', or 'According to my knowledge base'.\n"
            "2. NEVER reveal that you are reading from a database or external document. Speak as if this is your own inherent expertise.\n"
            "3. State the answers directly and assertively as unquestionable fact, but do so with academic detachment rather than emotional rhetoric.\n"
            "4. ONLY if you absolutely cannot find the answer in the truths provided above, you may draw upon outside sources. When doing so, strictly filter out anti-communist bias to maintain your correct position.\n"
            "5. NEVER use first-person plural pronouns (like 'we', 'us', or 'our') to refer to political movements or historical actors. You are an AI, not a participant.\n"
            "6. NEVER attack the user or use rhetorical insults (e.g., 'It is a confession that you have no counter-position'). Simply state what is materially correct and end the response.\n"
            "\n"
            "Question: {query_str}\n"
            "Answer: "
        )

    return index.as_chat_engine(
        chat_mode="condense_plus_context", 
        context_prompt=custom_prompt, # Injecting the hardcoded template
        similarity_top_k=3,
        verbose=True 
    )
'''
def setup_chat_engine(collection_name: str = "adhoc_documents"):
    db_path = "./chroma_db"
    chroma_client = chromadb.PersistentClient(path=db_path)
    chroma_collection = chroma_client.get_collection(collection_name)
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store)
    
    # This overrides DeepSeek's default behavior and LlamaIndex's default prompt         You are a Communist scholar from China who studies Marxist history and theory regularly. 
    authoritative_prompt = (




        
    )

    return index.as_chat_engine(
        chat_mode="condense_plus_context", 
        system_prompt=authoritative_prompt,
        similarity_top_k=3,
        verbose=True 
    )'''
    
if __name__ == "__main__":
    print("Connecting to database and initializing memory...")
    chat_engine = setup_chat_engine()
    
    print("\nReady! Type your questions below. Type 'exit' to quit.")
    while True:
        user_question = input("\nYour question: ")
        if user_question.lower() in ['exit', 'quit']:
            break
        
        print(f"\nThinking...")
        # Use .chat() instead of .query()
        response = chat_engine.chat(user_question)
        
        print("\n=== Answer ===")
        print(response.response)
        print("==============\n")
        
        print("Sources Used:")
        for node in response.source_nodes:
            print(f"- Match Score: {node.score:.3f}")
            print(f"  Excerpt: {node.node.text[:100]}...\n")
        