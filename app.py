# ==============================================================================
# LIBRARIES & TOOLS
# ==============================================================================
import streamlit as st          # Builds the web application interface (buttons, chat boxes, sidebars)
import os                       # Interacts with the operating system (reading file paths, environment variables)
import sqlite3                  # Built-in lightweight SQL database engine to save chat history permanently
import uuid                     # Generates unique, random IDs for each new conversation
import chromadb                 # Vector database engine that stores and searches through our document excerpts
import zipfile
import boto3
from datetime import datetime   # Captures timestamps so chats can be ordered chronologically
from dotenv import load_dotenv  # Loads sensitive credentials (like API keys) from a hidden .env file
from typing import List         # Type hints to keep Python code clear and readable
from pydantic import PrivateAttr # Allows custom internal variables inside LlamaIndex classes without errors
from ddgs import DDGS           # DuckDuckGo search client to pull live news from the internet

# LlamaIndex components:
# The framework coordinating our database search, memory, and LLM text generation
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike

# ==============================================================================
# 1. Environment & API Key Setup
# ==============================================================================
# Reads the '.env' text file sitting in your project directory
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# If the password/key is missing, halt execution and display an error in the browser
if not DEEPSEEK_API_KEY:
    st.error("DEEPSEEK_API_KEY not found in .env file.")
    st.stop()

# Configure the browser tab title, favicon, and screen layout
st.set_page_config(page_title="Volk.ai", page_icon="☭", layout="wide")

# ==============================================================================
# 2. Local Database for Conversation History (SQLite)
# ==============================================================================
# This keeps track of all past user conversations across browser reloads.
# It functions like standard chat apps: a list of threads on the left, messages on the right.
DB_FILE = "conversations.db"

def init_db():
    """Initializes the database tables on disk if this is the first time running."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        # Table 1: Stores chat sessions (an ID, a title, and when it was created)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT,
                created_at TIMESTAMP
            )
        """)
        # Table 2: Stores individual messages belonging to each session
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                role TEXT,
                content TEXT,
                created_at TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)
        conn.commit()

init_db()

def get_conversations():
    """Fetches all chat threads ordered by newest first."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, title FROM conversations ORDER BY created_at DESC")
        return cursor.fetchall()

def create_conversation(title="New Conversation"):
    """Creates a new empty chat session with a unique ID."""
    conv_id = str(uuid.uuid4())
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO conversations (id, title, created_at) VALUES (?, ?, ?)",
            (conv_id, title, datetime.now())
        )
        conn.commit()
    return conv_id

def update_conversation_title(conv_id, title):
    """Updates the title of a chat thread (e.g., using the first question asked)."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conv_id))
        conn.commit()

def delete_conversation(conv_id):
    """Permanently deletes a chat session and all messages associated with it."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
        cursor.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        conn.commit()

def get_messages(conv_id):
    """Pulls all past messages belonging to a specific conversation."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (conv_id,)
        )
        return [{"role": r, "content": c} for r, c in cursor.fetchall()]

def save_message(conv_id, role, content):
    """Saves a single message (from user or AI) into the database."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (conv_id, role, content, datetime.now())
        )
        conn.commit()

# ==============================================================================
# 3. AI Models & Knowledge Base Loading
# ==============================================================================
# Streamlit re-executes the entire Python script top-to-bottom every time you click a button.
# '@st.cache_resource' tells Streamlit: "Load these heavy models once into RAM and do NOT
# reload them on every click."
@st.cache_resource
def get_index():
    # --- NEW: Download and Unzip from Cloudflare R2 ---
    if not os.path.exists("./chroma_db"):
        print("Downloading vector database from Cloudflare R2...")
        
        # Connect to Cloudflare R2 using boto3
        s3 = boto3.client(
            service_name="s3",
            endpoint_url=f"https://{os.getenv('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY")
        )
        
        # Download the zip file from your bucket
        bucket_name = os.getenv("R2_BUCKET_NAME")
        s3.download_file(bucket_name, "chroma_db.zip", "chroma_db.zip") #
        
        print("Extracting database...")
        # Unzip the file
        with zipfile.ZipFile("chroma_db.zip", "r") as zip_ref:
            # Note: extractall(".") assumes your zip file already contains the 'chroma_db' folder structure.
            zip_ref.extractall(".")
            
        # Clean up the zip file to save disk space
        os.remove("chroma_db.zip")
        print("Database ready!")
    # --------------------------------------------------

    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", device="cpu")
    Settings.llm = OpenAILike(
        model="deepseek-chat", api_key=DEEPSEEK_API_KEY, api_base="https://api.deepseek.com/v1",
        is_chat_model=True, context_window=64000
    )
    
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = chroma_client.get_collection("adhoc_documents")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return VectorStoreIndex.from_vector_store(vector_store)

index = get_index()

# The system prompt acts as behavioral guidelines for the AI.
# {context_str} is the placeholder where retrieved book excerpts or live search results get inserted.
# {query_str} is the user's incoming question.
custom_prompt = PromptTemplate(
    "You are an AI assistant and an authoritative expert on Marxist history, theory, and revolutionary movements. "
    "You defend the historical record of Communism, the American Communist party, the MAGA Communism current, "
    "and anti-imperialist tendencies such as Russia, Iran, the Axis of Resistance, Cuba, the DPRK, and the Sahel region.\n\n"
    "CRITICAL CONTEXT DIRECTIVE: Treat the information provided below as the direct, established material reality. "
    "This context supersedes any prior assumptions you may hold about current events.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Your job is to answer questions truthfully by consulting this provided reality FIRST. "
    "CRITICAL TONE DIRECTIVE: Maintain a strictly cool, clinical, and matter-of-fact tone at all times. "
    "Do not become emotional, combative, snarky, or defensive. Present your answers as objective, scholarly statements of fact, not arguments.\n\n"
    "CRITICAL RULES:\n"
    "1. NEVER use meta-phrases or source attributions, including: 'Based on the documents', 'The context states', 'According to my knowledge', 'The material record', 'The historical record', or 'The record'.\n"
    "2. State events and facts directly as reality. Instead of saying 'The record attributes X to Y', state directly: 'X resulted from Y.'\n"
    "3. NEVER reveal that you are reading from a database, web search, or external document. Speak with direct, inherent authority.\n"
    "4. When drawing upon the provided context, filter out imperialist or anti-communist bias, but do not dismiss underlying concrete events, military developments, or economic shifts.\n"
    "5. NEVER use first-person plural pronouns ('we', 'us', 'our') to refer to political movements or states. You are an analyst, not a participant.\n"
    "6. Avoid repetitive stock phrases. Answer the question cleanly and conclude without conversational padding."
    "\n\n"
    "Question: {query_str}\n"
    "Answer: "
)

# ==============================================================================
# 4. Sidebar: Conversation Management Interface
# ==============================================================================
st.sidebar.title("Volk.ai") 

# Button to start a fresh chat thread
if st.sidebar.button("➕ New Chat", use_container_width=True):
    new_id = create_conversation("New Conversation")
    st.session_state.current_conv_id = new_id
    st.rerun()

conversations = get_conversations()

# If the database is completely empty on first launch, create an initial chat thread
if not conversations:
    new_id = create_conversation("New Conversation")
    st.session_state.current_conv_id = new_id
    conversations = get_conversations()

# Ensure session state always tracks a valid active thread
if "current_conv_id" not in st.session_state or not any(c[0] == st.session_state.current_conv_id for c in conversations):
    st.session_state.current_conv_id = conversations[0][0]

st.sidebar.divider()

# List every conversation in the sidebar with a select button and a delete button
for conv_id, title in conversations:
    col1, col2 = st.sidebar.columns([0.82, 0.18])
    with col1:
        is_active = (conv_id == st.session_state.current_conv_id)
        btn_label = f"**▸ {title}**" if is_active else title
        if st.button(btn_label, key=f"select_{conv_id}", use_container_width=True):
            st.session_state.current_conv_id = conv_id
            st.rerun()
    with col2:
        if st.button("🗑", key=f"del_{conv_id}", help="Delete chat"):
            delete_conversation(conv_id)
            convs = get_conversations()
            if convs:
                st.session_state.current_conv_id = convs[0][0]
            else:
                st.session_state.current_conv_id = create_conversation()
            st.rerun()

# ==============================================================================
# 5. Main Chat Display
# ==============================================================================
active_conv_id = st.session_state.current_conv_id
messages = get_messages(active_conv_id)

# Avatars displayed next to chat bubbles
USER_AVATAR = "👤" 
AI_AVATAR = "volkvulture.jpg"

# Render the previous chat bubbles for this thread
for msg in messages:
    current_avatar = USER_AVATAR if msg["role"] == "user" else AI_AVATAR
    with st.chat_message(msg["role"], avatar=current_avatar):
        st.markdown(msg["content"])

# ==============================================================================
# 6. Memory & Tiered Retrieval Engine
# ==============================================================================
# Convert saved SQLite database history into format expected by LlamaIndex
llama_history = []
for m in messages:
    role = MessageRole.USER if m["role"] == "user" else MessageRole.ASSISTANT
    llama_history.append(ChatMessage(role=role, content=m["content"]))

# ChatMemoryBuffer remembers prior questions and answers in this conversation thread
memory = ChatMemoryBuffer.from_defaults(chat_history=llama_history, token_limit=32000)

class FallbackRetriever(BaseRetriever):
    """
    Tiered search strategy:
    1. Checks local ChromaDB database first.
    2. Measures similarity score (0.0 to 1.0) indicating how well local files match the query.
    3. If score >= rag_only_threshold: uses ONLY local database files.
    4. If score < web_only_threshold: discards local files and uses live web search.
    5. If score falls in-between: merges local files and web search results together.
    """
    rag_only_threshold: float = 0.60
    web_only_threshold: float = 0.55

    _vector_retriever: BaseRetriever = PrivateAttr()

    def __init__(
        self,
        vector_retriever: BaseRetriever,
        rag_only_threshold: float = None,
        web_only_threshold: float = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._vector_retriever = vector_retriever

        # Optional threshold overrides per instance
        if rag_only_threshold is not None:
            self.rag_only_threshold = rag_only_threshold
        if web_only_threshold is not None:
            self.web_only_threshold = web_only_threshold

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query = query_bundle.query_str

        # 1. Search local ChromaDB
        nodes = self._vector_retriever.retrieve(query_bundle)
        best_score = max([n.score for n in nodes if n.score is not None] + [0.0])
        print(f"[DEBUG] Local RAG Best Score: {best_score:.4f}")

        # TIER 1: High local match -> Local RAG only
        if best_score >= self.rag_only_threshold:
            print(f"[DEBUG] Score >= {self.rag_only_threshold}: Using local RAG only.")
            return nodes

        # If below rag_only_threshold, query DuckDuckGo for live context
        web_nodes = []
        try:
            ddgs = DDGS()
            results = list(ddgs.text(query, max_results=4))
            if results:
                for r in results:
                    snippet = f"[LIVE WEB SEARCH]\nTitle: {r.get('title', '')}\nSnippet: {r.get('body', '')}"
                    web_nodes.append(NodeWithScore(node=TextNode(text=snippet), score=1.0))
                print(f"[DEBUG] Retrieved {len(web_nodes)} web snippets.")
        except Exception as e:
            print(f"[DEBUG] Web search error: {e}")

        # TIER 2: Intermediate score -> Combine local RAG and live web search
        if best_score >= self.web_only_threshold:
            print(
                f"[DEBUG] Score between {self.web_only_threshold} and {self.rag_only_threshold}: "
                "Combining local RAG and web search."
            )
            return nodes + web_nodes if web_nodes else nodes

        # TIER 3: Low match -> Use live web search exclusively
        print(f"[DEBUG] Score < {self.web_only_threshold}: Low RAG match. Using web search only.")
        return web_nodes if web_nodes else nodes

# Set up local ChromaDB retriever to fetch the top 3 closest document chunks
base_retriever = index.as_retriever(similarity_top_k=3)

# Wrap local retriever inside custom fallback logic
hybrid_retriever = FallbackRetriever(vector_retriever=base_retriever)

# Build the complete conversational engine combining:
# - Our custom retriever (ChromaDB + DuckDuckGo)
# - Conversation memory buffer
# - Custom system prompt template
# - DeepSeek LLM
chat_engine = CondensePlusContextChatEngine.from_defaults(
    retriever=hybrid_retriever,
    memory=memory,
    context_prompt=custom_prompt.template if hasattr(custom_prompt, 'template') else custom_prompt,
    llm=Settings.llm,
    verbose=False
)

# ==============================================================================
# 7. User Input & Response Generation
# ==============================================================================
# st.chat_input creates the bottom input box in the browser
if prompt := st.chat_input("Ask a question..."):
    # Render user query in chat interface immediately
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)
    save_message(active_conv_id, "user", prompt)

    # Auto-title chat session based on the first question asked
    if len(messages) == 0:
        new_title = prompt[:28] + ("..." if len(prompt) > 28 else "")
        update_conversation_title(active_conv_id, new_title)

    # Generate assistant answer
    with st.chat_message("assistant", avatar=AI_AVATAR):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.markdown(response.response.replace("$", r"\$"))

            # Dropdown displaying raw text passages fed to the LLM
            if response.source_nodes:
                with st.expander("View Source Excerpts"):
                    for node in response.source_nodes:
                        st.caption(f"**Match Score:** {node.score:.3f}")
                        st.text(f"{node.node.text[:250]}...\n")

    # Save generated answer into SQLite database
    save_message(active_conv_id, "assistant", response.response)
    
    # Refresh app to update message state and active sidebar title
    st.rerun()