import streamlit as st
import os
import sqlite3
import uuid
from datetime import datetime
from dotenv import load_dotenv

import chromadb
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike

# 1. Environment & API Key
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    st.error("DEEPSEEK_API_KEY not found in .env file.")
    st.stop()

st.set_page_config(page_title="Volk.ai", page_icon="☭", layout="wide")

# 2. Local Database for Conversation History (SQLite)
DB_FILE = "conversations.db"

def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT,
                created_at TIMESTAMP
            )
        """)
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
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, title FROM conversations ORDER BY created_at DESC")
        return cursor.fetchall()

def create_conversation(title="New Conversation"):
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
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conv_id))
        conn.commit()

def delete_conversation(conv_id):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
        cursor.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        conn.commit()

def get_messages(conv_id):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (conv_id,)
        )
        return [{"role": r, "content": c} for r, c in cursor.fetchall()]

def save_message(conv_id, role, content):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (conv_id, role, content, datetime.now())
        )
        conn.commit()

# 3. Cache Index & Embedding Models Only (Stateless)
@st.cache_resource
def get_index():
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        device="cuda"  # Switch to "cpu" if running strictly without GPU
    )

    Settings.llm = OpenAILike(
        model="deepseek-chat",
        api_key=DEEPSEEK_API_KEY,
        api_base="https://api.deepseek.com/v1",
        is_chat_model=True,
        context_window=64000
    )

    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = chroma_client.get_collection("adhoc_documents")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return VectorStoreIndex.from_vector_store(vector_store)

index = get_index()

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
    "6. NEVER attack the user or use rhetorical insults. Simply state what is materially correct and end the response.\n"
    "\n"
    "Question: {query_str}\n"
    "Answer: "
)

# 4. Sidebar: Conversation Management
st.sidebar.title("Volk.ai") 

if st.sidebar.button("➕ New Chat", use_container_width=True):
    new_id = create_conversation("New Conversation")
    st.session_state.current_conv_id = new_id
    st.rerun()

conversations = get_conversations()

# If no conversation exists at all, make the first one
if not conversations:
    new_id = create_conversation("New Conversation")
    st.session_state.current_conv_id = new_id
    conversations = get_conversations()

if "current_conv_id" not in st.session_state or not any(c[0] == st.session_state.current_conv_id for c in conversations):
    st.session_state.current_conv_id = conversations[0][0]

st.sidebar.divider()

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

# 5. Main Chat Interface
active_conv_id = st.session_state.current_conv_id
messages = get_messages(active_conv_id)

# DEFINE YOUR AVATARS HERE (Can be emojis or local file paths like "icon.png")
USER_AVATAR = "👤" 
AI_AVATAR = "volkvulture.jpg"

# Display conversation messages
for msg in messages:
    current_avatar = USER_AVATAR if msg["role"] == "user" else AI_AVATAR
    with st.chat_message(msg["role"], avatar=current_avatar):
        st.markdown(msg["content"])

# 6. Build the Engine with Hydrated Memory for Active Thread
llama_history = []
for m in messages:
    role = MessageRole.USER if m["role"] == "user" else MessageRole.ASSISTANT
    llama_history.append(ChatMessage(role=role, content=m["content"]))

memory = ChatMemoryBuffer.from_defaults(chat_history=llama_history, token_limit=32000)

chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=memory,
    context_prompt=custom_prompt,
    similarity_top_k=3,
    verbose=False
)

# 7. User Input Handling
if prompt := st.chat_input("Ask a question..."):
    # Render user prompt immediately
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)
    save_message(active_conv_id, "user", prompt)

    # Automatically rename the chat thread if it's the first question
    if len(messages) == 0:
        new_title = prompt[:28] + ("..." if len(prompt) > 28 else "")
        update_conversation_title(active_conv_id, new_title)

    # Generate assistant answer
    with st.chat_message("assistant", avatar=AI_AVATAR):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.markdown(response.response)

            if response.source_nodes:
                with st.expander("View Source Excerpts"):
                    for node in response.source_nodes:
                        st.caption(f"**Match Score:** {node.score:.3f}")
                        st.text(f"{node.node.text[:250]}...\n")

    save_message(active_conv_id, "assistant", response.response)
    st.rerun()