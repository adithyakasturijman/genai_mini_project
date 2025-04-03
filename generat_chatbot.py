import os
import time
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from supabase import create_client, Client
from dotenv import load_dotenv
import uuid
from streamlit_local_storage import LocalStorage
import json
from main import *

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure API clients
genai.configure(api_key=GEMINI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
local_storage = LocalStorage()

# Initialize session state variables
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None


def load_chat_history():
    """Load saved chat history from local storage"""
    saved_chats = local_storage.getItem("chat_sessions_local")
    if saved_chats:
        st.session_state.chat_sessions = json.loads(saved_chats)
        if st.session_state.chat_sessions:
            st.session_state.current_chat_id = list(st.session_state.chat_sessions.keys())[-1]


def generate_new_chat(chat_name=None):
    """Create a new chat session"""
    if not chat_name:
        chat_name = "General Knowledge Chat"

    new_chat_id = str(uuid.uuid4())

    st.session_state.chat_sessions[new_chat_id] = {
        "history": [],
        "context_type": "general",
        "Chat Name": chat_name
    }

    st.session_state.current_chat_id = new_chat_id

    local_storage.setItem("chat_sessions_local", json.dumps(st.session_state.chat_sessions), key="setting_sessions")
    st.rerun()


def switch_chat(chat_id):
    """Switch to an existing chat session"""
    st.session_state.current_chat_id = chat_id


def store_message(role, content):
    """Store a message in the current chat session"""
    if st.session_state.current_chat_id:
        chat_id = st.session_state.current_chat_id
        st.session_state.chat_sessions[chat_id]["history"].append({"role": role, "content": content})

        # Generate a unique key for local storage
        unique_key = f"storing_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        local_storage.setItem("chat_sessions_local", json.dumps(st.session_state.chat_sessions), key=unique_key)


def display_chat_history():
    """Display the current chat history"""
    chat_id = st.session_state.current_chat_id
    if chat_id and chat_id in st.session_state.chat_sessions:
        for message in st.session_state.chat_sessions[chat_id]["history"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


def vectorsearch_all(user_query, limit=5, threshold=0.6):
    """Search all vectors in the database without filtering by hash"""
    genai.configure(api_key=GEMINI_API_KEY) 

    # Get embedding for user query
    response = genai.embed_content(
        model="models/embedding-001",
        content=user_query,
        task_type="retrieval_query"
    )
    
    if "embedding" in response:
        query_embedding = response["embedding"]

        # Search for similar chunks across all documents
        response = supabase.rpc(
            "similarchuncks",
            {
                "query_embedding": query_embedding,
                "match_threshold": threshold,
                "match_count": limit
            }
        ).execute()

        return response.data if response.data else []

    return []


def get_llm_chain():
    """Create a LangChain chain for generating responses"""
    prompt_template = """
    You are a knowledgeable assistant that provides helpful information based on the context provided.
    Answer the user's question as detailed as possible using the provided context.
    If the information in the context is insufficient, acknowledge what you know from the context
    and mention that you have limited information on the topic.
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    model = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-pro-latest",
        google_api_key=GEMINI_API_KEY,
        temperature=0.3
    )

    return LLMChain(llm=model, prompt=prompt)


def general_knowledge_chatbot():
    """Main function for the general knowledge chatbot"""
    st.header("🧠 General Knowledge Chatbot")
    
    # Clean, modern UI description
    st.markdown("""
    <div style="padding: 15px; border-radius: 10px; background-color: #f7f7f7; margin-bottom: 20px;">
    Ask me anything! I'll search through all stored knowledge to find relevant information.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize chat session if needed
    if not st.session_state.current_chat_id:
        generate_new_chat()
    
    # Display existing chat history
    display_chat_history()
    
    # Chat input
    user_query = st.chat_input("Ask me anything...")
    
    if user_query:
        # Store and display user message
        store_message("user", user_query)
        st.chat_message("user").markdown(user_query)
        
        with st.spinner("Searching knowledge base..."):
            # Retrieve relevant chunks from all documents
            search_results = vectorsearch_all(user_query)
            
            if not search_results:
                response = "I don't have enough information in my knowledge base to answer that question confidently."
            else:
                # Extract text from search results
                context_texts = []
                sources = []
                
                # Process search results
                for result in search_results:
                    context_texts.append(result["text"])
                    if "pdf_hash" in result:
                        sources.append(result["pdf_hash"][:8])  # Use truncated hash as source reference
                
                # Get response from LLM
                chain = get_llm_chain()
                llm_response = chain(
                    {
                        "context": "\n\n".join(context_texts),
                        "question": user_query
                    },
                    return_only_outputs=True
                )
                
                response = llm_response.get("text", "I couldn't generate a response based on the available information.")
                
                # Add source information if available
                if sources:
                    unique_sources = list(set(sources))
                    response += f"\n\n*Information derived from {len(unique_sources)} source(s)*"
            
            # Store and display assistant response
            store_message("assistant", response)
            with st.chat_message("assistant"):
                st.markdown(response)


def main():
    st.set_page_config(
        page_title="Smart Knowledge Assistant",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for a clean, modern look
    st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
    }
    .stApp {
        max-width: 1600px;
        margin: 0 auto;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    h1, h2, h3 {
        color: #2E4057;
    }
    .stSidebar {
        background-color: #f8f9fa;
        padding-top: 2rem;
    }
    .stTextInput input {
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Load chat history
    load_chat_history()
    
    # Sidebar for navigation and chat selection
    with st.sidebar:
        st.title("📊 Knowledge Navigator")
        
        # Mode selection
        st.subheader("Select Mode")
        choice = st.radio(
            "Choose a mode",
            ["General Knowledge", "PDF Chat", "Domain Chat"],
            key="mode_selection"
        )
        
        st.divider()
        
        # Chat history selection
        st.subheader("Chat History")
        if st.button("🆕 New Chat", key="new_chat_btn"):
            generate_new_chat()
        
        if st.session_state.chat_sessions:
            st.divider()
            chat_options = {
                chat_id: session["Chat Name"] for chat_id, session in st.session_state.chat_sessions.items()
            }
            
            selected_chat_id = st.radio(
                "Select a conversation",
                list(chat_options.keys()),
                format_func=lambda x: chat_options[x],
                index=0 if st.session_state.current_chat_id is None else list(chat_options.keys()).index(st.session_state.current_chat_id)
            )
            
            if selected_chat_id != st.session_state.current_chat_id:
                switch_chat(selected_chat_id)
    
    if choice == "General Knowledge":
        general_knowledge_chatbot()
    elif choice == "PDF Chat":
        pdf_chatbot()
    elif choice == "Domain Chat":
        domain_chatbot()


if __name__ == "__main__":
    main()