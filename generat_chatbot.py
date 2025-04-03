import os
import time
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from bs4 import BeautifulSoup
from langchain.prompts import PromptTemplate
from supabase import create_client, Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import hashlib
from dotenv import load_dotenv
import uuid  
from streamlit_local_storage import LocalStorage
import json
from pdf_reader import *

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
local_storage = LocalStorage()

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}  

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None  


def load_chat_history():
    saved_chats = local_storage.getItem("local")
    if saved_chats:
        st.session_state.chat_sessions = json.loads(saved_chats)
        if st.session_state.chat_sessions:
            st.session_state.current_chat_id = list(st.session_state.chat_sessions.keys())[-1]


def generate_new_chat(Chat_name=None):
    if not Chat_name:
        Chat_name = "New Chat"

    new_chat_id = str(uuid.uuid4())

    st.session_state.chat_sessions[new_chat_id] = {
        "history": [],
        "context_sources": [],
        "context_type": None,
        "Chat Name": Chat_name
    }

    st.session_state.current_chat_id = new_chat_id

    local_storage.setItem("local", json.dumps(st.session_state.chat_sessions), key="setting_sessions")
    
    st.rerun()


def switch_chat(chat_id):
    st.session_state.current_chat_id = chat_id


def store_message(role, content):
    if st.session_state.current_chat_id:
        chat_id = st.session_state.current_chat_id
        st.session_state.chat_sessions[chat_id]["history"].append({"role": role, "content": content})

        # Generate a truly unique key using timestamp and a random component
        unique_key = f"storing_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        local_storage.setItem("local", json.dumps(st.session_state.chat_sessions), key=unique_key)


def display_chat_history():
    chat_id = st.session_state.current_chat_id
    if chat_id and chat_id in st.session_state.chat_sessions:
        for message in st.session_state.chat_sessions[chat_id]["history"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


def vectorsearch(user_query):
    genai.configure(api_key=GEMINI_API_KEY) 

    response = genai.embed_content(
        model="models/embedding-001",
        content=user_query,
        task_type="retrieval_query"
    )
    
    if "embedding" in response:
        query_embedding = response["embedding"]

        response = supabase.rpc(
            "similarity_retrival",
            {
                "query_embedding": query_embedding,
                "match_threshold": 0.5,
                "match_count": 5
            }
        ).execute()

        return response

    return []


def get_llm_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not available in the context, respond with:
    'Answer is not available in the context.'
   
    Context:\n {context}\n
    Question:\n {question}\n
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


def user_prompt(prompt):
    response = vectorsearch(prompt)
    relevant_chunks_with_sources = []
    for row in response.data:
        dict={}
        dict["text"]= row["text"]
        dict["source_name"]= row["source_name"]
        relevant_chunks_with_sources.append(dict)
    
    # Extract just the text for the LLM
    relevant_chunks = [item["text"] for item in relevant_chunks_with_sources]
    
    chain = get_llm_chain()
    
    if not relevant_chunks:
        return "Reply: No context provided in this PDF for that query.", []

    # Pass just the text to the LLM
    response = chain(
        {
            "context": "\n".join(relevant_chunks),
            "question": prompt
        },
        return_only_outputs=True
    )
    
    # Return both the response and the sources
    return response.get("text", "No valid response"), relevant_chunks_with_sources


def general_knowledge_chatbot():
    """Main function for the general knowledge chatbot"""
    st.header("🧠 General Knowledge Chatbot")
    
    # # Clean, modern UI description
    # st.markdown("""
    # <div style="padding: 15px; border-radius: 10px; background-color: #f7f7f7; margin-bottom: 20px;">
    # Ask me anything! I'll search through all stored knowledge to find relevant information.
    # </div>
    # """, unsafe_allow_html=True)
    
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
            search_results = vectorsearch(user_query)
            
            if not search_results:
                response = "I don't have enough information in my knowledge base to answer that question confidently."
            else:
                response,relevent_sources = user_prompt(user_query)
                sources = []
                # Process search results
                for result in relevent_sources:
                    sources.append(result.get("source_name")) 
                # Add source information if available
                if sources:
                    unique_sources = list(set(sources))
                    response += f"\n\n*Information derived from {unique_sources} source(s)*"
            
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
    # st.markdown("""
    # <style>
    # .main {
    #     background-color: #000000;
    # }
    # .stApp {
    #     max-width: 1600px;
    #     margin: 0 auto;
    # }
    # .stButton button {
    #     background-color: #4CAF50;
    #     color: white;
    #     border-radius: 8px;
    #     padding: 0.5rem 1rem;
    #     font-weight: bold;
    # }
    # h1, h2, h3 {
    #     color: #2E4057;
    # }
    # .stSidebar {
    #     background-color: #f8f9fa;
    #     padding-top: 2rem;
    # }
    # .stTextInput input {
    #     border-radius: 8px;
    # }
    # </style>
    # """, unsafe_allow_html=True)
    
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