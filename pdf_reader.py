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


def setup_selenium():
    options = Options()
    options.headless = False
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver


def fetch_content(driver, url):
    driver.get(url)
    time.sleep(3)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    return soup


def get_pdf_text(pdf_docs):
    all_texts = []
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
        all_texts.append({"text": text, "source_name": pdf.name})
    return all_texts


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)


def generate_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def store_vectors_in_db(text_chunks, source_name):
    genai.configure(api_key=GEMINI_API_KEY)

    full_text = "\n".join(text_chunks)
    pdf_hash = generate_hash(full_text)

    existing_entry = supabase.table("chunks_embedding").select("source_name").eq("source_name", source_name).execute()

    if existing_entry.data:
        print("Duplicate PDF detected. Skipping storage.")
        return

    for chunk in text_chunks:
        response = genai.embed_content(
            model="models/embedding-001",
            content=chunk,
            task_type="retrieval_document"
        )
        vector = response["embedding"]  

        supabase.table("chunks_embedding").insert({
            "text": chunk, 
            "embedding": vector,
            "hash": pdf_hash,
            "source_name": source_name
        }).execute()

    print("New PDF stored successfully!")


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


def user_prompt(prompt, pdf_names):
    response = vectorsearch(prompt)
    # Get relevant chunks FROM the uploaded PDFs along with their source information
    relevant_chunks_with_sources = [
        {"text": row["text"], "source": row["source_name"]} 
        for row in response.data if row.get("source_name") in pdf_names
    ]
    
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


# Modify the PDF chatbot function to display sources
def pdf_chatbot():
    st.header("📄 PDF Chatbot")

    pdf_docs = st.file_uploader("Upload your PDF Files", type=["pdf"], accept_multiple_files=True, key="file_uploaded")

    if pdf_docs:
        st.write("Uploaded PDF files:")
        for pdf in pdf_docs:
            st.write(pdf.name)  # Print the name of each uploaded PDF

        if not st.session_state.current_chat_id:
            generate_new_chat(pdf_docs[0].name)

        chat_id = st.session_state.current_chat_id

        if not st.session_state.chat_sessions[chat_id]["Chat Name"] or st.session_state.chat_sessions[chat_id]["Chat Name"] != pdf_docs[0].name:
            st.session_state.chat_sessions[chat_id]["Chat Name"] = pdf_docs[0].name

        with st.spinner("Processing..."):
            raw_texts = get_pdf_text(pdf_docs)
            if not raw_texts:
                st.error("No text extracted from the PDFs.")
                return

            pdf_names = []
            for i in raw_texts:
                pdf_names.append(i["source_name"])
                text_chunks = get_text_chunks(i["text"])
                store_vectors_in_db(text_chunks, i["source_name"])

            st.success("PDF processed successfully! You can now chat with it.")
            st.session_state.chat_sessions[chat_id]["context_sources"] = pdf_names
            st.session_state.chat_sessions[chat_id]["context_type"] = "pdf"

    display_chat_history()

    user_query = st.chat_input("Ask something about the PDF...")

    if user_query:
        store_message("user", user_query)
        st.chat_message("user").markdown(user_query)
        chat_id = st.session_state.current_chat_id

        if not st.session_state.chat_sessions[chat_id].get("context_sources"):
            with st.chat_message("assistant"):
                st.markdown("Please upload and process a PDF first.")
            store_message("assistant", "Please upload and process a PDF first.")
        else:
            pdf_names = st.session_state.chat_sessions[chat_id]["context_sources"]
            bot_response, sources = user_prompt(user_query, pdf_names)

            if bot_response:
                # Create a formatted response with sources
                formatted_response = f"{bot_response}\n\n---\n\n**Sources:**\n"
                
                # Add unique sources to avoid repetition
                unique_sources = set()
                for source in sources:
                    unique_sources.add(source["source"])
                
                # Add each unique source to the response
                for source in unique_sources:
                    formatted_response += f"- {source}\n"
                
                store_message("assistant", formatted_response)
                with st.chat_message("assistant"):
                    st.markdown(formatted_response)


# Similarly modify the domain chatbot function
def domain_chatbot():
    st.header("🌍 Domain Chatbot")

    domain_link = st.text_input("🔗 Enter Domain Link")
    if domain_link:
        if not st.session_state.current_chat_id:
            generate_new_chat(domain_link)

        chat_id = st.session_state.current_chat_id

        if not st.session_state.chat_sessions[chat_id]["Chat Name"] or st.session_state.chat_sessions[chat_id]["Chat Name"] != domain_link:
            st.session_state.chat_sessions[chat_id]["Chat Name"] = domain_link

    if st.button("Store Domain in Database"):
        with st.spinner("Fetching content..."):
            driver = setup_selenium()
            resp = fetch_content(driver, domain_link)
            raw_text = " ".join(resp.stripped_strings)
            text_chunks = get_text_chunks(raw_text)

            # Use the domain as the source name
            source_name = domain_link
            
            # Get current chat ID
            chat_id = st.session_state.current_chat_id
            
            # Set the source and type
            st.session_state.chat_sessions[chat_id]["context_sources"] = [source_name]
            st.session_state.chat_sessions[chat_id]["context_type"] = "domain"
            
            # Immediately save to local storage to ensure persistence
            unique_key = f"domain_store_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            local_storage.setItem("local", json.dumps(st.session_state.chat_sessions), key=unique_key)
            
            # Store vectors in database with source name
            store_vectors_in_db(text_chunks, source_name)
            st.success("Domain data stored successfully! You can now chat with it.")
            driver.quit()  # Close the browser

    display_chat_history()

    user_query = st.chat_input("Ask something about the domain...")

    if user_query:
        store_message("user", user_query)
        st.chat_message("user").markdown(user_query)
        
        chat_id = st.session_state.current_chat_id
        
        context_sources = st.session_state.chat_sessions[chat_id].get("context_sources")
        
        if not context_sources:
            with st.chat_message("assistant"):
                st.markdown("Please click 'Store Domain in Database' button first to process the website content.")
            store_message("assistant", "Please click 'Store Domain in Database' button first to process the website content.")
        else:
            bot_response, sources = user_prompt(user_query, context_sources)
            
            if bot_response:
                # Create a formatted response with sources
                formatted_response = f"{bot_response}\n\n---\n\n**Sources:**\n"
                
                # Add unique sources to avoid repetition
                unique_sources = set()
                for source in sources:
                    unique_sources.add(source["source"])
                
                # Add each unique source to the response
                for source in unique_sources:
                    formatted_response += f"- {source}\n"
                
                store_message("assistant", formatted_response)
                with st.chat_message("assistant"):
                    st.markdown(formatted_response)


def main():
    st.set_page_config(page_title="AI Assistant", layout="wide")
    load_chat_history()

    st.sidebar.title("🔍 Previous Chats")
    if st.session_state.chat_sessions:
        chat_options = {
            chat_id: session["Chat Name"] for chat_id, session in st.session_state.chat_sessions.items()
        }
    
        selected_chat_id = st.sidebar.radio(
            "Select a chat",
            list(chat_options.keys()),
            format_func=lambda x: chat_options[x],  # Display chat names instead of IDs
            index=0 if st.session_state.current_chat_id is None else list(chat_options.keys()).index(st.session_state.current_chat_id)
        )

        if selected_chat_id != st.session_state.current_chat_id:
            switch_chat(selected_chat_id)

    if st.sidebar.button("🆕 New Chat"):
        generate_new_chat()

    st.sidebar.title("📌 Select Mode")
    choice = st.sidebar.radio("Choose a mode", ["PDF Chat", "Domain Chat"])

    if choice == "PDF Chat":
        pdf_chatbot()
    elif choice == "Domain Chat":
        domain_chatbot()


if __name__ == "__main__":
    main()