import os
import time
import json
import csv
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from supabase import create_client, Client
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import hashlib
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def generate_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def store_vectors_in_db(text_chunks):
    genai.configure(api_key=GEMINI_API_KEY)

    full_text = "\n".join(text_chunks)
    pdf_hash = generate_hash(full_text)

    existing_entry = supabase.table("text_embeddings").select("pdf_hash").eq("pdf_hash", pdf_hash).execute()

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

        supabase.table("text_embeddings").insert({
            "text": chunk, 
            "embedding": vector,
            "pdf_hash": pdf_hash 
        }).execute()

    print("New PDF stored successfully!")


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)


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


def main():
    driver = setup_selenium()
    url = "https://www.apple.com"
    resp = fetch_content(driver,url)
    raw_text = " ".join(resp.stripped_strings)
    text_chunks = get_text_chunks(raw_text)
    store_vectors_in_db(text_chunks)
    print(text_chunks)


if __name__ == "__main__":
    main()
