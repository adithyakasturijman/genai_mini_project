from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as ggi


load_dotenv(".env")

fetcheed_api_key = os.getenv("GEMINI_API_KEY")

ggi.configure(api_key = fetcheed_api_key)

model = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-pro-latest",
        google_api_key=GEMINI_API_KEY,
        temperature=0.3
    )
chat = model.start_chat()

def LLM_Response(question):
    response = chat.send_message(question,stream=True)
    return response

st.title("Chat Application using Gemini Pro")

user_quest = st.text_input("Ask a question:")
btn = st.button("Ask")

if btn and user_quest:
    result = LLM_Response(user_quest)
    st.subheader("Response : ")
    for word in result:
        st.text(word.text)