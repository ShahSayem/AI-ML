# --- Web or app page (Front end)---
import requests
import streamlit as st 

# def get_openai_response(input_text):
#     response = requests.post("http://localhost:8000/poem/invoke",
#     json={'input': {'topic': input_text}})

#     return response.json()['output']

def get_ollama_response(input_text):
    response = requests.post("http://localhost:8000/essay/invoke",
    json={'input': {'topic': input_text}})

    return response.json()['output']


#streamlit framework
st.title("Langchain with LLAma3.1")
# input_text1=st.text_input("Write an poem on...")
input_text2=st.text_input("Write an essay on...")

# if input_text1:
#     st.write(get_openai_response(input_text1))

if input_text2:
    st.write(get_ollama_response(input_text2))