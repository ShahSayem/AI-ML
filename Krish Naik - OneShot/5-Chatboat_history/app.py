#Conversational Q&A Chatbot with Gemini

from dotenv import load_dotenv

load_dotenv()

import os

import google.generativeai as genai
import streamlit as st

genai.configure(api_key=os.getenv('Google_API_KEY'))


# function to load Gamini Pro model and get response
model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])


def get_gemini_response(question):
    response = chat.send_message(question, stream=True)

    return response



# initialize our streamlit app
st.header('Gemini LLM Chatbot by Shah Sayem')

# initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []


input_text = st.text_input('Ask here...')

if input_text:
    combined_prompt = "Previous conversation:" + str(st.session_state['chat_history']) + "<|system|> You are a helpful assistant. Plaease answer the prompt based on Previous conversation <|end|> <|user|>" + input_text + "<|end|> <|assistant|>"

    with st.spinner('Generating response...'):
        response = get_gemini_response(combined_prompt)

    #Add user querry and response to session chat history
    st.session_state['chat_history'].append(('User', input_text))
    st.subheader('The Response is: ')

    res_str = ""
    for chunk in response:
        res_str += chunk.text

    st.write(res_str)
    st.session_state['chat_history'].append(('Bot', res_str))


st.subheader('The chat history is: ')

for role, text in st.session_state['chat_history']:
    st.write(f'{role}:{text}')