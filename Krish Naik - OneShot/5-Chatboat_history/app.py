#Conversational Q&A Chatbot with Gemini

from dotenv import load_dotenv
load_dotenv()

import streamlit as st 
import os
import google.generativeai as genai

genai.configure(api_key=os.getenv('Google_API_KEY'))


# function to load Gamini Pro model and get response
model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])


def get_gemini_response(question):
    response = chat.send_message(question, stream=True)

    return response



# initialize our streamlit app
st.header('Gemini LLM Chatbot')

# initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []


input = st.text_input('Ask here...', key= 'input')
submit = st.button('Submit')

if submit and input:
    response = get_gemini_response(input)

    #Add user querry and response to session chat history
    st.session_state['chat_history'].append(('User', input))
    st.subheader('The Response is: ')

    for chunk in response:
        st.write(chunk.text)
        st.session_state['chat_history'].append(('Bot', chunk.text))


st.subheader('The chat history is')

for role, text in st.session_state['chat_history']:
    st.write(f'{role}:{text}')