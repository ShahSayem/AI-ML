## Have to solve problem

from dotenv import load_dotenv
load_dotenv()

import streamlit as st 
import os
import google.generativeai as genai

# Initialize Google Gemini with API key
genai.configure(api_key=os.getenv('Google_API_KEY'))

# Function to load Gemini Pro model and get response
model = genai.GenerativeModel('gemini-pro')

def get_gemini_response(question, history):
    # Start the chat session with the existing history
    chat = model.start_chat(history=history)
    
    # Send the user's question and get a response
    response = chat.send_message(question)
    
    # Append the new question and response to history
    history.append(("User", question))
    history.append(("Bot", response.text))
    
    return response.text, history

# Initialize Streamlit app
st.header('Gemini LLM Chatbot')

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# User input and submit button
input = st.text_input('Ask here...', key='input')
submit = st.button('Submit')

if submit and input:
    # Get the response and updated history
    response, updated_history = get_gemini_response(input, st.session_state['chat_history'])
    
    # Update session state history
    st.session_state['chat_history'] = updated_history
    
    # Display the response
    st.subheader('The Response is:')
    st.write(response)

# Display chat history
st.subheader('The chat history is:')
for role, text in st.session_state['chat_history']:
    st.write(f'{role}: {text}')
