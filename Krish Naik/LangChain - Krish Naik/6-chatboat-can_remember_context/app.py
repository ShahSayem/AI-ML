#Conversational Q&A Chatbot

import streamlit as st 
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()


# Streamlit UI
st.set_page_config(page_title= 'Conversational Q&A Chatbot')
st.header("Hey, Let's Chat")

chat = ChatOpenAI(temperature = 0.5)

if 'flowmessages' not in st.session_state:
    st.session_state['flowmessages'] = [
        SystemMessage(content='You are a comedian AI assistant')
    ]


# Function to load OpenAI model & get response
def get_openai_response(question):
    st.session_state['flowmessages'].append(HumanMessage(content=question))
    answer = chat(st.session_state['flowmessages'])
    st.session_state['flowmessages'].append(AIMessage(content=answer.content))

    return answer.content

input = st.text_input('Input: ', key='input')
response = get_openai_response(input)

submit = st.button('Ask')

# If asked button is clicked
if submit:
    st.subheader('The Response is: ')
    st.write(response)