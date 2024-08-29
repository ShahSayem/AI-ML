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


# Function to load OpenAI model & get response
def get_openai_response(question):
    llm = OpenAI(model_name = 'text-davinci-003', temperature = 0.5)
    response = llm(question)

    return response