from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

import streamlit as st
import os

os.environ["OPEN_API_KEY"]=os.getenv("OPEN_API_KEY")

#Langmith tracking
os.environ["LANGCHAIN-TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

#Prompt Tamplate
prompt=ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please response to the user queries"),
        ("user", "Question:{question}")
    ]
)

#streamlit framework
st.title("Chatbot app with OPEN API")
input_text=st.text_input("Ask here...")

#OpenAI LLM
llm=ChatOpenAI(model="gpt-3.5-turbo")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question': input_text}))