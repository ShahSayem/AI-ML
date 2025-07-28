import streamlit as st
import os 
from langchain_groq import ChatGroq 
from langchain_openai import OpenAIEmbeddings 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain_core.prompts import ChatMessagePromptTemplate 
from langchain.chains import create_retrieval_chain 
from langchain_community.vectorstores import FAISS 
from langchain_community.document_loaders import PyPDFDirectoryLoader 

from  dotenv import load_dotenv 

load_dotenv()

# load the GROQ and OpenAI API key
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
groq_api_key = os.environ('GROQ_API_KEY')

st.title('ChatGroq with Llama-3 Demo')

llm = ChatGroq(groq_api_key = groq_api_key,
               model_name= 'llama3-8b-8192')

prompt = ChatMessagePromptTemplate.from_template(
"""
Answer the questions based on thr provided context only.
Please provide the most accurate response based on the question
<context>
{context}
Questions:{input}

"""
)


def vector_embedding():
    if 'vectors' not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader('./us_census') #Data Ingestion
        st.session_state.docs = st.session_state.loader.load() #Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200) #Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) #Slpitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings) #Vector OpenAI embedding


prompt1 = st.text_input('Enter your question from Documnets')

if st.button('Documents Embedding'):
    vector_embedding()
    st.write('Vector Store DB is ready')


import time 

if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()

    response = retrieval_chain.invoke({'input': prompt1})
    print('Response time: ', time.process_time()-start)
    st.write(response['answer'])

    #with streamlit expander
    with st.expander('Document Similarity Search'):
        #Find the relevent chunk
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("-------------------------")
