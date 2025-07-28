import os
import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (PyPDFLoader, TextLoader,
                                        UnstructuredExcelLoader)



# Load environment variables
load_dotenv()

# Get the Hugging Face API token from environment variables
sec_key = os.getenv('HF_API_KEY')
os.environ['HUGGINGFACEHUB_API_TOKEN'] = sec_key

# Initialize HuggingFaceEndpoint with Microsoft Phi-3 model
repo_id = 'microsoft/Phi-3-mini-4k-instruct'


class PhiChatbot:
    def __init__(self):
        self.documents = []
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
        self.llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.7, token=sec_key)
        
    # Scrape a webpage and convert it into a LangChain Document object
    def scrape_website_as_document(self, url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text(separator="\n")
            webpage_document = Document(page_content=text, metadata={"source": url})

            return webpage_document
        
        except Exception as e:
            st.error(f"Error fetching or parsing the website: {e}")
            return None

    # Load PDF files as LangChain Document objects
    def load_pdf_documents(self, pdf_path):
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        return documents

    # Load Excel files as LangChain Document objects
    def load_excel_documents(self, excel_path):
        loader = UnstructuredExcelLoader(excel_path)
        documents = loader.load()
        
        return documents

    # Load TXT files as LangChain Document objects
    def load_txt_documents(self, txt_path):
        loader = TextLoader(txt_path)
        documents = loader.load()

        return documents

    # Split documents into chunks and embed them using HuggingFace
    def process_documents(self):
        final_documents = self.text_splitter.split_documents(self.documents)
        db = FAISS.from_documents(final_documents, self.embedding_function)

        return db



