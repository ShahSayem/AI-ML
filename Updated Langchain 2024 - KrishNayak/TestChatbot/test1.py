import os
from dotenv import load_dotenv
import google.generativeai as genai
import streamlit as st
from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredExcelLoader, UnstructuredPowerPointLoader, TextLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
#from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import docx
import nltk
nltk.download('punkt')
import requests
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv('Google_API_KEY'))

# Load Gemini Pro model
model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

# Function to get response from Gemini
def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return response


# Function to scrape text from a webpage and convert it into a LangChain Document object
def scrape_website_as_document(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator="\n")
    
        # Create a Document object from the scraped content
        webpage_document = Document(page_content=text, metadata={"source": url})
        return webpage_document
    except Exception as e:
        st.error(f"Error fetching or parsing the website: {e}")
        return None
    
# Function to load PDF documents as LangChain Document objects
def load_pdf_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents

# Functions to load other file types
def load_word_documents(word_path):
    loader = UnstructuredWordDocumentLoader(word_path)
    documents = loader.load()
    return documents

def load_excel_documents(excel_path):
    loader = UnstructuredExcelLoader(excel_path)
    documents = loader.load()
    return documents

def load_powerpoint_documents(ppt_path):
    loader = UnstructuredPowerPointLoader(ppt_path)
    documents = loader.load()
    return documents

def load_txt_documents(txt_path):
    loader = TextLoader(txt_path)
    documents = loader.load()
    return documents


# Streamlit UI
st.header('Gemini LLM Chatbot with Multiple Files & URLs by Shah Sayem')

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []


# Upload multiple PDFs and other files
uploaded_files = st.file_uploader("Upload files (PDF, Word, Excel, PPT, TXT)", type=["pdf", "docx", "xlsx", "pptx", "txt"], accept_multiple_files=True)

# Input multiple URLs
urls = st.text_area("Enter website URLs (separate by commas)").split(",")


# Create text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Prepare documents for embedding
documents = []


# Handle file uploads
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        with open(f"./{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.read())
        
        # Load different file types based on extension
        if file_extension == "pdf":
            documents.extend(load_pdf_documents(f"./{uploaded_file.name}"))
        elif file_extension == "docx":
            documents.extend(load_word_documents(f"./{uploaded_file.name}"))
        elif file_extension == "xlsx":
            documents.extend(load_excel_documents(f"./{uploaded_file.name}"))
        elif file_extension == "pptx":
            documents.extend(load_powerpoint_documents(f"./{uploaded_file.name}"))
        elif file_extension == "txt":
            documents.extend(load_txt_documents(f"./{uploaded_file.name}"))

# Handle URLs
if urls:
    for url in urls:
        if url.strip():  # Check if the URL is not empty
            webpage_document = scrape_website_as_document(url.strip())

            if webpage_document:
                documents.append(webpage_document)

# Split documents into chunks
if documents:
    final_documents = text_splitter.split_documents(documents)

    # Define embeddings using HuggingFace
    #embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") #2X Slower than 'paraphrase-MiniLM-L3-v2' but more accurate 
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2") #2X Faster than 'all-MiniLM-L6-v2' but less accurate 
    # embedding_function = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

    # Create FAISS vector store from documents
    db = FAISS.from_documents(final_documents, embedding_function)

# Input box for asking questions
input_text = st.text_input('Ask here...')

if input_text:
    # Display loading spinner
    with st.spinner('Generating response...'):
        # Retrieve relevant context from documents
        context = ""
        if documents:
            retriever = db.as_retriever()
            context_documents = retriever.get_relevant_documents(input_text)
            context = "\n".join([doc.page_content for doc in context_documents])

        # Combine chat history and retrieved context for prompt
        combined_prompt = (
            "Previous conversation: " + str(st.session_state['chat_history']) +
            "\nContext: " + context +
            "<|system|> You are a helpful assistant. Please response to the prompt based on the previous conversation and context. If you do not find previous conversation or context then make response by yourself."
            "<|user|>" + input_text + "<|end|> <|assistant|>"
        )

        # generating response
        response = get_gemini_response(combined_prompt)


    # Process and display the response
    res_str = ""
    for chunk in response:
        res_str += chunk.text

    st.subheader('The Response is: ')
    st.write(res_str)

    # Display source documents
    sources = set()
    store_sources = 'Chatbot Generated'

    curr = ''
    if documents:
        for doc in context_documents:
            curr = doc.metadata['source']
            if curr.startswith("./"):
                curr = curr[2:]

            sources.add(f"- {curr}")


        store_sources = ''
        for source in sources:    
            store_sources += ('\n' + source)

    st.write(f'**Sources**: {store_sources}')

    # Update chat history
    st.session_state['chat_history'].append(('**User**', input_text))
    st.session_state['chat_history'].append(('**Bot**', res_str))
    st.session_state['chat_history'].append(('Sources', store_sources))

    # Display the chat history
    st.subheader('The chat history is: ')
    for role, text in st.session_state['chat_history']:
        st.write(f'{role}: {text}')
else:
    st.write("You can upload files(PDF, Word, Excel, PPT, TXT), enter website URLs or just start your queries.")
