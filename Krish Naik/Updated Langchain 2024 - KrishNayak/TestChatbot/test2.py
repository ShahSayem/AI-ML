import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import requests
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()

# Get the Hugging Face API token from environment variables
sec_key = os.getenv('HF_API_KEY')
os.environ['HUGGINGFACEHUB_API_TOKEN'] = sec_key

# Initialize HuggingFaceEndpoint with Microsoft Phi-3 model
repo_id = 'microsoft/Phi-3-mini-4k-instruct'
llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.7, token=sec_key)

# Function to load PDF documents as LangChain Document objects
def load_pdf_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents

# Function to scrape text from a webpage and convert it into a LangChain Document object
def scrape_website_as_document(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text(separator="\n")
    
    # Create a Document object from the scraped content
    webpage_document = Document(page_content=text, metadata={"source": url})
    return webpage_document

# Streamlit UI
st.title("PDF and Web-Based Chatbot (Powered by Microsoft Phi-3 Instruct)")
st.write("Upload a PDF or enter a website URL, and ask questions based on the content!")

# Upload PDF or enter URL
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")
url = st.text_input("Enter a website URL")

# Create text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Prepare documents for embedding
documents = []
if pdf_file:
    with open(f"./{pdf_file.name}", "wb") as f:
        f.write(pdf_file.read())
    pdf_docs = load_pdf_documents(f"./{pdf_file.name}")
    documents.extend(pdf_docs)

if url:
    webpage_document = scrape_website_as_document(url)
    documents.append(webpage_document)

# Split documents into chunks
if documents:
    final_documents = text_splitter.split_documents(documents)

    # Define embeddings using Hugging Face
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS vector store from documents
    db = FAISS.from_documents(final_documents, embedding_function)

    # Define the prompt template for answering questions based on the context
    prompt_template = """
    Use the following context to answer the question.

    Context: {context}
    Question: {question}

    Answer:
    """

    # Input box for asking questions
    user_query = st.text_input("Ask a question based on the document or website:")

    # When user submits a query
    if user_query:
        with st.spinner('Generating response...'):
            # Retrieve relevant context from the vector store
            retriever = db.as_retriever()
            context_documents = retriever.get_relevant_documents(user_query)
            context = "\n".join([doc.page_content for doc in context_documents])
            
            # Construct the prompt
            full_prompt = prompt_template.format(context=context, question=user_query)
            
            # Query the Hugging Face Phi-3 Instruct model
            response = llm(full_prompt)

            # Display the answer and source documents
            st.write("**Answer:**", response)
            st.write("**Source Documents:**")
            for doc in context_documents:
                st.write(f"- {doc.metadata['source']}")
else:
    st.write("Please upload a PDF or enter a website URL to start.")
