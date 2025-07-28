import os
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from langchain_huggingface import HuggingFaceEndpoint
import base64

# Load environment variables
load_dotenv()

# Get the Hugging Face API token from environment variables
sec_key = os.getenv('HF_API_KEY')
os.environ['HUGGINGFACEHUB_API_TOKEN'] = sec_key

# Initialize HuggingFaceEndpoint with Microsoft Phi-3.5 Vision Instruct model
repo_id = 'microsoft/Phi-3.5-vision-instruct'
llm = HuggingFaceEndpoint(repo_id=repo_id, token=sec_key)

# Streamlit UI
st.title("Image-Based Chatbot (Powered by Microsoft Phi-3.5 Vision Instruct)")
st.write("Upload an image and ask questions based on the content of the image!")

# Image uploader
uploaded_file = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert the image to base64 format to embed in the prompt
    image_bytes = uploaded_file.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    # Input box for asking questions about the image
    user_query = st.text_input("Ask a question about the image:")

    # When the user submits a query
    if user_query:
        with st.spinner('Generating response...'):
            # Construct the full prompt with base64 image data and the question
            full_prompt = f"Here is an image in base64 format: {image_base64}\nQuestion: {user_query}\nAnswer the question based on the image."

            # Query the Hugging Face Phi-3.5 Vision model
            response = llm(full_prompt)

            # Display the answer
            st.write("**Answer:**", response)
else:
    st.write("Please upload an image to start.")
