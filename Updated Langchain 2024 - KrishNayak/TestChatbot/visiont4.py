import os
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import requests
import base64

# Load environment variables
load_dotenv()

# Get the Hugging Face API token from environment variables
HF_API_KEY = os.getenv('HF_API_KEY')

# Hugging Face Inference API URL for the model
API_URL = "https://api-inference.huggingface.co/models/microsoft/Phi-3.5-vision-instruct"

# Set up headers with the API key
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# Function to query Hugging Face Inference API
def query_huggingface_api(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

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
            # Create payload with the base64 image and the query
            payload = {
                "inputs": {
                    "image": image_base64,
                    "text": user_query
                }
            }

            # Query the Hugging Face API
            response = query_huggingface_api(payload)

            # Display the answer
            if 'error' in response:
                st.write("**Error:**", response['error'])
            else:
                st.write("**Answer:**", response.get("generated_text", "No response found"))
else:
    st.write("Please upload an image to start.")
