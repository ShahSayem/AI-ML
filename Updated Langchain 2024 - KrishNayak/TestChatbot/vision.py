import os
import streamlit as st
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()  # Take environment variables from .env
genai.configure(api_key=os.getenv('Google_API_KEY'))

# Load the Gemini Vision model
model_image = genai.GenerativeModel('gemini-pro-vision')

# Function to get response from Gemini Vision for image-related queries
def get_gemini_response_image(input_text=None, image=None):
    try:
        # If both input_text and image are provided (if supported by the model)
        if input_text and image:
            response_image = model_image.generate_content(image=image, stream=True)
        # If only image is provided
        elif image:
            response_image = model_image.generate_content(image=image, stream=True)
        else:
            st.error("Please provide at least one input (either text or image).")
            return None
        return response_image
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Streamlit UI configuration
st.set_page_config(page_title="Gemini Vision Image Demo")
st.header("Gemini Vision Application")

# Input text field
input_text = st.text_input("Ask here...", key="input")

# File uploader to accept image uploads (single image for now)
uploaded_file = st.file_uploader("Choose images (jpg, jpeg, png)", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

# Display the uploaded image
image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

# Submit button
submit = st.button("Submit")

# If submit button is clicked
if submit:
    if image is not None:  # Check if an image is uploaded
        # Get Gemini's response for the image
        response = get_gemini_response_image(input_text=input_text, image=image)
        if response:
            st.subheader("The Response is:")
            st.write(response)
    else:
        st.error('Upload at least one picture!', icon="🚨")
