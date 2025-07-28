import streamlit as st 
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Load your Hugging Face API key
api_key = os.getenv("HF_API_KEY")  

# Hugging Face Inference API endpoint
api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-v0.1"

headers = {
    "Authorization": f"Bearer {api_key}"
}

# Streamlit UI
st.set_page_config(page_title='Conversational Q&A Chatbot')
st.header("Hey, Let's Chat")

if 'flowmessages' not in st.session_state:
    st.session_state['flowmessages'] = [
        SystemMessage(content="You are an AI assistant. Respond conversationally and helpfully.")
    ]

# Function to load Mistral model & get response
def get_mistral_response(question):
    st.session_state['flowmessages'].append(HumanMessage(content=question))
    
    # Format the conversation for the model
    conversation = "\n".join([
        f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"Assistant: {msg.content}"
        for msg in st.session_state['flowmessages']
    ])
    
    # Send a request to the Hugging Face API
    response = requests.post(api_url, headers=headers, json={
        "inputs": conversation,
        "parameters": {
            "max_new_tokens": 150,  # Limit the response length
            "temperature": 0.7,      # Increase creativity
            "top_p": 0.9             # Adjust top_p for more diverse responses
        }
    })
    
    if response.status_code == 200:
        result = response.json()
        response_content = result[0]['generated_text'].split("Assistant:")[-1].strip()  # Extract the model's response
        st.session_state['flowmessages'].append(AIMessage(content=response_content))
        return response_content
    else:
        return "Error: Unable to get a response from the model."

input = st.text_input('Input: ', key='input')
submit = st.button('Ask')

# If asked button is clicked
if submit:
    response = get_mistral_response(input)
    st.subheader('The Response is: ')
    st.write(response)
