import streamlit as st
from app import PhiChatbot

# Streamlit UI
st.set_page_config(
    page_title="LLM Chatbot",
    initial_sidebar_state="collapsed"
)

st.header('Microsoft Phi-3-mini LLM Chatbot with Multiple Files & URLs by Shah Sayem')

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Initialize PhiChatbot instance
chatbot = PhiChatbot()

def load_files():
    for uploaded_file in uploaded_files:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        with open(f"./{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.read())
        
        # Load different file types based on extension
        if file_extension == "pdf":
            chatbot.documents.extend(chatbot.load_pdf_documents(f"./{uploaded_file.name}"))
        elif file_extension == "xlsx":
            chatbot.documents.extend(chatbot.load_excel_documents(f"./{uploaded_file.name}"))
        elif file_extension == "txt":
            chatbot.documents.extend(chatbot.load_txt_documents(f"./{uploaded_file.name}"))


def load_urls():
    for url in urls:
        if url.strip(): # Check if the URL is not empty
            webpage_document = chatbot.scrape_website_as_document(url.strip())

            if webpage_document:
                chatbot.documents.append(webpage_document)


def get_combined_prompt():
    context = ""
    if chatbot.documents:
        retriever = db.as_retriever()
        context_documents = retriever.get_relevant_documents(input_text)
        context = "\n".join([doc.page_content for doc in context_documents])

    # Combine chat history and retrieved context for prompt
    combined_prompt = (
        "Previous conversation: " + str(st.session_state['chat_history']) +
        "\nContext: " + context +
        "<|system|> You are a helpful assistant. Please respond to the prompt based on the previous conversation and context. If you do not find previous conversation or context then make response by yourself."
        "<|user|>" + input_text + "<|end|> <|assistant|>"
    )

    return combined_prompt


def process_display_chat_response_history():
    try:
        st.subheader('The Response is: ')
        st.write(response)
    except Exception as e:
        st.error(f"Broken response: {e}")

    # Display sources
    sources = 'Chatbot Generated'
    if chatbot.documents:
        sources = 'Uploaded Files & URLs'  

    st.write(f'**Sources**: {sources}')

    st.session_state['chat_history'].append(('**User**', input_text))
    st.session_state['chat_history'].append(('**Bot**', response))
    st.session_state['chat_history'].append(('Sources', sources))

    st.subheader('The chat history is: ')
    for role, text in st.session_state['chat_history']:
        st.write(f'{role}: {text}')


# Upload multiple files and URLs
with st.sidebar:
    uploaded_files = st.file_uploader("Upload files (PDF, Excel, TXT)", type=["pdf", "xlsx", "txt"], accept_multiple_files=True)
    urls = st.text_area("Enter website URLs (separate by commas)").split(",")

    # Handle file uploads
    if uploaded_files:
        load_files()
    
    # Handle URLs
    if urls:
        load_urls()

    process = st.button("Process files & urls")


if chatbot.documents:
    db = chatbot.process_documents()

    if process:
        with st.spinner("Processing..."):
            db = chatbot.process_documents()
            st.success("Processed files & urls")

# Input box for asking questions
input_text = st.text_input('Ask here...')

if input_text:
    # Display loading spinner
    with st.spinner('Generating response...'):
        
        combined_prompt = get_combined_prompt()

        # generating response
        try:
            response = chatbot.llm(combined_prompt)
        except Exception as e:
            response = e
            st.error(f"Broken response from Microsoft Phi-3 API: {e}")

        # Process and display the response and chat history
        process_display_chat_response_history()
else:
    st.write("You can upload files(PDF, Excel, TXT), enter website URLs or just start your queries.")

