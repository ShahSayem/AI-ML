import pprint
import google.generativeai as palm

# Configure the API key
from keys1 import googleapi_key
palm.configure(api_key=googleapi_key)

# Generate 
# response = palm.generate_text(prompt="Write a 4 line poem of my love for samosa")
# response = palm.generate_text(prompt="Write an email for asking refund from the university")

# print(response.result)

from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path='codebasics_faqs.csv', source_column='prompt')
data = loader.load()
# print(data)


from langchain_community.embeddings import HuggingFaceInstructEmbeddings

embeddings = HuggingFaceInstructEmbeddings()

e = embeddings.embed_query("What is your refund policy")
print(e)