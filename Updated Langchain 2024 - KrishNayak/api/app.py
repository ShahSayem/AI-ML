### -- API --

from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_community.llms import Ollama
from langserve import add_routes
from dotenv import load_dotenv

import uvicorn
import os

load_dotenv()

# os.environ["OPEN_API_KEY"]=os.getenv("OPEN_API_KEY")

app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API Server"
)

# add_routes(
#     app,
#     ChatOpenAI(),
#     path="/openai"
# )

# model=ChatOpenAI()


#Ollama LLAma3.1 LLM (8B - 4.7GB)
llm=Ollama(model="llama3.1")

# prompt1=ChatPromptTemplate.from_template("Write me an poem about {topic} with 100 words")
prompt2=ChatPromptTemplate.from_template("Write me an essay about {topic}")

#OpenAI
# add_routes(
#     app,
#     prompt1 | model,
#     path="/poem"
# )

#LLAma3.1
add_routes(
    app,
    prompt2 | llm,
    path="/essay"
)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
