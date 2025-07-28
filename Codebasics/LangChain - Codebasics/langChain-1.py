# From Codebasics: LangChain Crash Course For Beginners
# https://www.youtube.com/watch?v=nAmC7SoVLd8&t=506s

import os
from keys1 import openapi_key

#os.environ["OPEN_API_KEY"] = openapi_key

from langchain_community.llms import OpenAI

llm = OpenAI(temperature=0.6, openai_api_key=openapi_key)
# name = llm("I want to open resturant for Indian food. Suggest a fency name for this.")
# print(name)

from langchain.prompts import PromptTemplate

PromptTemplate(
    input_variables= ['cuisine'],
    templete = "I want to open a returant for {cuisine} food. Suggest a fency name for this"
)

prompt_template_name.format(cuisine="Italian")

from langchain.chains import LLMChain

chain = LLMChain(llm=llm, prompt= prompt_template_name)
chain.run("American")



prompt_template_name = PromptTemplate(
    input_variables=['cuisine'],
    template= "I want to open a returant for {cuisine} food. Suggest a fency name for this"
)
name_chain = LLMChain(llm=llm, prompt= prompt_template_name)

prompt_template_items = PromptTemplate(
    input_variables=['resturant_name'],
    template= "Suggest some menu items for {resturant_name} food. Return it as a comma separated list"
)
food_items_chain = LLMChain(llm=llm, prompt= prompt_template_name)


from langchain.chains import SimpleSequentialChain

chain = SimpleSequentialChain(chains = [name_chain, food_items_chain])
response = chain.run("Indian")
print(response)


prompt_template_name = PromptTemplate(
    input_variables=['cuisine'],
    template= "I want to open a returant for {cuisine} food. Suggest a fency name for this"
)
name_chain = LLMChain(llm=llm, prompt= prompt_template_name, output_keys= "resturant_name")

prompt_template_items = PromptTemplate(
    input_variables=['resturant_name'],
    template= "Suggest some menu items for {resturant_name} food. Return it as a comma separated list"
)
food_items_chain = LLMChain(llm=llm, prompt= prompt_template_name, output_keys= "menu_items")

from langchain.chains import sequentialChanin

sequentialChanin(
    chains = [name_chain, food_items_chain],
    input_variables = ['cuisine'],
    output_variables = ['resturant_name', 'menu_items']
)

chain({'cuisine': 'Arabic'})

