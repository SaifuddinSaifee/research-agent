import os
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
from langchain.schema import SystemMessage
from fastapi import FastAPI

load_dotenv()
brwoserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")

# 1. Tool for search


def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query
    })

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)

    return response.text

# search("What is meta's tread product?") # test

# 2. Tool for scraping
def scrape_website(objective: str, url: str):
    # scrape website, and also will summarize the content based on objective if the content is too large
    # objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

    print("Scraping website...")
    # Define the headers for the request
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Define the data to be sent in the request
    data = {
        "url": url
    }

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request
    post_url = f"https://chrome.browserless.io/content?token={brwoserless_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)

    # Check the response status code
    if response.status_code == 200: # Successfully extreacted content
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("CONTENT:", text)

        # if len(text) > 10000:
        #     output = summary(objective, text)
        #     return output
        # else:
        #     return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")



# def summary(objective, content):
#     llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

#     text_splitter = RecursiveCharacterTextSplitter(
#         separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
#     docs = text_splitter.create_documents([content])
#     map_prompt = """
#     Write a summary of the following text for {objective}:
#     "{text}"
#     SUMMARY:
#     """
#     map_prompt_template = PromptTemplate(
#         template=map_prompt, input_variables=["text", "objective"])

#     summary_chain = load_summarize_chain(
#         llm=llm,
#         chain_type='map_reduce',
#         map_prompt=map_prompt_template,
#         combine_prompt=map_prompt_template,
#         verbose=True
#     )

#     output = summary_chain.run(input_documents=docs, objective=objective)

#     return output
