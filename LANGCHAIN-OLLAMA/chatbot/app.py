from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate # required for chat bot
from langchain_core.output_parsers import StrOutputParser # default output parser but we can create custom output parser
from langchain_community.llms import Ollama
# from ollama import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# LANGSMITH TRACING
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

## Prompt Template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please response to the queries"),
        ("user", "Question:{question}")
    ]
)

## Streamlit Framework

st.title("LANGCHAIN DEMO WITH LLAMA3")
input_text = st.text_input("Search the topic you want")

## OLLAMA LLM

llm = Ollama(model="llama3.1")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if st.button("Submit"):
    response = chain.invoke({'question': input_text})
    st.write(response)
# if input_text:
#     st.write(chain.invoke({'question': input_text}))