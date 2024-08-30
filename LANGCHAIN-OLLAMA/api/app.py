from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_community.llms import Ollama
from langserve import add_routes
import uvicorn
import os
from dotenv import load_dotenv
load_dotenv()
# LANGSMITH TRACING
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

app=FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API Server"
)

# add_routes(
#     app,
#     Ollama(model="llava"),
#     path="/openai"
# )

# model = ChatOpenAI()
model = Ollama(model="llava")
##ollama llama2
llm=Ollama(model="llama3.1")

prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
prompt2 = ChatPromptTemplate.from_template("Write me a poem about {topic} with 5 lines for 5 year child")

add_routes(
    app,
    prompt1 | model,
    path="/essay",
)

add_routes(
    app,
    prompt2 | llm,
    path="/poem",
)

add_routes(
    app,
    model,  # This can be used for a general-purpose endpoint
    path="/openai",
)

if __name__=="__main__":
    uvicorn.run(app, host="localhost", port = 8000)