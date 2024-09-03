from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq.chat_models import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import streamlit as st
import os
import time
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

st.title("Gemma Model Chatbot")

llm=ChatGroq(groq_api_key=groq_api_key, model_name = "gemma2-9b-it")

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
</context>
Questions:{input}
"""
)


def vector_embedding():
    if "vectorstore" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")
        st.session_state.documents = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.documents)
        st.session_state.vectorstore = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

ip_prompt = st.text_input("Enter Your Question From Documents")

if st.button("Document Embedding"):
    vector_embedding()
    st.write("VectorDB is Ready")


if "vectorstore" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectorstore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({'input':ip_prompt})
    st.write(response['answer'])
else:
    st.write("Please initialize the vectorstore first by clicking the 'Document Embedding' button.")