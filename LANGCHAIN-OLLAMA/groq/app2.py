from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
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

##load the GROQ API KEY
groq_api_key = os.getenv('GROQ_API_KEY')

st.title("CHATGROQ with Llama3")

llm = ChatGroq(groq_api_key=groq_api_key,
               model_name="llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
""""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
</context>
Questions:{input}
"""
)

def VectorEmbeddings():
    if "vectorstore" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")
        st.session_state.documents = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.documents)
        st.session_state.vectorstore = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

prompt1 = st.text_input("Enter your Question From Domuments")


if st.button("Document Embedding"):
    VectorEmbeddings()
    st.write("Vector Store DB is Ready.")




if prompt1:
    stTime = time.process_time()
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectorstore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({'input':prompt1})
    print("Response Time : ", time.process_time()-stTime)
    st.write(response['answer'])

    #With a streamlit Exapander
    with st.expander("Document similarity Search"):
        #Find the Relevant Chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("-----------------------------------")