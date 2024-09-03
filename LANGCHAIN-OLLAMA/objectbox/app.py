from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_groq.chat_models import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_objectbox.vectorstores import ObjectBox
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
import streamlit as st
import os
import time
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')

llm=ChatGroq(groq_api_key=groq_api_key, model_name = "llama3-8b-8192")

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

embeddings = HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/all-MiniLM-l6-v2",
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True}
    )

## Vector Embedding and Object Vectorstore db
# It working good with FAISS and object box is not giving the any similarity search

def vector_embedding():
    if "vectorstore" not in st.session_state:
        st.session_state.embeddings = embeddings
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")
        st.session_state.documents = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.documents)
        st.session_state.vectorstore = ObjectBox.from_documents(st.session_state.final_documents, st.session_state.embeddings, embedding_dimensions = 384)
        # st.session_state.vectorstore = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

ip_prompt = st.text_input("Enter Your Question From Documents")

if st.button("Document Embedding"):
    vector_embedding()
    st.write("Object box DB is Ready")


if "vectorstore" in st.session_state:
    prompt = st.session_state.prompt if 'prompt' in st.session_state else prompt
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectorstore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({'input':ip_prompt})
    print("Response Time : ", time.process_time())
    st.write(response['answer'])

    # With a streamlit Expander
    # with st.expander("Document similarity Search"):
    #     # Find the Relevant Chunks
    #     for i, doc in enumerate(response["context"]):
    #         st.write(doc.page_content)
    #         st.write("-----------------------------------")
else:
    st.write("Please initialize the vectorstore first by clicking the 'Document Embedding' button.")