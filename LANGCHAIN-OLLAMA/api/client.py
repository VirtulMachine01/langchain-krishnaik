# This is Webapp or mobile app

import requests
import streamlit as st

def GetLlavaResponse(input_text):
    response = requests.post("http://localhost:8000/essay/invoke",
                             json={'input':{'topic': input_text}})
    return response.json()['output']

def GetLlama3Response(input_text):
    response = requests.post("http://localhost:8000/poem/invoke",
                             json={'input':{'topic': input_text}})
    return response.json()['output']

st.title('Langchain With Llama2 and Llava')
input_text = st.text_input("Write an essay with Llava model")
input_text1 = st.text_input("Write an poem with Llama3.1 model")

# if st.button("Submit"):
#     if input_text:
#         st.write(GetLlavaResponse(input_text))
#     if input_text1:
#         st.write(GetLlama3Response(input_text1))
if input_text:
    st.write(GetLlavaResponse(input_text))
if input_text1:
    st.write(GetLlama3Response(input_text1))