# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 05:03:56 2024

@author: Samsung
"""

import streamlit as st
from PyPDF2 import PdfReader
#from langchain.text_splitter import RecursiveCharacterSplitter 
from langchain_text_splitters import RecursiveCharacterTextSplitter

#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
#from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain.vectorstores import FAISS
#from langchain.llms import OpenAI 
from langchain_openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
#from langchain.callbacks import get_openai_callback
from langchain_community.callbacks.manager import get_openai_callback
#from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
#from streamlit_extras.add_vertical_space import add_vertical_space
import os
import base64
from PIL import Image
os.environ["OPENAI_API_KEY"] = "sk-cvnp1g7ittF6lUGjvuCJVhnBFJ1hzWT29lFaw0nJGYT3BlbkFJvI54nrhZFFpmb6g1z5N0M-HOTsoz03TqAexbAdXsoA"
st.markdown('''This app is powered by LLM''')
client = OpenAI(api_key="sk-cvnp1g7ittF6lUGjvuCJVhnBFJ1hzWT29lFaw0nJGYT3BlbkFJvI54nrhZFFpmb6g1z5N0M-HOTsoz03TqAexbAdXsoA")

image = Image.open('gba.png')

st.header("Chat with pdf")

st.image(image, caption='Enter any caption here')

main_bg = "llama.png"
main_bg_ext = "png"

side_bg = "llm.jpeg"
side_bg_ext = "jpeg"

form = st.form(key='my-form')
name = form.text_input('Enter your name')
submit = form.form_submit_button('Submit')

st.write('Press submit to have your name printed below')

if submit:
    st.write(f'hello {name}')

st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
   .sidebar .sidebar-content {{
        background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
)

pdf = st.file_uploader("Upload a pdf file" , type='pdf')

with st.sidebar:
    st.title("LLM Chat app")
    st.markdown("This app is powered by LLM for Rag")
    
if pdf is not None: 
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text+= page.extract_text()
        #st.markdown(text)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100,
        length_function = len
    )
    chunks = text_splitter.split_text(text=text)
    file_name = pdf.name[:-4]
    embeddings = OpenAIEmbeddings()
    
    Vector_store = FAISS.from_texts(chunks, embedding =  embeddings)
    #
    #try:
      #  Vector_store = FAISS.from_texts(chunks, embedding =  embeddings)
    #except client.BadRequestError as e:
    #        print(e)
    #except:
    #        print("uncaught")
    
    
   
    Vector_store.save_local("faiss_store_march2")
    query = st.text_input("Ask any question from your uploaded pdf")
    
    if query:
        docs = Vector_store.similarity_search(query=query, k = 3)
        print(docs)
        llm = OpenAI(
            temperature = 0,
            max_tokens = 20
        )
        chain = load_qa_chain(llm = llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents = docs, question=query)
        
        st.write(response)