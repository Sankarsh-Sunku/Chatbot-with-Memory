import streamlit as st
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from huggingface_hub import login
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class ChatBotManager:
    def __init__(self, model_name):
        self.prompt_template = """Use the following pieces of information to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        context: {context}\n
        chathistory : {chathistory}\n\n
        question: {question}\n

        Only return the helpful answer. Answer must be detailed and well explained.
        Helpful answer:
        """
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)
        self.prompt = ChatPromptTemplate.from_template(self.prompt_template)

    def format_docs(self,docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def get_response(self, query, chatHistory, retriever):
        chain = (
            self.prompt |
            self.llm |
            StrOutputParser()
        )
        retrival_chain = retriever |  self.format_docs
        docs = retrival_chain.invoke(query)
        response = chain.invoke({"context" : docs, "chathistory" : chatHistory, "question" : query})
        print(response)
        return response
    

