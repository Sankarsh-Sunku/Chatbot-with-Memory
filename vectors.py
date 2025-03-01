import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document  # Import Document
from langchain_community.vectorstores import FAISS

class VectorManager:
    def __init__(self, model_name,):
        self.model_name = model_name
        self.embeddings = GoogleGenerativeAIEmbeddings(model = model_name)
    
    def create_embeddings(self,pdf_docs):
        text = ""
        for pdf_doc in pdf_docs:
            pdf_reader = PdfReader(pdf_doc)
            for page in pdf_reader.pages:
                text +=page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=250
        )
        chunks = text_splitter.split_text(text)
        print(len(chunks))
        documents = [Document(page_content=chunk) for chunk in chunks]
        # Embed
        print("Print DOC ------------",documents)
        print("----------------------------------------------")

        vectorstore = Chroma.from_documents(documents=documents, 
                                    embedding=self.embeddings)
        
        return vectorstore.as_retriever(search_kwargs={"k": 2})


