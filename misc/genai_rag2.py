import os
import wget
from langchain.vectorstores import Qdrant
from langchain_community.document_loaders import BSHTMLLoader
from langchain.chains import RetrievalQA
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.llms import Ollama

#download War and Peace by Tolstoy
# wget.download("http://az.lib.ru/t/tolstoj_lew_nikolaewich/text_0073.shtml")
wget.download("https://www.cs.cmu.edu/~rgs/alice-I.html")

#load text from html
loader = BSHTMLLoader("alice-I.html", open_encoding='ISO-8859-1')
war_and_peace = loader.load()

#init Vector DB

embeddings = FastEmbedEmbeddings()  

doc_store = Qdrant.from_documents(
    war_and_peace, 
    embeddings,
    location=":memory:", 
    collection_name="docs",
)

llm = Ollama(model="mistral")
# ask questions

while True:
    question = input('Your question: ')
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=doc_store.as_retriever(),
        return_source_documents=False,
    )

    result = qa(question)
    print(f"Answer: {result}")
