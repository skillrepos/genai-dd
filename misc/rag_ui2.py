import streamlit as st
import PyPDF2
from transformers import pipeline
from chromadb import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from tempfile import NamedTemporaryFile
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_community.llms import Ollama


# Initialize the local language model (LLM) for text generation

llm = Ollama(model="mistral", base_url="http://127.0.0.1:11434")

# Initialize ChromaDB for chunking and embedding
chromadb = chromadb.Client()

# Function to perform Retrieval-Augmented Generation (RAG) with PDFs
def rag_with_pdf(prompt, pdf_path, llm, chromadb, top_k=5):
    loader = PyPDFLoader(pdf _path)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunked_documents = text_splitter.split_documents(document)

    Chroma.from_documents(
        documents=chunked_documents,
        embedding=embedding_function,
        collection_name=os.getenv("CHROMA_COLLECTION_NAME"),
        client=chroma_client,
    )
    print(f"Added {len(chunked_documents)} chunks to chroma db")

    chroma_client = chromadb.HttpClient(host=os.getenv("CHROMA_HOST"), port=int(os.getenv("CHROMA_PORT")), settings=Settings())
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

#    embedding_function = SentenceTransformerEmbeddingFunction()
#    print(embedding_function([token_split_texts[10]]))

    chroma_collection = chroma_client.get_or_create_collection("CHROMA_COLLECTION_NAME", embedding_function=embedding_function)

    results = collection.query(
       query_texts=["What are the forecasts for 2024?"],
       n_results=2 
    )

    print(results)

   
if __name__ == "__main__":
    main()
