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

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    text = ""
    with NamedTemporaryFile(dir='.', suffix='.pdf') as f:
        f.write(pdf_path.getbuffer())
        with open(f.name, "rb") as file:
            reader = PyPDF2.PdfReader(f.name)
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()      
            f.close()      
    return text

# Function to perform Retrieval-Augmented Generation (RAG) with PDFs
def rag_with_pdf(prompt, pdf_path, llm, chromadb, top_k=5):
    text_from_pdf = extract_text_from_pdf(pdf_path)
    # print(word_wrap(text_from_pdf))
    print(text_from_pdf)
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=0
    )
    character_split_texts = character_splitter.split_text('\n\n'.join(text_from_pdf ))

    # print(word_wrap(character_split_texts[10]))
    print(character_split_texts)
    print(f"\nTotal chunks: {len(character_split_texts)}")
    token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)

    token_split_texts = []
    for text in character_split_texts:
        token_split_texts += token_splitter.split_text(text)

    # print(word_wrap(token_split_texts[10]))
    print(token_split_texts[10])
    print(f"\nTotal chunks: {len(token_split_texts)}")

    embedding_function = SentenceTransformerEmbeddingFunction()
    print(embedding_function([token_split_texts[10]]))

    chroma_collection = chromadb.get_or_create_collection("example", embedding_function=embedding_function)

    ids = [str(i) for i in range(len(token_split_texts))]

    chroma_collection.add(ids=ids, documents=token_split_texts)
    chroma_collection.count()
    results = chroma_collection.query(query_texts=[prompt], n_results=5)
    retrieved_documents = results['documents'][0]
 
    return results['documents'][0]

# Streamlit UI
def main():
    st.title("RAG with Local PDFs")

    # Prompt input
    prompt = st.text_input("Enter Prompt", "")

    # PDF file upload
    pdf_file = st.file_uploader("Upload PDF File", type=["pdf"])

    if st.button("Generate Text"):
        if prompt == "":
            st.warning("Please enter a prompt.")
        elif pdf_file is None:
            st.warning("Please upload a PDF file.")
        else:
            # Perform RAG with the provided prompt and PDF file
            generated_text = rag_with_pdf(prompt, pdf_file, llm, chromadb)
            st.subheader("Generated Text")
            st.write(generated_text)

if __name__ == "__main__":
    main()
