import streamlit as st
import PyPDF2
from transformers import pipeline
from chromadb import chromadb
from tempfile import NamedTemporaryFile
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
    chunks = chromadb.chunk(text_from_pdf)
    embeddings = chromadb.embed(chunks)
    similarity_scores = []
    for chunk_embed in embeddings:
        similarity_scores.append(chromadb.similarity(prompt, chunk_embed))
    best_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)[:top_k]
    retrieved_chunks = [chunks[i] for i in best_indices]
    retrieved_text = " ".join(retrieved_chunks)
    generated_text = llm(prompt, context=retrieved_text, max_length=50, num_return_sequences=1)
    return generated_text[0]['generated_text']

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
