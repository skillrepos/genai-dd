import fitz  # PyMuPDF
from transformers import RagTokenizer, RagTokenForGeneration

def extract_text_from_pdf(pdf_path):
    """Extracts text from a given PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def setup_rag_model():
    """Sets up the RAG tokenizer and model."""
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
    model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq")
    return tokenizer, model

def answer_question(question, context, tokenizer, model):
    """Generates an answer to the question based on the context using RAG."""
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, padding=True)
    with tokenizer.as_target_tokenizer():
        output_ids = model.generate(**inputs)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Example usage
pdf_path = 'sample.pdf'  # Path to your PDF file
context = extract_text_from_pdf(pdf_path)
tokenizer, model = setup_rag_model()
question = "What is the main topic of the document?"
answer = answer_question(question, context, tokenizer, model)
print("Answer:", answer)


### Running the Example
#1. Replace `'sample.pdf'` with the path to your PDF file.
#2. Make sure to have a valid question that relates to the content of the PDF.
#3. Execute the script.

### How It Works
#- **PDF Text Extraction**: The `extract_text_from_pdf` function reads the PDF and extracts all text from it. This text serves as the context for generating answers.
#- **Model Setup**: The `setup_rag_model` function loads the pre-trained RAG tokenizer and model.
#- **Answer Generation**: The `answer_question` function uses the model and tokenizer to generate an answer to the input question based on the extracted PDF text.

