import sys
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from tempfile import NamedTemporaryFile

llm = Ollama(model="mistral", base_url="http://127.0.0.1:11434")


pdf_path = sys.argv[1]

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



embed_model = OllamaEmbeddings(
    model="mistral",
    base_url='http://127.0.0.1:11434'
)


text = extract_text_from_pdf(pdf_path)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
chunks = text_splitter.split_text(text)

vector_store = Chroma.from_texts(chunks, embed_model)


retriever = vector_store.as_retriever()

chain = create_retrieval_chain(combine_docs_chain=llm,retriever=retriever)

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

combine_docs_chain = create_stuff_documents_chain(
    llm, retrieval_qa_chat_prompt
)

retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)    

prompt = ChatPromptTemplate.from_messages(
    [("system","Tell me name of monkeys and where do they live?\n\n{context}")]
)

response = retrieval_chain.invoke({"context": retrieval_chain})
print(response['answer'])
