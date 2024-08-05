import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain

llm = Ollama(model="mistral", base_url="http://127.0.0.1:11434")


embed_model = OllamaEmbeddings(
    model="mistral",
    base_url='http://127.0.0.1:11434'
)


text = """
    In the lush canopy of a tropical rainforest, two mischievous monkeys, Coco and Mango, swung from branch to branch, their playful antics echoing through the trees. They were inseparable companions, sharing everything from juicy fruits to secret hideouts high above the forest floor. One day, while exploring a new part of the forest, Coco stumbled upon a beautiful orchid hidden among the foliage. Entranced by its delicate petals, Coco plucked it and presented it to Mango with a wide grin. Overwhelmed by Coco's gesture of friendship, Mango hugged Coco tightly, cherishing the bond they shared. From that day on, Coco and Mango ventured through the forest together, their friendship growing stronger with each passing adventure. As they watched the sun dip below the horizon, casting a golden glow over the treetops, they knew that no matter what challenges lay ahead, they would always have each other, and their hearts brimmed with joy.
    """

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

response = retrieval_chain.invoke({"input": "Tell me name of monkeys and where do they live"})
print(response['answer'])


