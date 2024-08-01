import chromadb
from chromadb.utils import embedding_functions

# set some config variables for ChromaDB
CHROMA_DATA_PATH = "vdb_data/"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "demo_docs"

# create a new ChromaDB client
vdb_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

# specify the embedding functions we'll use
vdb_embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
     model_name=EMBED_MODEL
)

# create a new collection or reuse existing one
vdb_collection = vdb_client.get_or_create_collection(
     name=COLLECTION_NAME,
     embedding_function=vdb_embedding_func,
     metadata={"hnsw:space": "cosine"},
)

# our virtual "documents" to draw on for context
datadocs = [

     "Honey never spoils. Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible.",
     "The shortest war in history was between Britain and Zanzibar on August 27, 1896. Zanzibar surrendered after 38 minutes.",
     "Canada has the longest coastline of any country in the world, stretching over 202,080 kilometers (125,567 miles).",
     "The longest novel ever written is 'In Search of Lost Time' by Marcel Proust. It has approximately 1.2 million words.",
     "The first 1GB hard drive, released in 1980 by IBM, weighed over 500 pounds and cost $40,000.",
     "The world’s largest grand piano was built by a 15-year-old in New Zealand. It is 5.7 meters long and took four years to build.",
     "Leonardo da Vinci's 'Mona Lisa' has no eyebrows because it was the fashion in Renaissance Florence to shave them off.",
     "A single strand of spider silk is five times stronger than a strand of steel of the same thickness.",
     "The longest tennis match in history took place at Wimbledon in 2010 between John Isner and Nicolas Mahut, lasting 11 hours and 5 minutes over three days.",
     "There is a giant cloud of alcohol in Sagittarius B, a gas cloud in the Milky Way, containing enough ethyl alcohol to make 400 trillion trillion pints of beer.",
]

# add some metadata about the categories of the documents
# provides additional info about the document
# can also query on this
categories = [
     "science",
     "history",
     "geography",
     "literature",
     "technology",
     "music",
     "art",
     "nature",
     "sports",
     "space",
]

vdb_collection.add(
    documents=datadocs,
    ids=[f"id{i}" for i in range(len(datadocs))],
    metadatas=[{"category": c} for c in categories]
)


while True:
    query = input("\nQuery: ")
    if query == "exit":
        break
    if query.strip() == "":
        continue
    query_results = vdb_collection.query(
        query_texts=[query],
        n_results=1,
    )
    
    print(query_results["documents"])
    print(query_results["distances"])
    print(query_results["metadatas"])
