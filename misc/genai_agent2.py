from crewai import Crew, Agent, Task 
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings  # Use the wrapper
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from crewai_tools import BaseTool
from crewai_tools import tool
from crewai_tools import WebsiteSearchTool
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

searchtool = WebsiteSearchTool(
    website="https://www.almanac.com/thanksgiving-day", 
    config=dict(
        llm=dict(
            provider="ollama", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="phi3",
            ),
        ),
        embedder=dict(
            provider="ollama",
                config=dict(
                    model="mxbai-embed-large:latest",
                    
            ),
        ),
    )
)

  
   
llm = Ollama(model="phi3")

agent = Agent(
    role="Calendar",
    goal="What day of the month is Thanksgiving on in 2024?",
    backstory="You are a calendar assistant. You provide information about dates. ",
    tools=[searchtool],
    llm=llm,
    allow_delegation=False, verbose=True)

task = Task(description="What day of the month is Thanksgiving on in 2024?",
                   agent=agent,
                   expected_output="Date of Thanksgiving in 2024")

crew = Crew(agents=[agent], tasks=[task], verbose=True)
res = crew.kickoff()
print(res)
