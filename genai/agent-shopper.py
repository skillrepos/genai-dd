import crewai
from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun
from crewai_tools import tool
from crewai_tools import WebsiteSearchTool

llm = Ollama(model="mannix/phi3-mini-4k")

@tool("Search Amazon")
def search_amazon(q: str) -> str:
    """Search Amazon"""
    return DuckDuckGoSearchRun().run(f"site:https://amazon.com {q}")

@tool("DuckDuckGoSearch")
def search(search_query: str):
    """Search the web for information on a given topic"""
    return DuckDuckGoSearchRun().run(search_query)

def callback_function(output):
    print(f"Task completed: {output.raw_output}")

prompt = '''Find info on a good laptop under $1000'''

agent_amazon_shopper = crewai.Agent(
    role="Amazon Shopper",
    goal="Find info about a good laptop under $1000",
    backstory="As a savvy shopper, you need to find info on a good laptop under $1000",
    tools=[search],
    llm=llm,
    allow_delegation=False, verbose=True)

task_shop = crewai.Task(description=prompt,
                   agent=agent_amazon_shopper,
                   expected_output='''info on laptop''')

crew = crewai.Crew(agents=[agent_amazon_shopper], tasks=[task_shop], verbose=True)
res = crew.kickoff()
print(res)
