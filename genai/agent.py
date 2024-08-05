import crewai
from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun
from crewai_tools import tool

llm = Ollama(model="mistral")

def callback_function(output):
    print(f"Task completed: {output.raw_output}")

@tool("DuckDuckGoSearch")
def search(search_query: str) -> str:
    """Search the web for information on a given topic"""
    return DuckDuckGoSearchRun().run(search_query)

agent = crewai.Agent(
    role="Calendar",
    goal="What day of the month is Thanksgiving on in 2024?",
    backstory="You are a calendar assistant. You provide information about dates. ",
    tools=[search],
    llm=llm,
    allow_delegation=False, verbose=True)

task = crewai.Task(description="What day of the month is Thanksgiving on in 2024?",
                   agent=agent,
                   expected_output="Date of Thanksgiving in the current year")

crew = crewai.Crew(agents=[agent], tasks=[task], verbose=True)
res = crew.kickoff()
print(res)
