import crewai
from langchain_community.tools import DuckDuckGoSearchRun
from crewai_tools import tool
from crewai_tools import WebsiteSearchTool

# Import the LLM class
from crewai import Agent, LLM

    
llm=LLM(
    model="ollama/mannix/phi3-mini-4k"
)

# Define custom search tool
#@tool('DuckDuckGoSearch')
@tool('Search')
def search(search_query: str):
    """Search the web for information on a given topic"""
    return DuckDuckGoSearchRun().run(search_query)
   # return "The Olympic Games, like almost all Greek games, were an intrinsic part of a religious festival. They were held in honour of Zeus at Olympia by the city-state of Elis in the northwestern Peloponnese. The first Olympic champion listed in the records was Coroebus of Elis, a cook, who won the sprint race in 776 bce. The modern Olympic Games (OG; or Olympics; French: Jeux olympiques, JO; Greek: Ολυμπιακοί Αγώνες) are the leading international sporting events featuring summer and winter sports competitions in which thousands of athletes from around the world participate in a variety of competitions.The Olympic Games are considered the world's foremost sports competition with more than 200 ... The history of the Games is incredibly rich and spans millennia. The first written evidence of the official Games dates from 776 BC, when the Greeks began measuring time in Olympiads, or the duration between each edition of the Olympic Games. The first Olympic Games were held every four years in honour of the god Zeus. The Summer Olympic Games, also known as the Games of the Olympiad, is a major international multi-sport event normally held once every four years. The inaugural Games took place in 1896 in Athens, Greece, and the most recent Games are being held in 2024 in Paris, France.This was the first international multi-sport event of its kind, organized by the International Olympic Committee (IOC ... July 10, 2024 5:23 PM EDT. V iewers around the world will come together to watch athletes compete in the Olympics in Paris this summer, four years since they last saw gymnasts, swimmers, and ..."

# Define agents
researcher = Agent(
    name="Researcher",
    role="Senior Researcher",
    goal="Find the latest topic about olympics on the internet",
    backstory="""you are an expert in researching the latest news and information about various topics""",
    verbose=True,
    allow_delegation=False,
    tools=[search],
    llm=llm,
     )

writer = Agent(
     name="Writer",
     role="Technical Writer",
     goal="Write a brief but engaging report about the info provided using simple English",
     backstory="""You are expert in writing various domain articles. Your articles are engaging and interesting.""",
     verbose=True,
     allow_delegation=True,
     llm=llm,
     )

# Define task
def summarize_info(topic):
  # Researcher gathers information
  info = researcher.run_tool("searchtool", query=topic)
  # Writer summarizes the information
  summary = writer.ask(f"Summarize this information: {info}")
  return summary

# Create and run the Crew
task = Task(description="Summarize a topic", function=summarize_info, args=["olympics"],expected_output="summary of olympics info", agent=writer)
crew = Crew(agents=[researcher, writer], tasks=[task])
result = crew.kickoff()

print(f"Summary: {result}")
