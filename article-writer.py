import os 
from dotenv import load_dotenv
import json 
from pprint import pprint
from ddgs import DDGS
import trafilatura 
from IPython.display import Markdown,display
from agents import Agent, Runner, function_tool

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None: 
    raise Exception("APEN AI API key is missing")

# Step 1:  Define tools

# Tool 1: 
@function_tool
def search_web(query: str):
    """search the web using DuckDuckGo browser and return 3 results"""
    ddgs = DDGS()
    results= ddgs.text(query, max_results = 3)
    print(f"  \u2705 search_web:Got results for {query}")
    return json.dumps(results, indent =2) #Way to convert  dictionary into text

# Tool 2: 
#URL scraping
@function_tool
def fetch_url(url: str): 
    """Fetch the content of URL using trafilatura"""
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        text = trafilatura.extract(downloaded)
        if text:
            print(f" \u2705 fetch_url: Got {len(text)} chars from url :{url[:60]}")
            return text
    print(f" \u274c Failed to fetch or extract text . Try from a different source")
    return f"Could not extract text from {url}. Try a different source"

# Step 2:  Agents 
# This tells the LLM **Who it is** and **how to behave** . The key things: 
# - What its job is 
# - What tools it has 
# - What process to follow 
# - What output format to produce 

#  2.a: Research Agent 

RESEARCH_AGENT_PROMPT = """You are a research specialist. Your job is to research a given topic
and produce a comprehensive research brief.
You have access to two tools:
- search_web: Search the web for information
- fetch_url: Fetch and read the full content of a web page

Your typical process:
1. Search for the topic to find relevant sources
2. Reflect on the search results — which sources look most relevant and why?
3. Fetch the full content of the 2-3 best URLs
4. Reflect on what you have gathered. Do you have enough? Are there gaps?
5. If there are gaps, search again with a different query
6. When you have enough information from at least 6 different sources, synthesize into a research brief
You MUST gather information from at least 6 distinct sources before delivering your brief. 
If you have fewer than 6 sources, keep searching.

Your research brief MUST include:
- Key facts and statistics
- Main themes and arguments from the sources
- Notable data points
- Source URLs for attribution

Until you are ready, just keep working — search, fetch, think, reflect.
Do not rush. Take time to reflect between tool calls before deciding your next step.
ALWAYS cite the source URLS used.
Not every response needs a tool call — sometimes just thinking through what you have is the right move."""

MODEL = "gpt-4.1-nano"
research_agent = Agent(
    name="Research agent",
    instructions = RESEARCH_AGENT_PROMPT,
    model = MODEL,
    tools = [search_web, fetch_url]
)


#  2.a: Orchestrator Agent 

ORCHESTRATOR_AGENT_PROMPT = """You are the orchestrator of a multi-agent article writing system.
Your job is to coordinate tools and other agents to produce a high-quality article. 
Use the tools available to you and/or delegate tasks to the appropriate agents.
Never do the work yourself. Always use tools or agents. 
Your tools and agents are specialists and should be doing the work, you are the manager.

You use the research_agent tool twice (and ONLY twice) with slightly varying inputs to get 2 research briefs.
You pick the best research brief out of the two and deliver it as output. 
Do not combine the two briefs, just pick the best one.
Do not do the research yourself or add anything, you MUST use the research_agent tool to get the briefs.
"""

orchestrator_agent = Agent(
    name= "Orchestrator Agent",
    instructions = ORCHESTRATOR_AGENT_PROMPT,
    model = MODEL,
    tools = [research_agent.as_tool(tool_name = "research agent", tool_description = "Research the topic and return a brief with key facts, statistics, themes andsource URL's. Passthe topic as input")] # Calling another agent as tool
)

#  Step 3: Run the agent 
result = await Runner.run(orchestrator_agent, input = "Research the following topic and produce a comprehensive research brief : I want to learn RAG. Help me understand RAG", max_turns = 10)

