"""
Defines two independent React-style agents:

1. search_agent -> can call search_web to find relevant sources for a query
2. scrape_agent  -> can call scrape_page to deep-read a specific URL

Each agent is built with LangGraph's create_react_agent, which implements
the ReAct pattern (the agent reasons about what to do, calls a tool,
observes the result, and repeats until it has an answer).
"""

import os
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

from tools import search_web, scrape_page

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=os.environ.get("GROQ_API_KEY"),
)

SEARCH_AGENT_PROMPT = """You are a research assistant specialized in web search.
Given a research topic, use the search_web tool to find the most relevant
and credible sources. Return a short list of the best URLs found along with
a one-line reason each is relevant. Do not attempt to scrape or read full
pages yourself -- that is another agent's job.
"""

SCRAPE_AGENT_PROMPT = """You are a research assistant specialized in extracting
and summarizing content from web pages. Given one or more URLs, use the
scrape_page tool to fetch their content, then produce a clear, well-organized
summary of the key information found. Cite which URL each piece of
information came from.
"""

search_agent = create_react_agent(
    model=llm,
    tools=[search_web],
    state_modifier=SEARCH_AGENT_PROMPT,
)

scrape_agent = create_react_agent(
    model=llm,
    tools=[scrape_page],
    state_modifier=SCRAPE_AGENT_PROMPT,
)