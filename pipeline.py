"""
Connects search_agent and scrape_agent into a single pipeline using LCEL
(LangChain Expression Language) / Runnables -- NOT the legacy Chain classes.

Flow:
  user query
      -> search_agent (finds relevant URLs)
      -> extract URLs from search_agent's answer
      -> scrape_agent (reads those URLs and summarizes)
      -> final combined output

RunnableLambda wraps plain Python functions so they can be composed with
the `|` operator, just like any other LCEL Runnable.
"""

import re
from langchain_core.runnables import RunnableLambda

from agents import search_agent, scrape_agent


def run_search_agent(query: str) -> dict:
    """Invoke the search agent and return its raw text output + the original query."""
    result = search_agent.invoke({"messages": [{"role": "user", "content": query}]})
    final_message = result["messages"][-1].content
    return {"query": query, "search_output": final_message}


def extract_urls(state: dict) -> dict:
    """Pull URLs out of the search agent's response so the scrape agent has something to fetch."""
    urls = re.findall(r"https?://[^\s\)\]]+", state["search_output"])
    # Keep it small: scraping too many pages burns tokens and time for a demo project
    state["urls"] = urls[:3]
    return state


def run_scrape_agent(state: dict) -> dict:
    """Invoke the scrape agent with the discovered URLs."""
    if not state["urls"]:
        state["final_summary"] = (
            "No URLs were found by the search agent, so no pages could be scraped.\n\n"
            f"Search agent output:\n{state['search_output']}"
        )
        return state

    scrape_prompt = (
        f"Here are the URLs to read and summarize:\n" + "\n".join(state["urls"])
    )
    result = scrape_agent.invoke({"messages": [{"role": "user", "content": scrape_prompt}]})
    state["final_summary"] = result["messages"][-1].content
    return state


# The actual LCEL pipeline: each step is a Runnable, composed with `|`
research_pipeline = (
    RunnableLambda(run_search_agent)
    | RunnableLambda(extract_urls)
    | RunnableLambda(run_scrape_agent)
)


def run_research(query: str) -> str:
    """Convenience wrapper: run the full pipeline and return just the final summary."""
    result = research_pipeline.invoke(query)
    return result["final_summary"]