"""
Tools used by our agents:
1. search_web  -> uses Tavily to search the internet
2. scrape_page -> uses BeautifulSoup to fetch and clean text from a URL

Both are defined using the @tool decorator, which turns a plain Python
function into something a LangChain/LangGraph agent can call.
"""

import os
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))


@tool
def search_web(query: str) -> str:
    """
    Search the web for up-to-date information on a given query.
    Use this when you need current facts, news, or general information
    that isn't already known. Returns a summary of the top results.
    """
    response = tavily_client.search(query=query, max_results=5)

    results = response.get("results", [])
    if not results:
        return "No search results found."

    formatted = []
    for r in results:
        formatted.append(f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content']}\n")

    return "\n---\n".join(formatted)


@tool
def scrape_page(url: str) -> str:
    """
    Fetch a webpage and extract its main readable text content.
    Use this when you have a specific URL (e.g. from search_web results)
    and need the full article/page content rather than just a snippet.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        return f"Failed to fetch URL: {e}"

    soup = BeautifulSoup(response.content, "html.parser")

    # Remove elements that aren't useful readable content
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned_text = "\n".join(lines)

    # Cap length so we don't blow up the LLM context window
    max_chars = 6000
    if len(cleaned_text) > max_chars:
        cleaned_text = cleaned_text[:max_chars] + "\n...[truncated]"

    return cleaned_text