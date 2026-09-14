"""
Simple Streamlit web interface for the multi-agent research system.

Run with: streamlit run app.py
"""

import streamlit as st
from pipeline import run_research

st.set_page_config(page_title="Multi-Agent Research System", page_icon="🔎")

st.title("🔎 Multi-Agent Research System")
st.caption("Search agent finds sources → Scrape agent reads and summarizes them")

query = st.text_input("Enter your research question:", placeholder="e.g. What is LangChain?")

if st.button("Research", type="primary") and query:
    with st.spinner("Running search agent → scrape agent pipeline..."):
        try:
            summary = run_research(query)
            st.markdown("### Research Summary")
            st.markdown(summary)
        except Exception as e:
            st.error(f"Something went wrong: {e}")

st.divider()
st.caption(
    "Built with LangGraph (ReAct agents), LangChain LCEL/Runnables, "
    "Tavily search, and BeautifulSoup scraping."
)