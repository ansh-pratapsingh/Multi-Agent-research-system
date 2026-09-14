"""
Entry point for the multi-agent research system.

Run with: python main.py "your research question here"
"""

import sys
from pipeline import run_research


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter your research question: ")

    print(f"\n🔎 Researching: {query}\n")
    print("Running search agent -> scrape agent pipeline...\n")

    summary = run_research(query)

    print("=" * 60)
    print("FINAL RESEARCH SUMMARY")
    print("=" * 60)
    print(summary)


if __name__ == "__main__":
    main()