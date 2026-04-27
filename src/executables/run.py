#!/usr/bin/env python3

import sys
from agent_building.agent import build_agent

def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py \"<your question about machine learning.>\"")
        return
    query = sys.argv[1]
    agent = build_agent(verbose=2)
    print("Query:", query)
    result = agent.run(query)
    print("\n=== Final Answer ===\n", result)



def run_agent(email, query, topic,mode,include_history="yes"):
    agent = build_agent(verbose=2)
    new_query = f'[Email: {email}, query: {query}, topic: {topic}, mode: {mode}, history: {include_history}]'
    result = agent.run(new_query)
    return result
if __name__ == "__main__":
    main()
