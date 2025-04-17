# Standard library imports
import os
from typing import Annotated

# Third-party imports
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

from graphviz import save_graph_visualization

# Load environment variables
load_dotenv()


class State(TypedDict):
    """State type for the chat graph.
    
    Messages have the type "list". The `add_messages` function in the annotation 
    defines how this state key should be updated (appends messages rather than overwriting).
    """
    messages: Annotated[list, add_messages]


def chatbot(state: State) -> dict:
    """Process messages in state through LLM and return response."""
    return {"messages": [llm.invoke(state["messages"])]}

def stream_graph_updates(graph: StateGraph, user_input: str) -> None:
    """Stream updates from the graph for a given user input."""
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)



if __name__ == "__main__":

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        api_key=os.environ.get("OPENAI_API_KEY")
    )


    # Initialize graph and LLM
    graph_builder = StateGraph(State)

    # Build graph
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    graph = graph_builder.compile()


    # Example usage
    user_input = "What do you know about LangGraph?"
    print("User:", user_input)
    stream_graph_updates(graph, user_input)
    save_graph_visualization(graph, "assets/0_graph_visualization.png")