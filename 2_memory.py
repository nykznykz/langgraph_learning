# https://langchain-ai.github.io/langgraph/reference/checkpoints/#langgraph.checkpoint.memory.MemorySaver
from langgraph.checkpoint.memory import MemorySaver 
from typing import Annotated

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from graphviz import save_graph_visualization
from dotenv import load_dotenv
import os

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}




if __name__ == "__main__":

    # Initialize graph
    graph_builder = StateGraph(State)

    # Add nodes
    tool = TavilySearchResults(max_results=2)
    tools = [tool]

    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        api_key=os.environ.get("OPENAI_API_KEY")
    )

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    # Add nodes
    graph_builder.add_node("chatbot", chatbot)

    # Add tool node
    tool_node = ToolNode(tools=[tool])
    graph_builder.add_node("tools", tool_node)

    # Add conditional edges
    graph_builder.add_conditional_edges(
        source="chatbot",
        path=tools_condition,
    )
    # Any time a tool is called, we return to the chatbot to decide the next step
    graph_builder.add_edge("tools", "chatbot")

    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "1"}}

    # Define user input
    user_input = "Hi there! My name is Will."

    # The config is the **second positional argument** to stream() or invoke()!
    events = graph.stream(
        input={"messages": [{"role": "user", "content": user_input}]},
        config=config,
        stream_mode="values",
    )
    for event in events:
        event["messages"][-1].pretty_print()

    user_input = "Remember my name?"

    # The config is the **second positional argument** to stream() or invoke()!
    events = graph.stream(
        input={"messages": [{"role": "user", "content": user_input}]},
        config=config,
        stream_mode="values",
    )
    for event in events:
        event["messages"][-1].pretty_print()

    # The only difference is we change the `thread_id` here to "2" instead of "1"
    events = graph.stream(
        input={"messages": [{"role": "user", "content": user_input}]},
        config={"configurable": {"thread_id": "2"}},
        stream_mode="values",
    )
    for event in events:
        event["messages"][-1].pretty_print()

    snapshot = graph.get_state(config)
    print(snapshot)

    save_graph_visualization(graph, "assets/2_graph_visualization.png")