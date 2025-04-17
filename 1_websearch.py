# Standard library imports
import os
from typing import Annotated

# Third-party imports
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
import json

from langchain_core.messages import ToolMessage

from langchain_community.tools.tavily_search import TavilySearchResults

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
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def stream_graph_updates(graph: StateGraph, user_input: str) -> None:
    """Stream updates from the graph for a given user input."""
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        # state is a list of messages
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        # messages is a list of messages
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        # ai message has tool calls, so route to the tool node
        return "tools"
    return END


# Create a function to actually run the tools if they are called. We'll do this by adding the tools to a new node.
class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


if __name__ == "__main__":



    # Initialize the TavilySearchResults tool
    tool = TavilySearchResults(max_results=2)
    tools = [tool]

    # initialize the LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        api_key=os.environ.get("OPENAI_API_KEY")
    )

    # bind the tools to the LLM
    llm_with_tools = llm.bind_tools(tools)

    # Initialize graph and LLM
    graph_builder = StateGraph(State)

    # Build graph
    graph_builder.add_node("chatbot", chatbot)
    tool_node = BasicToolNode(tools=[tool])
    graph_builder.add_node("tools", tool_node)

    # The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
    # it is fine directly responding. This conditional routing defines the main agent loop.
    graph_builder.add_conditional_edges(
        source="chatbot",
        path=route_tools,
        # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
        # It defaults to the identity function, but if you
        # want to use a node named something else apart from "tools",
        # You can update the value of the dictionary to something else
        # e.g., "tools": "my_tools"
        path_map={"tools": "tools", END: END},
    )

    # Any time a tool is called, we return to the chatbot to decide the next step. 
    # (chatbot may call a tool but a tool must return to the chatbot)
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    graph = graph_builder.compile()


    user_input = "What is the price of bitcoin?"
    print("User: " + user_input)
    stream_graph_updates(graph,user_input)
    save_graph_visualization(graph, "assets/1_graph_visualization.png")