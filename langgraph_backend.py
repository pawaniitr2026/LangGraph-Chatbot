import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
model_name = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")

if not api_key:
    raise ValueError(
        "OPENROUTER_API_KEY is missing. "
        "Add it to the .env file and restart Streamlit."
    )

llm = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    base_url=base_url,
    temperature=0,
)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


def stream_chat(user_input: str, thread_id: str = "thread-1"):
    """Stream assistant response text chunks to Streamlit."""
    config = {"configurable": {"thread_id": thread_id}}

    for message_chunk, metadata in chatbot.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
        stream_mode="messages",
    ):
        if metadata.get("langgraph_node") == "chat_node":
            content = message_chunk.content

            if isinstance(content, str) and content:
                yield content
