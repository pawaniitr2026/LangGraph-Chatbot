import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph_backend import chatbot

st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="💬",
    layout="centered",
)

st.title("💬 CHATBOT BY PAWAN")

# The same thread_id lets LangGraph's checkpointer retain the conversation.
CONFIG = {"configurable": {"thread_id": "thread-1"}}

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# Display previous messages.
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Type your message...")

if user_input:
    # Display and store the user's message.
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.write(user_input)

    try:
        response = chatbot.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=CONFIG,
        )

        # LangGraph returns the state under the "messages" key.
        ai_message = response["messages"][-1].content

        st.session_state["message_history"].append(
            {"role": "assistant", "content": ai_message}
        )

        with st.chat_message("assistant"):
            st.write(ai_message)

    except Exception as e:
        st.error(f"Error while calling the LLM: {e}")
