import streamlit as st
import time
from langgraph_backend import stream_chat

st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="💬",
    layout="centered",
)

#st.title("💬 CHATBOT BY PAWAN")
st.markdown(
    """
    <style>
    .main-title {
        font-family: "Trebuchet MS", sans-serif;
        font-size: 40px;
        font-weight: 700;
        text-align: center;
        letter-spacing: 1px;
        margin-top: -20px;
        margin-bottom: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<h1 class="main-title">💬 CHATBOT BY PAWAN</h1>',
    unsafe_allow_html=True
)

THREAD_ID = "thread-1"

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            for chunk in stream_chat(user_input, thread_id=THREAD_ID):
                for char in chunk:
                    full_response += char
                    response_placeholder.markdown(full_response + " ")
                    time.sleep(0.02)

            response_placeholder.markdown(full_response)

            st.session_state["message_history"].append(
                {"role": "assistant", "content": full_response}
            )

        except Exception as e:
            response_placeholder.empty()
            st.error(f"Error while calling the LLM: {e}")
