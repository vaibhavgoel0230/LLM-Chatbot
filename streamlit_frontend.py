from importlib.metadata import metadata
import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage

thread_id = "123"
config = {"configurable": {"thread_id": thread_id}}


if "message_history" not in st.session_state:
    st.session_state.message_history = []

for msg in st.session_state.message_history:
    with st.chat_message(msg["role"]):
        st.text(msg["content"])

user_input = st.chat_input("Enter a message")

if user_input:
    st.session_state.message_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]}, config=config, stream_mode="messages"
            )
        )

    st.session_state.message_history.append({"role": "assistant", "content": ai_message})