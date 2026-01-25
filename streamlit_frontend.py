from importlib.metadata import metadata
import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage, AIMessage
import uuid

def generate_thread_id():
    thread_id = str(uuid.uuid4())
    return thread_id

def trim_label(label, max_length=40):
    normalized = " ".join(label.split())
    if len(normalized) <= max_length:
        return normalized
    if max_length <= 3:
        return normalized[:max_length]
    return f"{normalized[:max_length - 3]}..."

def new_chat():
    new_thread_id = generate_thread_id()
    st.session_state.thread_id = new_thread_id
    add_chat_thread(new_thread_id)
    st.session_state.message_history = []

def add_chat_thread(thread_id):
    chat_threads = st.session_state.chat_threads
    if thread_id not in chat_threads:
        chat_threads[thread_id] = {"thread_id": thread_id, "latest_message": None}
        st.session_state.chat_thread_order.append(thread_id)

def select_chat_thread(thread_id):
    st.session_state.thread_id = thread_id
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", [])

    message_history = []
    for message in messages:
        if isinstance(message, HumanMessage):
            message_history.append({"role": "user", "content": message.content})
        elif isinstance(message, AIMessage):
            message_history.append({"role": "assistant", "content": message.content})
    st.session_state.message_history = message_history
    if messages:
        first_message = messages[0].content
        thread_info = st.session_state.chat_threads.get(thread_id)
        if thread_info and not thread_info["latest_message"]:
            thread_info["latest_message"] = first_message

# Session Setup
if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state.chat_threads = {}

if "chat_thread_order" not in st.session_state:
    st.session_state.chat_thread_order = []

add_chat_thread(st.session_state.thread_id)
# Main UI
for msg in st.session_state.message_history:
    with st.chat_message(msg["role"]):
        st.text(msg["content"])

user_input = st.chat_input("Enter a message")

if user_input:
    st.session_state.message_history.append({"role": "user", "content": user_input})
    thread_info = st.session_state.chat_threads.get(st.session_state.thread_id)
    if thread_info and not thread_info["latest_message"]:
        thread_info["latest_message"] = user_input
    
    with st.chat_message("user"):
        st.text(user_input)
    
    CONFIG = {"configurable": {"thread_id": st.session_state.thread_id}}
    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]}, config=CONFIG, stream_mode="messages"
            )
        )

    st.session_state.message_history.append({"role": "assistant", "content": ai_message})

# Sidebar UI
st.sidebar.title("LangGraph Chatbot")
st.sidebar.markdown("This is a chatbot powered by LangGraph and OpenAI.")
st.sidebar.markdown(
    """
    <style>
      [data-testid="stSidebar"] .stButton button {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }
      [data-testid="stSidebar"] .stButton button > div {
        overflow: hidden;
      }
      [data-testid="stSidebar"] .stButton button p {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }
    </style>
    """,
    unsafe_allow_html=True,
)
st.sidebar.button('New Chat', on_click=new_chat)
st.sidebar.header("My Chats")

for thread_id in st.session_state.chat_thread_order[::-1]:
    thread_info = st.session_state.chat_threads.get(thread_id)
    if not thread_info:
        continue
    is_active = thread_id == st.session_state.thread_id
    latest_message = thread_info["latest_message"]
    if latest_message:
        st.sidebar.button(
            trim_label(latest_message),
            on_click=select_chat_thread,
            args=(thread_id,),
            type="primary" if is_active else "secondary",
            use_container_width=True,
        )