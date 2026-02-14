import streamlit as st
from langgraph_backend import chatbot, retrieve_all_chat_threads, submit_async_task
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from rag_utils import ingest_pdf, thread_document_metadata
import uuid
import queue

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
    st.session_state["ingested_docs"].setdefault(str(thread_id), {})
    
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
    st.session_state.chat_threads = retrieve_all_chat_threads()

if "chat_thread_order" not in st.session_state:
    st.session_state.chat_thread_order = []

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

if st.session_state.chat_threads and not st.session_state.chat_thread_order:
    st.session_state.chat_thread_order = list(st.session_state.chat_threads.keys())

add_chat_thread(st.session_state.thread_id)

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})

# Main UI
if len(st.session_state.message_history) > 0:   
    for msg in st.session_state.message_history:
        if msg["content"] == "":
            continue
        with st.chat_message(msg["role"]):
            st.text(msg["content"])
else:
    st.markdown(
        """
        <div style="text-align:center; padding: 3rem 1rem; color: #9aa0a6;">
            <div style="font-size: 3rem; font-weight: 600; margin-bottom: 0.5rem;">
                How can I help you today?
            </div>
            <div style="font-size: 1.2rem;">
                Start a new chat to get started.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

user_input = st.chat_input("Ask about your document or use tools")

if user_input:
    st.session_state.message_history.append({"role": "user", "content": user_input})
    thread_info = st.session_state.chat_threads.get(st.session_state.thread_id)
    if thread_info and not thread_info["latest_message"]:
        thread_info["latest_message"] = user_input
    
    with st.chat_message("user"):
        st.text(user_input)
    
    CONFIG = {"configurable": {"thread_id": st.session_state.thread_id}}
    with st.chat_message("assistant"):
        # def ai_only_stream():
        #     for message_chunk, metadata in chatbot.stream(
        #         {"messages": [HumanMessage(content=user_input)]}, config=CONFIG, stream_mode="messages"
        #     ):
        #         if isinstance(message_chunk, AIMessage):
        #             yield message_chunk.content
        status_holder = {"box": None}
        def ai_only_stream():
            event_queue = queue.Queue()

            async def run_stream():
                try:
                    async for message_chunk, metadata in chatbot.astream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=CONFIG,
                        stream_mode="messages",
                    ):
                        event_queue.put((message_chunk, metadata))
                except Exception as e:
                    event_queue.put((None, str(e)))
                finally:
                    event_queue.put(None)

            submit_async_task(run_stream())

            while True:
                event = event_queue.get()
                if event is None:
                    break
                message_chunk, metadata = event

                # Stream failed with an exception
                if message_chunk is None and isinstance(metadata, str):
                    yield f"Error: {metadata}"
                    continue

                # Lazily create & update the SAME status container when any tool runs
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content
        ai_message = st.write_stream(ai_only_stream())
        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    st.session_state.message_history.append({"role": "assistant", "content": ai_message})

    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"Document indexed: {doc_meta.get('filename')} "
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )

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

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chart", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF...", expanded = True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)
            st.rerun()

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