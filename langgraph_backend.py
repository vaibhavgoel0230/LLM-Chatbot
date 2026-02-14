from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langgraph_tools import search_tool, calculator_tool, client
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver 
from dotenv import load_dotenv
import aiosqlite
import requests
import asyncio
import threading

load_dotenv()

# Dedicated async loop for backend tasks
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()

def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)

def run_async(coro):
    return _submit_async(coro).result()

def submit_async_task(coro):
    return _submit_async(coro)

# ------ State Definition ------
class ChatState(TypedDict):
  messages: Annotated[list[BaseMessage], add_messages]


# ------ LLM Initialization ------
llm = ChatOpenAI(model='gpt-4o-mini')

# ------ Tools Binding ------
def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(client.get_tools())
    except Exception as e:
        return []

mcp_tools = load_mcp_tools()

tools = [search_tool, calculator_tool, *mcp_tools]
llm = llm.bind_tools(tools) if tools else llm


# ------ Nodes ------
async def chat_node(state: ChatState):
  """LLM chat node that may answer or use the search and calculator tools"""
  messages = state['messages']
  response = await llm.ainvoke(messages)
  return {'messages': [response]}       

tool_node = ToolNode(tools) if tools else None # ToolNode is a prebuilt node that can be used to call tools

# ------ Checkpointer ------
async def _init_checkpointer():
    conn = await aiosqlite.connect(database='chatbot.db')
    return AsyncSqliteSaver(conn=conn)

checkpointer = run_async(_init_checkpointer())

# ------ Graph ------
graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')

if tool_node:
    graph.add_node('tools', tool_node)
    graph.add_conditional_edges('chat_node', tools_condition)
    graph.add_edge('tools', 'chat_node')

graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)

async def _alist_threads():
    all_threads = {}
    # alist yields newest-first. Use checkpoint data directly instead of
    # chatbot.get_state(): AsyncSqliteSaver raises if sync get_tuple() is
    # called from the same event loop (e.g. from this async function).
    async for checkpoint_tuple in checkpointer.alist(None):
        thread_id = checkpoint_tuple.config["configurable"]["thread_id"]
        if thread_id in all_threads:
            continue  # already have latest for this thread
        checkpoint = checkpoint_tuple.checkpoint
        channel_values = checkpoint.get("channel_values", {}) if isinstance(checkpoint, dict) else getattr(checkpoint, "channel_values", {})
        messages = channel_values.get("messages", []) if isinstance(channel_values, dict) else []
        first_message = messages[0].content if messages else None
        all_threads[thread_id] = {"thread_id": thread_id, "latest_message": first_message}

    print(all_threads)
    return all_threads

def retrieve_all_chat_threads():
    return run_async(_alist_threads())