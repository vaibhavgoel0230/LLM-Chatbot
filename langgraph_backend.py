from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langgraph_tools import search_tool, calculator_tool
from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver 
from dotenv import load_dotenv
import sqlite3

load_dotenv()

class ChatState(TypedDict):
  messages: Annotated[list[BaseMessage], add_messages]

llm = ChatOpenAI(model='gpt-4o-mini')
tools = [search_tool, calculator_tool]
llm = llm.bind_tools(tools)

def chat_node(state: ChatState):
  """LLM chat node that may answer or use the search and calculator tools"""
  messages = state['messages']
  response = llm.invoke(messages)
  return {'messages': [response]}       

tool_node = ToolNode(tools) # ToolNode is a prebuilt node that can be used to call tools

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)
graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_chat_threads():
    all_threads = {}
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
        thread_ids = [row[0] for row in cursor.fetchall()]
    except sqlite3.OperationalError:
        return all_threads

    for thread_id in thread_ids:
        state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
        messages = state.values.get("messages", [])
        first_message = messages[0].content if messages else None
        all_threads[thread_id] = {"thread_id": thread_id, "latest_message": first_message}

    return all_threads