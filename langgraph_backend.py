from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

class ChatState(TypedDict):
  messages: Annotated[list[BaseMessage], add_messages]

llm = ChatOpenAI(model='gpt-4o-mini')

def chat_node(state: ChatState):
  messages = state['messages']
  response = llm.invoke(messages)
  return {'messages': [response]}

checkpointer = MemorySaver()
graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)