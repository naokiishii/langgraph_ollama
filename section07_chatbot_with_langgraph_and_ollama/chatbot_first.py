from dotenv import load_dotenv
import os
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from PIL import Image
import io

load_dotenv('./../.env')

llm = ChatOllama(model='llama3.2', base_url = 'http://localhost:11434')

# print(llm.invoke('Hi'))

class State(TypedDict):
  # Messages have the type 'list'.
  # The `add_messages` function in the annotation defines how this state key should be updated.
  # (in this case, it appends messages to the list, rather than overwriting them)
  messages: Annotated[list, add_messages]

def chatbot(state: State):
  response = llm.invoke(state['messages'])
  return {'messages': [response]}

# A `State Graph` object defines the structure of our chatbot as a "state machine".
# We'll add `nodes` to represent the llm and functions our chatbot can call and `edges` to specify how the bot should transition between these functions.
graph_builder = StateGraph(State)
graph_builder.add_node('chatbot', chatbot)

graph_builder.add_edge(START, 'chatbot')

graph_builder.add_edge('chatbot', END)

graph = graph_builder.compile()

res = graph.invoke({'messages': ['Hi', 'My name is Naoki.']})
print(res)

# Enhancing the Chatbot 3with Tools and Memory - Making it More Like an Agent
# https://github.com/laxmimerit/Langchain-and-Ollama
@tool
def internet_search(query: str):
  """
  Search the web for realtime and latest information.
  For example, new, stock market, weather updates etc.
  
  Args:
  query: The search query
  """
  search = TavilySearchResults(
    max_results = 3,
    search_depth = 'advanced',
    include_answer = True,
    include_raw_content = True,
  )

  response = search.invoke(query)
  return response

@tool
def llm_search(query: str):
  """
  Use the LLM model for general and basic information
  """
  message = HumanMessage(query)
  response = llm.invoke(message)
  return response

tools = [
  internet_search,
  llm_search
]

llm_with_tools = llm.bind_tools(tools)

class State2(TypedDict):
  messages: Annotated[list, add_messages]

def chatbot2(state: State2):
  response = llm_with_tools.invoke(state['messages'])
  return {'messages': [response]}

memory = MemorySaver()

graph_builder2 = StateGraph(State2)
graph_builder2.add_node('chatbot2', chatbot)
tool_node = ToolNode(tools = tools)
graph_builder2.add_node('tools', tool_node)
graph_builder2.add_conditional_edges('chatbot2', tools_condition)
graph_builder2.add_edge('tools', 'chatbot2')
graph_builder2.set_entry_point('chatbot2')

graph2 = graph_builder2.compile(checkpointer = memory)

#display(Image(graph.get_graph().draw_mermaid_png()))
img = Image.open(io.BytesIO(graph2.get_graph().draw_mermaid_png()))
img.show()

# without memory
# res = graph2.invoke({'messages': ['Tell me what you can do?']})
# print(res)

# with memory
config = {'configurable': {'thread_id': 1}}
while True:
  user_input = input()
  if user_input in ['exit', 'quit', 'q']:
    print('Exiting...')
    break
  
  res = graph2.invoke({'messages': [user_input]}, config = config)
  print(res['messages'][-1].pretty_print())