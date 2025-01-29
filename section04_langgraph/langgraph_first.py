from dotenv import load_dotenv
import os
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
#from IPython.display import display, Image
from PIL import Image
import io

load_dotenv('./../.env')

llm = ChatOllama(model='llama3.2', base_url = 'http://localhost:11434')

# print(llm.invoke('Hi'))

class State(TypedDict):
  messages: Annotated[list, add_messages]

def chatbot(state: State):
  response = llm.invoke(state['messages'])
  return {'messages': [response]}

graph_builder = StateGraph(State)
graph_builder.add_node('chatbot', chatbot)

graph_builder.add_edge(START, 'chatbot')

graph_builder.add_edge('chatbot', END)

graph = graph_builder.compile()

#display(Image(graph.get_graph().draw_mermaid_png()))
img = Image.open(io.BytesIO(graph.get_graph().draw_mermaid_png()))
img.show()

res = graph.invoke({'messages': ['Hi']})
print(res)

res = graph.invoke({'messages': ['Hi', 'My name is Naoki.']})
print(res)

while True:
  user_input = input('You: ')
  if user_input in ['q', 'quit', 'exit']:
    print('Bye!')
    break
  
  response = graph.invoke({'messages': [user_input]})
  print('Assistant: ', response['messages'][-1].content)