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
import requests
from langchain_community.utilities import SQLDatabase
from langchain import hub
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

load_dotenv('./../.env')

# Download the Chinook database
url = 'https://github.com/laxmimerit/All-CSV-ML-Data-Files-Download/raw/refs/heads/master/db_samples/Chinook.db'
response = requests.get(url)

if response.status_code == 200:
  if os.path.isfile('Chinook.db') == False:
    with open('Chinook.db', 'wb') as file:
      file.write(response.content)
else:
  print('Failed to download the file')
  print(response.status_code)

# LLM Connection and MySQL Tools
db = SQLDatabase.from_uri('sqlite:///Chinook.db')

#print(db.get_usable_table_names())
#print(db.get_table_info())

#print(db.run('SELECT * FROM album LIMIT 5'))

#print(db.run('SELECT * FROM artist LIMIT 2'))

#print(db.run('SELECT * FROM Invoice AS inv JOIN Customer as c on inv.CustomerId = c.CustomerId'))

# LLM Connection

llm = ChatOllama(model='qwen2.5', base_url = 'http://localhost:11434')

#print(llm.invoke('Hello'))

# Application State
class State(TypedDict):
  question: str # user question
  query: str  # mysql query prepared by LLM
  result: str # mysql result
  answer: str # LLM answer

### https://smith.langchain.com/hub/langchain-ai/sql-query-system-prompt
query_prompt_template = hub.pull('langchain-ai/sql-query-system-prompt')

print(query_prompt_template.messages[0].pretty_print())

# Write, Execute and Generate MySQL Response
## Write Node for MySQL Query
class QueryOutput(TypedDict):
  """Generated SQL query"""
  query: Annotated[str, ..., 'Syntactically correct and valid SQL query']

print(QueryOutput({'query': 'SELECT * FROM album LIMIT 2'}), QueryOutput.__annotations__)

def write_query(state: State):
  """Generate MySQL query to fetch information"""
  prompt = query_prompt_template.invoke({
    'dialect': db.dialect,
    'top_k': 5,
    #'table_info': db.get_table_info(table_names=['Album']),
    'table_info': db.get_table_info(),
    'input': state['question'],
  })
  structured_llm = llm.with_structured_output(QueryOutput)
  print('====== prompt ======')
  print(prompt)
  #result = llm.invoke(prompt)
  result = structured_llm.invoke(prompt)
  print('====== result ======')
  print(result)
  return {'query': result['query']}

q = write_query({'question': 'List all the albums'})
print(q)

## Execute Query
db.run('SELECT COUNT(*) AS EmployeeCount FROM Employee')

def execute_query(state: State):
  """Execute SQL query and return the result"""
  query = state['query']
  execute_query_tool = QuerySQLDataBaseTool(db=db)
  
  return {'result': execute_query_tool.invoke({'query': query})}

execute_query({'query': 'SELECT * FROM album LIMIT 2'})

# Generate Answer
print('=== Generate Answer ===')
def generate_answer(state: State):
  """Generate answer using retrieved information as the context"""
  
  prompt = (
    'Given the following user question, corresponding SQL query, and SQL result, answer the user question.\n\n'
    f"Question: {state['question']}\n"
    f"SQL Query: {state['query']}\n"
    f"SQL Result: {state['result']}"
  )
  
  response = llm.invoke(prompt)
  
  return {'answer': response.content}

question = 'List all the albums' #'How many employees are there?'
query = write_query({'question': question})
print(query)

result = execute_query(query)
print(result)

state = {'question': question, **query, **result}
print(state)

generate_answer(state)

# Building Graph
from langgraph.graph import START, END, StateGraph
from PIL import Image

graph_builder = StateGraph(State)
graph_builder.add_node('write_query', write_query)
graph_builder.add_node('execute_query', execute_query)
graph_builder.add_node('generate_answer', generate_answer)

graph_builder.add_edge(START, 'write_query')
graph_builder.add_edge('write_query', 'execute_query')
graph_builder.add_edge('execute_query', 'generate_answer')

graph = graph_builder.compile()

img = Image.open(io.BytesIO(graph.get_graph().draw_mermaid_png()))
img.show()

query = {'question': 'List all the albums'}
for step in graph.stream(query, stream_mode='updates'):
  print(step)

# LangGraph Agents
## - They can query the database as many times as needed to answer the user question.
## - They can recover from errors by running a generated query, catching the traceback and regenerating it correctly.
## - They can answer questions based on the databases' scheema as well as on the databases' content (like describing a specific table).


prompt = hub.pull('langchain-ai/sql-agent-system-prompt')
print(prompt.messages[0].pretty_print())

system_prompt = prompt.invoke({'dialect': db.dialect, 'top_k': 5})
system_prompt = prompt.format(dialect = db.dialect, top_k = 5                                                                                                                                                                                                                                                                                                                                                                       )

from langchain_community.agent_toolkits import SQLDatabaseToolkit

toolkit = SQLDatabaseToolkit(db = db, llm = llm)
print(toolkit.get_context())

tools = toolkit.get_tools()
print(tools)

print(tools[0].invoke('SELECT * FROM Album LIMIT 2'))
print(tools[1].invoke('Album, Customer'))

### Agent Coding
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

agent_executor = create_react_agent(llm, tools, state_modifier=system_prompt)

img = Image.open(io.BytesIO(agent_executor.get_graph().draw_mermaid_png()))
img.show()

question = "Which country's customers have made the most purchases?"
query = {'messages': [HumanMessage(question)]}

for step in agent_executor.stream(query, stream_mode='updates'):
  step['messages'][-1].pretty_print()