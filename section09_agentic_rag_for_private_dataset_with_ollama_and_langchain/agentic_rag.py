from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import faiss
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
import os
import warnings
from dotenv import load_dotenv
from PIL import Image
import io

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings('ignore')

load_dotenv('./../.env')

model = 'llama3.2'
llm = ChatOllama(model=model, base_url='http://localhost:11434')
#print(llm.invoke('Hello, how are you?'))

embeddings = OllamaEmbeddings(
    model='nomic-embed-text',
    base_url='http://localhost:11434'
)

db_name = 'health_supplements'
vector_store = FAISS.load_local(db_name, embeddings, allow_dangerous_deserialization=True)

retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 5})

question = 'how to gain muscle mass?'
result = retriever.invoke(question)
print(result)

## Create Retriever as Tool Node
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
  retriever,
  'health_supplements',
  'Search and return information about the Health Supplements for workout and gym',
)

tools = [retriever_tool]

## We can lay out an agentic RAG graph like this:
## - The stte is a set of messages
## - Each node will update (append to) state
## - Conditional edges decide which node to visit next
from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class State(TypedDict):
  messages: Annotated[Sequence[BaseMessage], add_messages]

from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from pydantic import BaseModel, Field
from langgraph.prebuilt import tools_condition

def grade_documents(state) -> Literal['generate', 'rewrite']:
  """
  Determines whether the retrieved documents are relevant to the question.
  
  Args:
    state (messages): The current stte
  
  Return:
    str: A decision for whether the documents are relevant or not
  """
  
  print('---CHECK RELEVANCE---')
  
  # Data model
  class grade(BaseModel):
    """Binary score for relevance check."""
    
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")
  
  # LLM with tool and validation
  llm_with_structured_output = llm.with_structured_output(grade)
  
  # Prompt
  prompt = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question.\n
    Here is the retrieved document: \n\n {context} \n\n
    Here is the user question: {question} \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.\n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
    input_variables=['context', 'question'],
  )
  
  # Chain
  chain = prompt | llm_with_structured_output
  
  messages = state['messages']
  last_message = messages[-1]
  
  question = messages[0].content
  docs = last_message.content
  
  scored_result = chain.invoke({'question': question, 'context': docs})
  
  score = scored_result.binary_score
  
  if score == 'yes':
    print('---DECISION: DOCS RELEVANT---')
    return 'generate'
  else:
    print('---DECISION: DOCS NOT RELEVANT---')
    print(score)
    return 'rewrite'

# Agent Node
def agent(state):
  """
  Invokes the agent model to generate a response based on the current state.
  Given the question, it will decide to retrieve using the retriever tool, or simply end.
  
  Args:
    state (messages): The current state
  
  Returns:
    dict: The updated state with the agent response appended to messages
  """
  print('---CALL AGENT---')
  messages = state['messages']
  
  llm_with_tools = llm.bind_tools(tools, tool_choice='required')
  response = llm_with_tools.invoke(messages)
  
  # We return a list, because this will get added to the existing list
  return {'messages': [response]}

# Rewrite Node
def rewrite(state):
  """
  Transform the query to produce a better question.
  
  Args:
    state (messages): The current state
  
  Returns:
    dict: The updated state with re-phrased question
  """
  
  print('---TRANSFORM QUERY---')
  messages = state['messages']
  question = messages[0].content
  
  msg = [
    HumanMessage(
      content=f"""\n
      Look at the input and try to reason about the underlying semantic intent / meaning.\n
      Here is the initial question:
      \n ------- \n
      {question}
      \n ------- \n
      Formulate an improved question: """,
    )
  ]
  
  # Grader
  response = llm.invoke(msg)
  return {'messages': [response]}

# Generate Node
def generate(state):
  """
  Generate answer
  
  Args:
    state (messages): The current state
  
  Returns:
    dict: The updated state with re-phrased question
  """
  print('---GENERATE---')
  messages = state['messages']
  question = messages[0].content
  last_message = messages[-1]
  
  docs = last_message.content
  
  # Prompt
  prompt = hub.pull('rlm/rag-prompt')
  
  # Post-processing
  def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)
  
  # Chain
  rag_chain = prompt | llm | StrOutputParser()
  
  # Run
  response = rag_chain.invoke({'context': docs, 'question': question})
  return {'messages': [response]}

# Graph
## - Start with an agent, call_model
## - Agent make a decision to call a function
## - If so, then action to call tool (retriever)
## - Then call agent with the tool output added to messages (state)
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode

graph_builder = StateGraph(State)

graph_builder.add_node('agent', agent)
retriever = ToolNode([retriever_tool])
graph_builder.add_node('retriever', retriever)
graph_builder.add_node('rewrite', rewrite)

graph_builder.add_node('generate', generate)

graph_builder.add_edge(START, 'agent')

graph_builder.add_conditional_edges(
  'agent',
  tools_condition,
  {
    'tools': 'retriever',
    END: END
  }
)

graph_builder.add_conditional_edges(
  'retriever',
  grade_documents,
)

graph_builder.add_edge('generate', END)
graph_builder.add_edge('rewrite', 'agent')

graph = graph_builder.compile()

img = Image.open(io.BytesIO(graph.get_graph().draw_mermaid_png()))
img.show()

# Run
from pprint import pprint

#query = {'messages': [HumanMessage(content='How to gain muscle mass?')]}
#query = {'messages': [HumanMessage(content='What are the risks of taking too much protein?')]}
query = {'messages': [HumanMessage(content='Tell me about the LangChain')]}

# state = graph.invoke(query)

for output in graph.stream(query):
  for key, value in output.items():
    pprint(f"Output from node '{key}':")
    pprint('----')
    pprint(value, indent=4, width=120)
  pprint('\n------\n')