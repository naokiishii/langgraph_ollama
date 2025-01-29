import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv('./../.env')
print(os.environ['LANGCHAIN_ENDPOINT'])

base_url = 'http://localhost:11434'
model = 'llama3.2'

llm = ChatOllama(
    model = model,
    temperature = 0.8,
    num_predict = 256,
    base_url = base_url,
    # other params ...
)

res = llm.invoke('Hello, how are you?')
print(res.content)

for chunk in llm.stream('Hello, how are you?'):
    print(chunk.content)