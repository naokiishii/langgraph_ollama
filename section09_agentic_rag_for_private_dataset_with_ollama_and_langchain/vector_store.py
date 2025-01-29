# pip install faiss-cpu
# https://github.com/laxmimerit/rag-dataset

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import faiss
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
import os
import warnings
from dotenv import load_dotenv

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings('ignore')

load_dotenv('./../.env')

# Document Loader

loader = PyMuPDFLoader(r'1. Analysis of Actual Fitness Supplement.pdf')
loader.load()
print(loader)

pdfs = []
for root, dirs, files in os.walk('./'):
    for file in files:
        if file.endswith('.pdf'):
            pdfs.append(os.path.join(root, file))
print(pdfs)

docs = []
for pdf in pdfs:
    loader = PyMuPDFLoader(pdf)
    temp = loader.load()
    docs.extend(temp)

print(docs)

# Document Chunking

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)

print(len(docs), len(chunks))
print(docs[0].metadata)
print(docs[0].page_content)
print(chunks[0].page_content)

# Document Vector Embedding
# https://github.com/facebookresearch/faiss

embeddings = OllamaEmbeddings(
    model='nomic-embed-text',
    base_url='http://localhost:11434'
)

vector = embeddings.embed_query(chunks[0].page_content)
print(vector)

# Storing Embedding in Vector Store
index = faiss.IndexFlatIP(len(vector))
print(index.ntotal, index.d)

vector_store = FAISS(embedding_function=embeddings,
                     index=index, docstore=InMemoryDocstore(), index_to_docstore_id={})
print(vector_store.index.ntotal)

ids = vector_store.add_documents(documents=chunks)
print(ids)

question = 'how to gain muscle mass?'
result = vector_store.search(query=question, k=5, search_type='similarity')
print(result)

db_name = 'health_supplements'
vector_store.save_local(db_name)