# chatbot11

# RAG-Based Chatbot using LangChain, FAISS, and Transformers

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot using **LangChain**, **FAISS**, **SentenceTransformers**, and a **transformer-based language model**. The chatbot is capable of understanding user queries and answering based on content from a given PDF or text document.



## Features
```
- Ingests documents like PDFs using `PyPDFLoader`
- Chunks text using `RecursiveCharacterTextSplitter`
- Embeds text using `SentenceTransformers`
- Stores vectors using `FAISS` for fast retrieval
- Generates context-aware answers using GPT-2 or FLAN-T5
- Provides both CLI and GUI (Gradio) interfaces

```

## Installation

Install the required packages:


bash
pip install langchain openai faiss-cpu sentence-transformers pypdf gradio


 ## Project
 ```
chatbot/
│
├── mlpdf.pdf                     # Input knowledge base
├── faiss_index/                 # Saved FAISS index
├── chatbot.py                   # Main script
├── requirements.txt             # All dependencies
├── README.md                    # Project description
```

# How to Run

  # Prepare the Data:
    
    
```
    from langchain.document_loaders import PyPDFLoader
    loader = PyPDFLoader("mlpdf.pdf")
     documents = loader.load()
```
# Split & Embed:
```
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)
```

# Generate Embeddings & Build FAISS Index:
```

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")
```

# Chat Function with RAG:
    
```
from transformers import pipeline
def rag_chat(query):
    docs = vectorstore.similarity_search(query, k=3)
    context = " ".join([doc.page_content for doc in docs])
    qa = pipeline("text-generation", model="gpt2")
    response = qa(context + "\nQuestion: " + query, max_new_tokens=100)[0]['generated_text']
    return response
    



```
# Gradio Interface:


```
import gradio as gr
def chat_interface(query):
    return rag_chat(query)
gr.Interface(fn=chat_interface, inputs="text", outputs="text").launch()
```
