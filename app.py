import os
import streamlit as st
from dotenv import load_dotenv
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing import List, TypedDict

# Load environment variables
load_dotenv()

# Initialize LLM
from langchain.chat_models import init_chat_model
llm = init_chat_model("llama3-8b-8192", model_provider="groq")

# Initialize embeddings model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Load RAG-specific prompt
prompt = hub.pull("rlm/rag-prompt")

# Define State format
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Define retrieval function
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

# Define generation function
def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Build and compile state graph
graph_builder = StateGraph(State)
graph_builder.add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Streamlit UI
st.title("\U0001F50D AI-Powered Q&A with RAG")
st.write("Enter a website URL and ask a question about its content.")

# User Input
url = st.text_input("Enter a URL:", "")
question = st.text_input("Ask a Question:", "")

if st.button("Submit") and url and question:
    try:
        # ✅ Fetch webpage content
        loader = WebBaseLoader(web_paths=[url])
        docs = loader.load()

        # ✅ Use better text chunking
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
        all_splits = text_splitter.split_documents(docs)

        # ✅ Save webpage content to a text file
        with open("fetched_content.txt", "w", encoding="utf-8") as file:
            for doc in all_splits:
                file.write(doc.page_content + "\n\n")

        # ✅ Create a new vector store instance (fix for `clear()`)
        global vector_store
        vector_store = InMemoryVectorStore(embeddings)
        vector_store.add_documents(documents=all_splits)

        # ✅ Invoke graph with user question
        response = graph.invoke({"question": question})

        # Display Answer
        st.subheader("Answer:")
        st.write(response["answer"])

        # Provide file download link
        with open("fetched_content.txt", "r", encoding="utf-8") as file:
            st.download_button("Download Webpage Content", file, file_name="webpage_content.txt")

    except Exception as e:
        st.error(f"Error: {e}")