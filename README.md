🐉 Game of Thrones RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built using LangChain, FAISS, Hugging Face Embeddings, and Streamlit.

This project answers questions from a Game of Thrones text document (got.txt) using semantic search and an LLM.

🚀 Features

Loads custom .txt document

Splits text using RecursiveCharacterTextSplitter

Creates embeddings using sentence-transformers/all-MiniLM-L6-v2

Stores embeddings in FAISS vector database

Uses Hugging Face LLM (Qwen/Qwen3-Coder-Next-FP8)

Interactive Streamlit UI

Persistent FAISS index storage

🏗️ Tech Stack

Python

LangChain

FAISS

Hugging Face Hub

Sentence Transformers

Streamlit

dotenv
