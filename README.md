# 🐉 Game of Thrones RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built using LangChain, FAISS, Hugging Face Embeddings, and Streamlit.
This project answers questions from a Game of Thrones text document (got.txt) using semantic search and an LLM.

## 🚀 Features
Loads custom .txt document<br>
Splits text using RecursiveCharacterTextSplitter<br>
Creates embeddings using sentence-transformers/all-MiniLM-L6-v2<br>
Stores embeddings in FAISS vector database <br>
Uses Hugging Face LLM (Qwen/Qwen3-Coder-Next-FP8) <br>
Interactive Streamlit UI<br>
Persistent FAISS index storage<br>

## 🏗️ Tech Stack
Python<br>
LangChain<br>
FAISS<br>
Hugging Face Hub<br>
Sentence Transformers<br>
Streamlit<br>
dotenv<br>




# ⚙️ Setup Instructions
## 1️⃣ Clone Repository
git clone https://github.com/your-username/your-repo-name.git<br>
cd your-repo-name

## 2️⃣ Create Virtual Environment
python -m venv venv

## Activate:
Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

## 3️⃣ Install Dependencies
pip install langchain langchain-community langchain-core langchain-huggingface faiss-cpu sentence-transformers streamlit python-dotenv<br>
pip install -r requirements.txt

## 4️⃣ Add Hugging Face API Token
Create .env file:
HUGGINGFACEHUB_API_TOKEN=your_token_here

## 🧠 Create FAISS Index

Run:
python create_index.py

This will:<br>
Load got.txt<br>
Split into chunks<br>
Generate embeddings<br>
Save FAISS index<br>

## 💬 Run the Chatbot
streamlit run app.py

Open:
http://localhost:8501
Ask Game of Thrones related questions 🎯

