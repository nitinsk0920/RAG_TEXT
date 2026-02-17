# 🐉 Game of Thrones RAG Chatbot

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




⚙️ Setup Instructions
1️⃣ Clone Repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

2️⃣ Create Virtual Environment
python -m venv venv

Activate:
Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

3️⃣ Install Dependencies
pip install langchain langchain-community langchain-core langchain-huggingface faiss-cpu sentence-transformers streamlit python-dotenv
pip install requirements.txt

4️⃣ Add Hugging Face API Token
Create .env file:
HUGGINGFACEHUB_API_TOKEN=your_token_here

🧠 Create FAISS Index

Run:
python create_index.py

This will:
Load got.txt
Split into chunks
Generate embeddings
Save FAISS index

💬 Run the Chatbot
streamlit run app.py

Open:
http://localhost:8501
Ask Game of Thrones related questions 🎯

