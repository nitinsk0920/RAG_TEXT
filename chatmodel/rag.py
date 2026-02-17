from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
loader = TextLoader('got.txt', encoding='utf-8')
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " "]
)

chunks = splitter.split_documents(docs)


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local("faiss_index")
print("FAISS index saved successfully!")
