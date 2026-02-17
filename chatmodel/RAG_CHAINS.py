from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st
# Same embedding model MUST be used
load_dotenv()
st.header('GOT CHATBOT')
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-Next-FP8",
     task="text-generation",
    max_new_tokens=300
)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load existing FAISS DB
vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

print("Retriever loaded successfully!")


prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, try to get information from your learning.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

question          = st.text_input("Enter your question")
retrieved_docs    = retriever.invoke(question)

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parser= StrOutputParser()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-Next-FP8",
     task="text-generation",
    max_new_tokens=300
)
model = ChatHuggingFace(llm=llm)
main_chain = parallel_chain | prompt | model | parser

if st.button("Send"):
    result=main_chain.invoke(question)
    st.write(result)

