import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ Updated import
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["HUGGINGFACE_HUB_TOKEN"] = st.secrets["HUGGINGFACE_HUB_TOKEN"]  # ✅ Set the token
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama3-70b-8192"

# --- Function: Extract text from PDF ---
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# --- Function: Create FAISS vector store ---
def create_faiss_vector_store(text, path="faiss_index"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local(path)

# --- Function: Load FAISS vector store ---
def load_faiss_vector_store(path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return vector_store

# --- Function: Build QA chain ---
def build_qa_chain(vector_store_path="faiss_index"):
    vector_store = load_faiss_vector_store(vector_store_path)
    retriever = vector_store.as_retriever()
    llm = ChatGroq(api_key=GROQ_API_KEY, model_name=GROQ_MODEL)
    qa_chain = load_qa_chain(llm, chain_type="stuff")
    return RetrievalQA(retriever=retriever, combine_documents_chain=qa_chain)

# --- Streamlit UI ---
st.title("📄 RAG Chatbot with FAISS + Groq")
st.write("Upload a PDF and ask questions based on its content.")

uploaded_file = st.file_uploader("📎 Upload your PDF file", type="pdf")

if uploaded_file is not None:
    os.makedirs("uploaded", exist_ok=True)
    pdf_path = os.path.join("uploaded", uploaded_file.name)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info("📑 Extracting text and creating vector store...")
    text = extract_text_from_pdf(pdf_path)
    create_faiss_vector_store(text)

    st.info("⚙️ Initializing chatbot...")
    qa_chain = build_qa_chain()
    st.success("✅ Chatbot is ready!")

    question = st.text_input("🤔 Ask a question about the PDF:")
    if question:
        st.info("💬 Getting answer...")
        answer = qa_chain.run(question)
        st.success(f"🧠 Answer: {answer}")
