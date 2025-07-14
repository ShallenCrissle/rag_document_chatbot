# 🧠 RAG PDF Chatbot – LangChain + FAISS + Groq

This project is an interactive **PDF chatbot** that lets you upload a document and ask natural-language questions based on its content. It uses **semantic search + large language models (LLMs)** to return contextual answers.

---

## 🔁 Evolution

### ✅ Phase 1 – Local Ollama Setup
Initially, the chatbot used **Ollama** with the `phi3:mini` or `llama3.2` model for local inference. It worked well but had **heavy memory requirements** and limited portability.

### 🌐 Phase 2 – Migrated to Groq + LLaMA 3 (Cloud)
To improve speed, reduce RAM usage, and make it deployment-ready, the app was updated to use **Groq’s blazing-fast LLMs** (like `llama3-70b-8192`) via their **free API**.

---

## 🚀 Features

- 📄 Upload any PDF document
- 🔎 Perform **semantic search** with FAISS
- 💬 Ask natural questions — get smart, contextual answers
- ⚡ Powered by **Groq-hosted LLaMA 3 model**
- 🧠 Uses **Hugging Face sentence embeddings**
- 🎈 Built with **Streamlit** for a fast, responsive UI

---

## 🛠️ Tech Stack

| Layer              | Tool/Library                    |
|-------------------|----------------------------------|
| Frontend          | Streamlit                        |
| RAG Engine        | LangChain + FAISS                |
| Embeddings        | sentence-transformers (MiniLM)   |
| LLM Backend       | Groq API (`llama3-70b-8192`)     |
| PDF Processing    | PyPDF2                           |

---

## 🔐 Setup

### 1. Clone this repo

git clone https://github.com/your-username/rag_document_chatbot.git
cd rag_document_chatbot
### 2.Create and Activate a Virtual Environment

python -m venv venv
venv\Scripts\activate       # Windows
### 4. Install Dependencies

pip install -r requirements.txt
### 4. Create .env for groq api key
### 5.Run the app

streamlit run app.py
