## 🩺 Medical RAG Chatbot

An intelligent Retrieval-Augmented Generation (RAG) chatbot designed to answer medical-related queries using local documents.
It combines:

FAISS for vector search

Hugging Face Transformers for embeddings + language model

Flask API to serve the chatbot

⚠️ Disclaimer: This chatbot is not a substitute for professional medical advice. Always consult a healthcare provider for real medical concerns.


---

## 🚀 Features

Retrieval-Augmented Generation (RAG) with FAISS

Local or Hugging Face transformer models (no OpenAI API required)

Source citation support ([pX] references from documents)

REST API (/ask) for integration with frontend or other services

Dockerized for easy deployment

---



## 🛠️ Setup
1. Clone the repository
git clone https://github.com/Dammyjosh/Med-RAG-Chatbot.git
cd Med-RAG-Chatbot

2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

3. Install dependencies
pip install -r requirements.txt

4. Prepare FAISS index & documents

Place your processed documents (PDFs, text, etc.) in data/docs/.

Build the FAISS index (you should already have data/index/medical_faiss).

▶️ Run Locally
Start Flask API
python app.py


The API will be available at:

http://127.0.0.1:5000 or localhost:5000
