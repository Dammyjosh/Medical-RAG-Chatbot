import os
import argparse
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

def load_pdf_text(pdf_path):
    """Extract text from each page of the PDF."""
    reader = PdfReader(pdf_path)
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            docs.append({"page_content": text, "metadata": {"page": i + 1, "source": pdf_path}})
    return docs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--index", required=True, help="Path to save FAISS index")
    args = parser.parse_args()

    # 1️⃣ Load local embedding model
    # You can change the model to another from https://huggingface.co/sentence-transformers
    model_name = os.getenv("LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    print(f"🔍 Loading local embedding model: {model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # 2️⃣ Load PDF
    print(f"📄 Loading PDF: {args.pdf}")
    docs = load_pdf_text(args.pdf)

    # 3️⃣ Split into chunks
    print("✂️ Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = []
    for d in docs:
        for chunk in splitter.split_text(d["page_content"]):
            chunks.append({"page_content": chunk, "metadata": d["metadata"]})

    texts = [c["page_content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    print(f"📦 Total chunks: {len(texts)}")

    # 4️⃣ Build FAISS index locally
    print("⚡ Generating embeddings and building FAISS index...")
    vs = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)

    # 5️⃣ Save index
    os.makedirs(os.path.dirname(args.index), exist_ok=True)
    vs.save_local(args.index)
    print(f"✅ Index saved to {args.index}")