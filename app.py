import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from rag import RAGPipeline

# Load environment variables
load_dotenv()

# Create Flask app, tell it where to find templates
app = Flask(__name__, template_folder="templates")

# Initialise RAG pipeline
INDEX_PATH = os.getenv("INDEX_PATH", "data/index/medical_faiss")
try:
    rag_pipeline = RAGPipeline(index_path=INDEX_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to initialise RAG pipeline: {e}")

import traceback


@app.route("/ask", methods=["POST"])
def ask():
    """
    POST JSON: { "question": "your medical question" }
    Returns: { "answer": "...", "debug": {...} }
    """
    try:
        data = request.get_json()
        if not data or "question" not in data:
            return jsonify({"error": "Missing 'question' in request body"}), 400

        question = data["question"].strip()
        if not question:
            return jsonify({"error": "Question cannot be empty"}), 400

        # Get answer from RAGPipeline
        result = rag_pipeline.answer(question)

        return jsonify(result)

    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500



    
@app.route("/", methods=["GET"])
def home():
    # Serve the HTML UI
    return render_template("index.html")

if __name__ == "__main__":
    # Change host/port as needed
    app.run(host="0.0.0.0", port=5000, debug=True)