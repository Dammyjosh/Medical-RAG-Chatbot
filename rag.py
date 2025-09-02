import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

class RAGPipeline:
    def __init__(self, index_path: str):
        local_embed_model = os.getenv("LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        local_llm_model = os.getenv("LOCAL_LLM_MODEL", "google/flan-t5-small")

        print(f"🔍 Using embedding model: {local_embed_model}")
        self.embeddings = HuggingFaceEmbeddings(model_name=local_embed_model)

        print(f"📂 Loading FAISS index from: {index_path}")
        self.vs = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
        self.retriever = self.vs.as_retriever(search_kwargs={"k": 4})

        print(f"🤖 Using local LLM model: {local_llm_model}")
        self.llm = pipeline("text2text-generation", model=local_llm_model)

    def _gather_context(self, docs):
        """Concatenate all retrieved documents into a single string."""
        return "\n\n".join(d.page_content for d in docs)

    def _build_prompt(self, question, context):
        return (
            "You are a careful and professional medical assistant. "
            "Use only the context provided to answer the question. "
            "If unsure, say you don't know and recommend consulting a medical professional.\n\n"
            "Instructions:\n"
            "- Provide a clear, structured explanation.\n"
            "- Break down complex information step-by-step.\n"
            "- Use simple, easy-to-understand language.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )

    def answer(self, question):
        docs = self.retriever.invoke(question)
        context = self._gather_context(docs)
        prompt = self._build_prompt(question, context)

        try:
            raw_output = self.llm(prompt, max_new_tokens=512)
        except Exception as e:
            return {
                "answer": f"❌ Model invocation failed: {e}",
                "debug": {"prompt": prompt, "error": str(e)}
            }

        # Parse output
        answer = ""
        if isinstance(raw_output, str):
            answer = raw_output.strip()
        elif isinstance(raw_output, list) and len(raw_output) > 0:
            if isinstance(raw_output[0], dict) and "generated_text" in raw_output[0]:
                answer = raw_output[0]["generated_text"].strip()
            else:
                answer = str(raw_output).strip()

        return {
            "answer": answer,
            "debug": {
                "retrieved_docs_count": len(docs),
                "prompt_sent_to_llm": prompt
            }
        }

