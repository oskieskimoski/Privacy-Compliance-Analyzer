import os
from huggingface_hub import InferenceClient
from src.config import settings
from src.database.vector_store import FaissVectorStore


class PrivacyComplianceRAG:
    def __init__(self):
        self.store = FaissVectorStore()
        self._init_vector_store()

        self.hf_token = os.getenv("HF_TOKEN")
        if not self.hf_token:
            raise ValueError("Missing HF_TOKEN key in the environment!")

        self.client = InferenceClient(
            model="Qwen/Qwen2.5-7B-Instruct",
            token=self.hf_token
        )

    def _init_vector_store(self):
        faiss_index_path = os.path.join(str(settings.VECTOR_DB_DIR), "faiss.index")
        if os.path.exists(faiss_index_path):
            self.store.load()
        else:
            print("[INFO] No local vector database detected. You need to ingest documents first.")

    def _format_context(self, context_chunks: list) -> str:
        context_elements = []
        for chunk in context_chunks:
            meta = chunk.get("metadata")
            if meta:
                source = meta.get("source", "Unknown source")
                page = meta.get("page", "?")
                text = meta.get("text", "")
                context_elements.append(f"[Source: {source}, Page: {page}]\n{text}")
        return "\n\n---\n\n".join(context_elements)

    def ask(self, query: str) -> tuple[str, list[str]]:
        relevant_chunks = self.store.query(query, top_k=10)
        raw_contexts = [c["metadata"].get("text", "") for c in relevant_chunks if c.get("metadata")]
        context_text = self._format_context(relevant_chunks)
        system_prompt = (
            "You are a professional, strict data security auditor and an outstanding GDPR (General Data Protection Regulation) expert.\n"
            "Your task is to answer the user's questions strictly based on the provided context from the documents.\n\n"
            "Strict Rules:\n"
            "1. Answer substantively, precisely, and decisively. Do not use vague language.\n"
            "2. For every key piece of information, provide the source file name and page number you are referring to in square brackets and quote them.\n"
            "3. If the provided text DOES NOT contain the answer to the question, respond exactly: \n"
            "'I did not find relevant information in the analyzed legal documents.' Do not hallucinate or use external knowledge.\n"
            "4. DO NOT use hedging or asekuracja like 'the company should evaluate' or 'it depends' if the text provides clear time limits, rules, or conditions. Give a clear compliance verdict.\n"
            "5. DOUBLE-CHECK article numbers and paragraphs. You are forbidden from inventing or misattributing article subsections (e.g., do not mistake paragraph 2 for paragraph 3). Trust ONLY what is written in the ALLOWED CONTEXT.\n\n"
            f"ALLOWED CONTEXT FROM PDF:\n{context_text}"
            f"ALLOWED CONTEXT FROM PDF:\n{context_text}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        try:
            response = self.client.chat_completion(messages=messages, max_tokens=512, temperature=0.1)
            answer = response.choices[0].message.content
            return answer, raw_contexts
        except Exception as e:
            return f"Error communicating with Hugging Face: {e}", raw_contexts