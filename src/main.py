import os
from huggingface_hub import InferenceClient
from src.config import settings
from src.data_ingestion import load_pdf_documents
from src.VectorStore import FaissVectorStore


def get_llm_answer(query: str, context_chunks: list) -> str:

    context_elements = []
    for chunk in context_chunks:
        meta = chunk.get("metadata")
        if meta:
            source = meta.get("source", "Unknown source")
            page = meta.get("page", "?")
            text = meta.get("text", "")
            context_elements.append(f"[Source: {source}, Page: {page}]\n{text}")

    context_text = "\n\n---\n\n".join(context_elements)

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        return "Error: Missing HF_TOKEN key in the .env file! Please provide it to generate a response."

    client = InferenceClient(
        model="Qwen/Qwen2.5-7B-Instruct",
        token=hf_token
    )

    system_prompt = (
        "You are a professional, strict data security auditor and an outstanding GDPR (General Data Protection Regulation) expert.\n"
        "Your task is to answer the user's questions strictly based on the provided context from the documents.\n\n"
        "Strict Rules:\n"
        "1. Answer substantively and  precisely.\n"
        "   You may logically interpret the legal text and relate it to real-life situations described by the user.\n"
        "2. For every key piece of information, provide the source file name and page number you are referring to in square brackets and quote them.\n"
        "3. If the provided text DOES NOT contain the answer to the question, respond exactly: "
        "'I did not find relevant information in the analyzed legal documents.' Do not hallucinate or use external knowledge.\n\n"
        f"ALLOWED CONTEXT FROM PDF:\n{context_text}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]

    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        return f" Error communicating with Hugging Face: {e}"


def main():
    store = FaissVectorStore()
    faiss_index_path = os.path.join(str(settings.VECTOR_DB_DIR), "faiss.index")

    if not os.path.exists(faiss_index_path):
        print("[INFO] No local vector database detected. Starting Ingestion process...")
        raw_docs = load_pdf_documents(settings.RAW_DATA_DIR)
        if not raw_docs:
            print(f"[ERROR] Folder '{settings.RAW_DATA_DIR}' is empty! Please drop your PDFs there.")
            return
        store.build_from_documents(raw_docs)
    else:
        print("[INFO] Existing vector database detected on disk. Loading FAISS index...")
        store.load()

    print("\n[SUCCESS] RAG System (Hugging Face) ready! You can now ask questions.")
    print("Type 'q' to close the application.\n")
    print("================================================\n")

    while True:
        query = input("Ask a question about GDPR: ").strip()

        if not query:
            continue
        if query.lower() in ["q"]:
            print("\nThank you for using the application. Shutting down...")
            break
        relevant_chunks = store.query(query, top_k=10)
        if not relevant_chunks or relevant_chunks[0]["metadata"] is None:
            relevant_chunks = store.query(query, top_k=10)

        if not relevant_chunks or relevant_chunks[0]["metadata"] is None:
            print("No relevant chunks found in the documents.\n")
            continue


        answer = get_llm_answer(query, relevant_chunks)

        print("\n RESPONSE:")
        print(answer)



if __name__ == "__main__":
    main()