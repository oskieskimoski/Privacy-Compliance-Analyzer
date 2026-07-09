from src.pipeline.privacy_compliance_rag import PrivacyComplianceRAG

def main():
    try:
        rag = PrivacyComplianceRAG()
    except Exception as e:
        print(f"[ERROR]: {e}")
        return

    print("Write 'q' to exit.\n")
    print("================================================\n")

    while True:
        query = input("Ask a question about GDPR: ").strip()

        if not query:
            continue
        if query.lower() in ["q"]:
            print("\nClosing...")
            break

        answer, _ = rag.ask(query)

        print("\n Answer:")
        print(answer)
if __name__ == "__main__":
    main()