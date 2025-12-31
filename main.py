import os
import sys
from rag_pipeline import SpeculativeRAG

def main():
    print("--- Speculative RAG Pipeline (Groq + Gemini + Langfuse) ---")
    
    # Check for .env
    if not os.path.exists(".env"):
        print("Error: .env file not found. Please copy .env.example to .env and fill in your keys.")
        return

    rag = SpeculativeRAG()
    
    # Get PDF Path
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = input("Enter path to PDF file: ").strip()
        # Remove quotes if user copied as path
        if pdf_path.startswith('"') and pdf_path.endswith('"'):
            pdf_path = pdf_path[1:-1]
            
    if not os.path.exists(pdf_path):
        print(f"Error: File {pdf_path} does not exist.")
        return

    try:
        rag.ingest_pdf(pdf_path)
    except Exception as e:
        print(f"Error ingesting PDF: {e}")
        return

    print("\nReady! Ask questions about the PDF. Type 'exit' to quit.\n")
    
    while True:
        question = input("User: ")
        if question.lower() in ['exit', 'quit', 'q']:
            break
            
        try:
            response = rag.ask(question)
            print("\n--- Draft (Groq) ---")
            print(response["draft"])
            print("\n--- Final (Gemini Verified) ---")
            print(response["final"])
            print("\n" + "-"*30 + "\n")
        except Exception as e:
            print(f"Error generating answer: {e}")

if __name__ == "__main__":
    main()
