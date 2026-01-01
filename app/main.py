import os
import sys
import logging

# Add project root to sys.path to allow running script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.pipeline import SpeculativeRAG
from app.core.config import get_settings
from app.core.logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)
settings = get_settings()

def main():
    print(f"\n--- {settings.APP_NAME} (CLI) ---\n")

    rag = SpeculativeRAG()

    # Get PDF path (CLI arg or prompt)
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = input("Enter path to PDF file or directory: ").strip()
        if pdf_path.startswith('"') and pdf_path.endswith('"'):
            pdf_path = pdf_path[1:-1]

    if not os.path.exists(pdf_path):
        logger.error(f"File not found: {pdf_path}")
        return

    # Ingest PDF
    try:
        print("\nIngesting document...")
        rag.ingest_pdf(pdf_path)
    except Exception as e:
        logger.error(f"Failed to ingest PDF: {e}")
        return

    print("\n✅ Ready! Ask questions about the document.")
    print("Type 'exit' or 'quit' to stop.\n")

    # Interactive QA loop
    while True:
        question = input("User: ").strip()

        if question.lower() in {"exit", "quit", "q"}:
            print("\nGoodbye 👋")
            break
            
        try:
            response = rag.ask(question)
            if "error" in response:
                print(f"Error: {response['error']}")
                continue
                
            print("\n--- Draft ---")
            print(response["draft"])
            print("\n--- Final ---")
            print(response["final"])
            print("\n" + "-"*30 + "\n")
        except Exception as e:
            logger.error(f"Error generating answer: {e}")

if __name__ == "__main__":
    main()
