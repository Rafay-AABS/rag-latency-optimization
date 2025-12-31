from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from app.models.embeddings import get_embeddings
from app.models.llms import get_drafter, get_verifier
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.splitter import split_docs
from app.vectorstore.chroma_store import create_chroma
from app.core.prompts import DRAFT_PROMPT, VERIFY_PROMPT
from app.observability.langfuse import get_langfuse_handler


class SpeculativeRAG:
    def __init__(self):
        self.embeddings = get_embeddings()
        self.drafter = get_drafter()
        self.verifier = get_verifier()
        self.langfuse = get_langfuse_handler()

        self.vectorstore = None
        self.retriever = None
        self.chain = None

    def ingest_pdf(self, pdf_path: str):
        docs = load_pdf(pdf_path)
        splits = split_docs(docs)

        self.vectorstore = create_chroma(splits, self.embeddings)
        self.retriever = self.vectorstore.as_retriever()

        self._build_chain()

    def _build_chain(self):
        draft_chain = (
            RunnablePassthrough.assign(context=lambda x: x["context"])
            | DRAFT_PROMPT
            | self.drafter
            | StrOutputParser()
        )

        self.chain = (
            RunnableParallel(
                context=self.retriever,
                question=RunnablePassthrough()
            )
            .assign(draft_answer=draft_chain)
            .assign(
                final_answer=VERIFY_PROMPT
                | self.verifier
                | StrOutputParser()
            )
        )

    def ask(self, question: str):
        if not self.chain:
            raise RuntimeError("PDF not ingested")

        config = {}
        if self.langfuse:
            config["callbacks"] = [self.langfuse]

        result = self.chain.invoke(question, config=config)

        return {
            "draft": result["draft_answer"],
            "final": result["final_answer"]
        }
