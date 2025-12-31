from langchain_chroma import Chroma

def create_chroma(docs, embeddings):
    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="rag_collection",
        persist_directory="data/chroma"
    )
