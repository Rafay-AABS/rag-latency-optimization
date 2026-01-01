import os
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader

def load_pdf(path: str):
    if os.path.isdir(path):
        loader = PyPDFDirectoryLoader(path)
    else:
        loader = PyPDFLoader(path)
    return loader.load()
