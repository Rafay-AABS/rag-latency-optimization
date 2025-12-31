from langchain_core.prompts import ChatPromptTemplate

DRAFT_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question based ONLY on the following context.
If you don't know the answer, say that you don't know.

Context:
{context}

Question: {question}

Draft Answer:
""")

VERIFY_PROMPT = ChatPromptTemplate.from_template("""
You are a senior editor and fact-checker.

Context:
{context}

Question: {question}

Draft Answer:
{draft_answer}

Final Refined Answer:
""")
