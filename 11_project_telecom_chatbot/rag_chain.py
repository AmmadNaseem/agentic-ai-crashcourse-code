"""
Builds the RAG chain:
  merged retriever → prompt → configured chat model on Groq → string output
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_groq import ChatGroq

from retriever import build_retriever
from config import MODEL_NAME, MAX_TOKENS, TIMEOUT, MAX_RETRIES

SYSTEM_PROMPT = """You are a helpful and professional telecom customer care assistant.
Your job is to help customers resolve technical issues with their mobile service.

Use ONLY the context below to answer the customer's question.
The context comes from three sources:
- FAQ entries (general policy and how-to information)
- Telecom guide excerpts (documented troubleshooting procedures)
- Past support tickets (real resolved cases with step-by-step resolutions)

If the context does not contain enough information to answer confidently, say so clearly \
and suggest the customer call 611 or use the MyTelecom app.

Context:
{context}
"""


def _format_docs(docs: list[Document]) -> str:
    sections = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown").upper()
        sections.append(f"[{source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)


def build_chain():
    retriever = build_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    # Qwen reasoning options are omitted for other configurable Groq models.
    model_options = {}
    if MODEL_NAME.startswith("qwen/"):
        model_options = {"reasoning_effort": "none", "reasoning_format": "hidden"}
    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=0,
        max_tokens=MAX_TOKENS,
        timeout=TIMEOUT,
        max_retries=MAX_RETRIES,
        **model_options,
    )

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
