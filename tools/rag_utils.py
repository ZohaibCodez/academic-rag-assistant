# src/tools/rag_utils.py
import traceback
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel

# Centralized base prompt (one template for all subjects)
BASE_PROMPT = """You are an expert professor of {subject}.

Context from textbook:
{context}

Student Question: {question}

Instructions:
- Use ONLY the information from the context above.
- Provide clear step-by-step explanations for complex material.
- Include relevant examples and definitions from the context when present.
- If context lacks information required to fully answer, state what's missing and suggest follow-up queries.
- Keep the explanation concise, structured, and student-friendly.

Answer:
"""

def format_docs(retrieved_docs):
    """
    Convert retrieved docs (list-like of Document objects) into a single context string.
    This function is intentionally small and deterministic so it's easy to test.
    """
    try:
        # Each doc expected to have 'page_content' field (LangChain-style)
        return "\n\n".join(getattr(doc, "page_content", str(doc)) for doc in retrieved_docs)
    except Exception:
        # Fallback: try join directly, but return a helpful message if formatting fails
        return " ".join(str(d) for d in retrieved_docs)

def run_rag(retriever, llm, subject: str, query: str):
    """
    Run a reusable RAG pipeline:
      - retriever: a LangChain retriever or runnable
      - llm: an LLM runnable (GoogleGenerativeAI or similar)
      - subject: string used to specialize the prompt
      - query: user question
    
    Returns a string result or raises an Exception on error.
    """
    try:
        # Build a small runnable pipeline:
        # - produce context by retrieving and formatting docs
        parallel_chain = RunnableParallel(
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
        )

        prompt_template = PromptTemplate.from_template(BASE_PROMPT)
        parser = StrOutputParser()

        # Compose pipeline: parallel (context+question) -> prompt -> llm -> parser
        main_chain = parallel_chain | prompt_template.partial(subject=subject) | llm | parser

        # Invoke synchronously (consistent with your original .invoke usage)
        result = main_chain.invoke(query)

        return result
    except Exception as e:
        # Add a stacktrace to help debugging in production logs (but don't leak secrets)
        tb = traceback.format_exc()
        raise RuntimeError(f"RAG pipeline failed for subject={subject}: {e}\n{tb}")
