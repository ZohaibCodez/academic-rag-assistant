import asyncio
import os
from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool
from agents.memory import openai_conversations_session
from dotenv import load_dotenv
from langchain.retrievers import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings  # Fixed: Use only one import
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai.llms import GoogleGenerativeAI
from langchain_pinecone.vectorstores import Pinecone, PineconeVectorStore
from openai import AsyncOpenAI
import streamlit as st
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Handle event loop for async operations
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Validate API keys
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable is not set")

# Initialize OpenAI client for Gemini
external_client = AsyncOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=GOOGLE_API_KEY,
)

model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",
    openai_client=external_client,
)

# Streamlit configuration
st.set_page_config(
    page_title="Academic RAG Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: bold;
}

.subject-badge {
    padding: 0.5rem;
    margin: 0.5rem 0;
    background: linear-gradient(135deg, #e8eaf6 0%, #c5cae9 100%);
    border-left: 4px solid #3f51b5;
    border-radius: 6px;
    border: 1px solid #c5e1a5;
    color: black;
    font-weight: bold;
}

.chat-message {
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 10px;
}

.user-message {
    background-color: #e3f2fd;
    margin-left: 2rem;
}

.assistant-message {
    background-color: #f5f5f5;
    margin-right: 2rem;
}

.thinking-animation {
    color: #1f77b4;
    font-style: italic;
}

.footer {
    position: fixed;
    bottom: 0;
    width: 100%;
    text-align: center;
    background: #f5f5f5;
    padding: 10px;
    font-size: 0.9rem;
    color: #555;
    border-top: 1px solid #ddd;
    z-index: 1000;
}
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_initialized" not in st.session_state:
    st.session_state.agent_initialized = False


@st.cache_resource
def initialize_embeddings():
    """Initialize lightweight embeddings model (cached for performance)"""
    try:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    except Exception as e:
        st.error(f"Failed to initialize embeddings: {str(e)}")
        return None


@st.cache_resource
def initialize_pinecone():
    """Initialize Pinecone connection (cached for performance)"""
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        return pc
    except Exception as e:
        st.error(f"Failed to initialize Pinecone: {str(e)}")
        return None


@st.cache_resource
def setup_vector_store():
    """Setup vector store connections with error handling"""
    embeddings = initialize_embeddings()
    if not embeddings:
        return None

    index_name = "semester-books"
    vector_stores = {}
    subjects = [
        "linear_algebra",
        "discrete_structures",
        "calculas_&_analytical_geometry",
    ]

    try:
        for subject in subjects:
            vector_stores[subject] = PineconeVectorStore(
                index_name=index_name, embedding=embeddings, namespace=subject
            )
        return vector_stores
    except Exception as e:
        st.error(f"Failed to setup vector stores: {str(e)}")
        return None


def format_docs(retrieved_docs):
    """Format retrieved documents for context"""
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


# Initialize vector stores and retrievers
vector_stores = setup_vector_store()
if not vector_stores:
    st.error("Failed to initialize vector stores. Please check your configuration.")
    st.stop()

# Create retrievers
try:
    mmr_retriever_lin = vector_stores["linear_algebra"].as_retriever(
        search_type="mmr", search_kwargs={"k": 2, "lambda_mul": 0.7}
    )

    mmr_retriever_dis = vector_stores["discrete_structures"].as_retriever(
        search_type="mmr", search_kwargs={"k": 2, "lambda_mul": 0.7}
    )

    mmr_retriever_cal = vector_stores["calculas_&_analytical_geometry"].as_retriever(
        search_type="mmr", search_kwargs={"k": 2, "lambda_mul": 0.7}
    )

    multiquery_retriever_lin = MultiQueryRetriever.from_llm(
        retriever=vector_stores["linear_algebra"].as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5, "namespace": "linear_algebra"},
        ),
        llm=GoogleGenerativeAI(model="gemini-1.5-flash"),
    )

    multiquery_retriever_dis = MultiQueryRetriever.from_llm(
        retriever=vector_stores["discrete_structures"].as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3, "namespace": "discrete_structures"},
        ),
        llm=GoogleGenerativeAI(model="gemini-1.5-flash"),
    )

    multiquery_retriever_cal = MultiQueryRetriever.from_llm(
        retriever=vector_stores["calculas_&_analytical_geometry"].as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3, "namespace": "calculas_&_analytical_geometry"},
        ),
        llm=GoogleGenerativeAI(model="gemini-1.5-flash"),
    )

    llm = GoogleGenerativeAI(model="gemini-1.5-flash")

except Exception as e:
    st.error(f"Failed to initialize retrievers: {str(e)}")
    st.stop()


@function_tool
def answer_from_linear_algebra(query: str) -> str:
    """
    REQUIRED for ALL linear algebra questions including:
    - Matrices, vectors, eigenvalues, determinants
    - Linear transformations and vector spaces
    - Systems of linear equations
    - Matrix operations and properties
    - Linear independence, basis, dimension
    - Any question mentioning: matrix, vector, linear, eigen, determinant

    Args:
        query: The student's question about linear algebra

    Returns:
        Detailed answer from linear algebra textbook content
    """
    try:
        print(f"[Debug] answer_from_linear_algebra function call with query: {query}")

        parallel_chain = RunnableParallel(
            {
                "context": multiquery_retriever_lin | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
        )

        prompt = PromptTemplate.from_template(
            """
            You are an expert professor of Linear Algebra. 

            Context from textbook:
            {context}

            Student Question: {question}

            Instructions:
            - Use ONLY the information from the context above
            - Provide clear, step-by-step explanations
            - Include examples and definitions from the context
            - If context lacks info, state clearly what's missing

            Answer:
            """,
        )

        parser = StrOutputParser()
        main_chain = parallel_chain | prompt | llm | parser
        result = main_chain.invoke(query)

        print(f"[Debug] RAG function call with response: {result[:100]}...")
        return result

    except Exception as e:
        error_msg = f"Error in linear algebra query: {str(e)}"
        print(f"[Error] {error_msg}")
        return error_msg


@function_tool
def answer_from_discrete_structures(query: str) -> str:
    """
    REQUIRED for ALL discrete mathematics questions including:
    - Graph theory, trees, networks
    - Sets, relations, functions (discrete context)
    - Logic, proofs, mathematical induction
    - Combinatorics, permutations, combinations
    - Discrete probability and counting
    - Any question mentioning: graph, set, proof, induction, discrete, combinatorics

    Args:
        query: The student's question about discrete structures

    Returns:
        Detailed answer from discrete structures textbook content
    """
    try:
        print(
            f"[Debug] answer_from_discrete_structures function call with query: {query}"
        )

        parallel_chain = RunnableParallel(
            {
                "context": multiquery_retriever_dis | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
        )

        prompt = PromptTemplate.from_template(
            """
            You are an expert professor of Discrete Structures. 

            Context from textbook:
            {context}

            Student Question: {question}

            Instructions:
            - Use ONLY the information from the context above
            - Provide clear, step-by-step explanations
            - Include examples and definitions from the context
            - If context lacks info, state clearly what's missing

            Answer:
            """
        )

        parser = StrOutputParser()
        main_chain = parallel_chain | prompt | llm | parser
        result = main_chain.invoke(query)

        print(f"[Debug] RAG function call with response: {result[:100]}...")
        return result

    except Exception as e:
        error_msg = f"Error in discrete structures query: {str(e)}"
        print(f"[Error] {error_msg}")
        return error_msg


@function_tool
def answer_from_calana(query: str) -> str:
    """
    REQUIRED for ALL calculus and analytical geometry questions including:
    - Derivatives, integrals, limits
    - Continuous functions and continuity
    - Optimization and critical points
    - Series and sequences (calculus context)
    - Analytical geometry and curves
    - Any question mentioning: derivative, integral, limit, continuous, calculus

    Args:
        query: The student's question about calculus/analytical geometry

    Returns:
        Detailed answer from calculus textbook content
    """
    try:
        print(f"[Debug] answer_from_calana function call with query: {query}")

        parallel_chain = RunnableParallel(
            {
                "context": multiquery_retriever_cal | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
        )

        prompt = PromptTemplate.from_template(
            """
            You are an expert professor of Calculas & Analytical Geometry. 

            Context from textbook:
            {context}

            Student Question: {question}

            Instructions:
            - Use ONLY the information from the context above
            - Provide clear, step-by-step explanations
            - Include examples and definitions from the context
            - If context lacks info, state clearly what's missing

            Answer:
            """
        )

        parser = StrOutputParser()
        main_chain = parallel_chain | prompt | llm | parser
        result = main_chain.invoke(query)

        print(f"[Debug] RAG function call with response: {result[:100]}...")
        return result

    except Exception as e:
        error_msg = f"Error in calculus query: {str(e)}"
        print(f"[Error] {error_msg}")
        return error_msg


# Initialize agent
agent = Agent(
    name="Academic RAG Assistant",
    instructions="""You are an academic assistant. 
    For every user query:
    - Always decide the subject (Linear Algebra, Discrete Structures, or Calculus).
    - Always call the matching tool.
    - Never answer directly without a tool.
    - If a prompt you think given by a user dont well aligned and not better for retrieval of contents from vector db, Redefine user's query on your own way and then pass to your tools
    """,
    tools=[
        answer_from_linear_algebra,
        answer_from_discrete_structures,
        answer_from_calana,
    ],
    model=model,
)


async def get_agent_response(agent, query: str) -> str:
    """Get response from agent with error handling"""
    try:
        result = await Runner.run(agent, query)
        return result.final_output
    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        print(f"[Error] {error_msg}")
        return error_msg


def main():
    # Header
    st.markdown(
        '<h1 class="main-header">🎓 Academic RAG Assistant</h1>', unsafe_allow_html=True
    )

    # Sidebar
    with st.sidebar:
        st.header("📚 Available Subjects")

        subjects = [
            "Linear Algebra",
            "Discrete Structures",
            "Calculus & Analytical Geometry",
        ]
        for subject in subjects:
            st.markdown(
                f'<div class="subject-badge">📖 {subject}</div>', unsafe_allow_html=True
            )

        st.markdown("---")
        st.subheader("ℹ️ How to use")
        st.write("1. Ask questions about your course material")
        st.write("2. The assistant will search relevant textbooks")
        st.write("3. Get detailed explanations with examples")

        st.markdown("---")
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # Main chat interface
    st.subheader("💬 Chat with your Academic Assistant")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Initialize agent (this happens once)
    if not st.session_state.agent_initialized:
        with st.spinner("Initializing Academic Assistant..."):
            if agent:
                st.session_state.agent = agent
                st.session_state.agent_initialized = True
                st.success("✅ Academic Assistant ready!")
            else:
                st.error("❌ Failed to initialize assistant")
                return

    # Chat input
    if prompt := st.chat_input("Ask me anything about your courses..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking and searching through textbooks..."):
                response = asyncio.run(
                    get_agent_response(st.session_state.agent, prompt)
                )

            st.markdown(response)

        # Add assistant message
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
            }
        )

        st.rerun()

    # Footer
    st.markdown(
        '<div class="footer">🎓 Academic RAG Assistant | Built with ❤️ using Streamlit</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
