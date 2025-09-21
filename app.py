import asyncio
from datetime import datetime
import os
import logging
import sys
from typing import Optional, Dict, Any
from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    RunResultStreaming,
    Runner,
    SQLiteSession,
    function_tool,
)
from agents.memory import openai_conversations_session, session
from dotenv import load_dotenv
from langchain.retrievers import MultiQueryRetriever
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai.llms import GoogleGenerativeAI
from langchain_pinecone.vectorstores import Pinecone, PineconeVectorStore
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent
import streamlit as st
from langchain_core.output_parsers import StrOutputParser

# No additional imports needed for tracing control


# Configure logging
def setup_logging():
    """Setup logging configuration"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create logs directory if it doesn't exist
    if not os.path.exists("logs"):
        os.makedirs("logs")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(f'logs/app_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Set specific loggers
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("pinecone").setLevel(logging.WARNING)

    return logging.getLogger(__name__)


# Initialize logger
logger = setup_logging()

# User-friendly error messages
ERROR_MESSAGES = {
    "api_key_invalid": "❌ There seems to be an issue with your API key. Please check if it's correct and try again.",
    "connection_error": "🌐 Unable to connect to the service. Please check your internet connection and try again.",
    "retrieval_error": "📚 I'm having trouble accessing the course materials right now. Please try again in a moment.",
    "processing_error": "🤖 I encountered an issue while processing your request. Please try rephrasing your question.",
    "initialization_error": "⚙️ The assistant is having trouble starting up. Please refresh the page and try again.",
    "general_error": "😅 Something unexpected happened. Please try again, and if the problem persists, contact support.",
}

load_dotenv()

# Handle event loop for async operations
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
try:
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
except:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Handle OpenAI API key for tracing (optional)
# If OpenAI API key is not provided in .env, disable tracing to prevent errors
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except KeyError:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    # Disable OpenAI tracing when no API key is provided
    os.environ["OPENAI_TRACING_ENABLED"] = "false"
    os.environ["OPENAI_API_KEY"] = "dummy-key"  # Required to prevent 401 errors
    logger.info("OpenAI API key not found - tracing disabled")


# Validate API keys with better error handling
def validate_api_keys():
    """Validate required API keys"""
    try:
        if not GOOGLE_API_KEY:
            logger.error("GOOGLE_API_KEY environment variable is not set")
            return False, "Google API key is missing"
        if not PINECONE_API_KEY:
            logger.error("PINECONE_API_KEY environment variable is not set")
            return False, "Pinecone API key is missing"
        return True, "API keys validated"
    except Exception as e:
        logger.error(f"Error validating API keys: {str(e)}")
        return False, "Error validating API keys"


# Streamlit configuration
st.set_page_config(
    page_title="Academic RAG Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS (keeping your existing styles)
st.markdown(
    """
<style>
.main-header {
    font-size: 2.5rem;
    color: #f6ad55;  /* Warm Gold */
    text-align: center;
    margin-bottom: 2rem;
    font-weight: bold;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.stHeading{
text-align:center
}

.subject-badge {
    padding: 0.75rem;
    margin: 0.5rem 0;
    background: linear-gradient(135deg, #6b46c1 0%, #553c9a 100%);  /* Deep Purple gradient */
    border-left: 4px solid #f6ad55;  /* Warm Gold border */
    border-radius: 8px;
    border: 1px solid #4c1d95;
    color: white;  /* White text for contrast */
    font-weight: bold;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    transition: all 0.3s ease;
}

.subject-badge:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(246, 173, 85, 0.3);
    cursor:pointer;
}

.chat-message {
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 12px;
}

.user-message {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);  /* Dark gradient */
    margin-left: 2rem;
    border-left: 3px solid #f6ad55;  /* Gold accent */
    color: white;
}

.assistant-message {
    background: linear-gradient(135deg, #374151 0%, #4b5563 100%);  /* Charcoal gradient */
    margin-right: 2rem;
    border-left: 3px solid #6b46c1;  /* Purple accent */
    color: white;
}

.thinking-animation {
    color: #f6ad55;  /* Warm Gold */
    font-style: italic;
}

#chat-with-your-academic-assistant {
    text-align: center;
    color: #f6ad55;  /* Warm Gold */
}

/* Input field styling */
.stChatInput > div > div > textarea {
    background-color: #374151 !important;
    border: 2px solid #4b5563 !important;
    border-radius: 12px !important;
    color: white !important;
}

.stChatInput > div > div > textarea:focus {
    border-color: #f6ad55 !important;  /* Gold focus border */
    box-shadow: 0 0 0 2px rgba(246, 173, 85, 0.2) !important;
}

/* Send button styling */
.stChatInput button {
    background-color: #f6ad55 !important;  /* Gold button */
    border: none !important;
    border-radius: 50% !important;
    color: #1a202c !important;
}

.stChatInput button:hover {
    background-color: #ed8936 !important;  /* Darker gold on hover */
    transform: scale(1.02);
}

/* Sidebar styling */
.css-1d391kg {
    background-color: #1a202c !important;  /* Dark navy sidebar */
}

/* Spinner color */
.stSpinner > div {
    border-top-color: #f6ad55 !important;  /* Gold spinner */
}

/* Success message */
.stSuccess {
    background-color: rgba(246, 173, 85, 0.1) !important;
    border-left: 4px solid #f6ad55 !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    justify-content: center !important;
    align-items: center !important;
    width: 100% !important;
    background-color: transparent !important;
}

/* Individual tab button styling */
.stTabs [data-baseweb="tab"] {
    flex: 1 !important;
    text-align: center !important;
    justify-content: center !important;
    padding: 0.75rem 0.25rem !important;
    min-width: 0 !important;
    width: 33.33% !important;
    background-color: #2d3748 !important;
    border: none !important;
    border-bottom: 3px solid #4a5568 !important;
    color: #a0aec0 !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}

/* Tab button text */
.stTabs [data-baseweb="tab"] > div {
    text-align: center !important;
    justify-content: center !important;
    width: 100% !important;
    color: inherit !important;
}

/* Active tab styling */
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background-color: #374151 !important;
    border-bottom-color: #ff6b35 !important;
    color: #e2e8f0 !important;
    font-weight: 600 !important;
}

/* Hover state for inactive tabs */
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
    background-color: #374151 !important;
    border-bottom-color: #ff8a65 !important;
    color: #cbd5e0 !important;
}

svg{
width:23px !important;
height:23px !important;
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
if "session_name" not in st.session_state:
    st.session_state.session_name = SQLiteSession("assistant_memory")
if "vector_stores" not in st.session_state:
    st.session_state.vector_stores = None
if "retrievers" not in st.session_state:
    st.session_state.retrievers = {}


@st.cache_resource
def initialize_embeddings():
    """Initialize lightweight embeddings model (cached for performance)"""
    try:
        logger.info("Initializing HuggingFace embeddings model")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        logger.info("Successfully initialized embeddings model")
        return embeddings
    except ImportError as e:
        logger.error(f"Failed to import required libraries for embeddings: {str(e)}")
        st.error(ERROR_MESSAGES["initialization_error"])
        return None
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {str(e)}")
        st.error(ERROR_MESSAGES["initialization_error"])
        return None


@st.cache_resource
def initialize_pinecone():
    """Initialize Pinecone connection (cached for performance)"""
    try:
        logger.info("Initializing Pinecone connection")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        logger.info("Successfully initialized Pinecone connection")
        return pc
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {str(e)}")
        st.error(ERROR_MESSAGES["connection_error"])
        return None


@st.cache_resource
def setup_vector_store():
    """Setup vector store connections with error handling"""
    try:
        logger.info("Setting up vector stores")
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

        for subject in subjects:
            logger.info(f"Setting up vector store for subject: {subject}")
            vector_stores[subject] = PineconeVectorStore(
                index_name=index_name, embedding=embeddings, namespace=subject
            )

        logger.info("Successfully set up all vector stores")
        return vector_stores
    except Exception as e:
        logger.error(f"Failed to setup vector stores: {str(e)}")
        st.error(ERROR_MESSAGES["initialization_error"])
        return None


def format_docs(retrieved_docs):
    """Format retrieved documents for context"""
    try:
        return "\n\n".join(doc.page_content for doc in retrieved_docs)
    except Exception as e:
        logger.error(f"Error formatting documents: {str(e)}")
        return "Error retrieving document content"


def initialize_retrievers_with_model(model_name: str) -> bool:
    """Initialize retrievers with the selected model"""
    try:
        logger.info(f"Initializing retrievers with model: {model_name}")

        if st.session_state.vector_stores is None:
            st.session_state.vector_stores = setup_vector_store()

        if st.session_state.vector_stores is None:
            logger.error("Vector stores not available")
            return False

        # Initialize LLM with selected model
        llm = GoogleGenerativeAI(model=model_name)
        logger.info(f"Successfully initialized LLM with model: {model_name}")

        # Create retrievers with the selected model
        st.session_state.retrievers = {
            "mmr_retriever_lin": st.session_state.vector_stores[
                "linear_algebra"
            ].as_retriever(
                search_type="mmr", search_kwargs={"k": 2, "lambda_mul": 0.7}
            ),
            "mmr_retriever_dis": st.session_state.vector_stores[
                "discrete_structures"
            ].as_retriever(
                search_type="mmr", search_kwargs={"k": 2, "lambda_mul": 0.7}
            ),
            "mmr_retriever_cal": st.session_state.vector_stores[
                "calculas_&_analytical_geometry"
            ].as_retriever(
                search_type="mmr", search_kwargs={"k": 2, "lambda_mul": 0.7}
            ),
            "multiquery_retriever_lin": MultiQueryRetriever.from_llm(
                retriever=st.session_state.vector_stores["linear_algebra"].as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 2, "namespace": "linear_algebra"},
                ),
                llm=llm,
            ),
            "multiquery_retriever_dis": MultiQueryRetriever.from_llm(
                retriever=st.session_state.vector_stores[
                    "discrete_structures"
                ].as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 2, "namespace": "discrete_structures"},
                ),
                llm=llm,
            ),
            "multiquery_retriever_cal": MultiQueryRetriever.from_llm(
                retriever=st.session_state.vector_stores[
                    "calculas_&_analytical_geometry"
                ].as_retriever(
                    search_type="similarity",
                    search_kwargs={
                        "k": 2,
                        "namespace": "calculas_&_analytical_geometry",
                    },
                ),
                llm=llm,
            ),
            "llm": llm,
        }
        logger.info("Successfully initialized all retrievers")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize retrievers: {str(e)}")
        st.error(ERROR_MESSAGES["initialization_error"])
        return False


def create_function_tools(model_name: str):
    """Create function tools with the selected model"""
    logger.info(f"Creating function tools for model: {model_name}")

    @function_tool
    def answer_from_linear_algebra(query: str) -> str:
        """RAG tool for linear algebra"""
        try:
            logger.info(f"Linear algebra query received: {query[:100]}...")

            parallel_chain = RunnableParallel(
                {
                    "context": st.session_state.retrievers["multiquery_retriever_lin"]
                    | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                }
            )

            prompt = PromptTemplate.from_template(
                """
                You are a helpful academic tutor helping a student with their coursework.
                You have access to relevant sections from their course textbook.

                Course Material Context:
                {context}

                Student's Question: {question}

                Instructions:
                - Answer the question using the provided course material context
                - Explain concepts step-by-step in simple terms
                - Include examples or analogies when helpful
                - If the question asks for "steps" or "method", provide a clear numbered list
                - If asking about general concepts (like "how to solve linear systems"), provide the standard method from the textbook
                - For sample problems, create appropriate examples if none are in the context
                - If the context is insufficient, provide what you can and mention what additional information might be helpful

                Provide a clear, educational response:
                """,
            )

            parser = StrOutputParser()
            main_chain = (
                parallel_chain | prompt | st.session_state.retrievers["llm"] | parser
            )
            result = main_chain.invoke(query)

            logger.info(f"Linear algebra response generated successfully")
            return result

        except Exception as e:
            logger.error(f"Error in linear algebra query: {str(e)}")
            return "I'm having trouble accessing the Linear Algebra materials right now. Please try rephrasing your question or try again in a moment."

    @function_tool
    def answer_from_discrete_structures(query: str) -> str:
        """RAG tool for discrete structures"""
        try:
            logger.info(f"Discrete structures query received: {query[:100]}...")

            parallel_chain = RunnableParallel(
                {
                    "context": st.session_state.retrievers["multiquery_retriever_dis"]
                    | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                }
            )

            prompt = PromptTemplate.from_template(
                """
                You are a helpful academic tutor helping a student with their coursework.
                You have access to relevant sections from their course textbook.

                Course Material Context:
                {context}

                Student's Question: {question}

                Instructions:
                - Answer the question using the provided course material context
                - Explain concepts step-by-step in simple terms
                - Include examples or analogies when helpful
                - If the question asks for "steps" or "method", provide a clear numbered list
                - If asking about general concepts (like "how to solve linear systems"), provide the standard method from the textbook
                - For sample problems, create appropriate examples if none are in the context
                - If the context is insufficient, provide what you can and mention what additional information might be helpful

                Provide a clear, educational response:
                """
            )

            parser = StrOutputParser()
            main_chain = (
                parallel_chain | prompt | st.session_state.retrievers["llm"] | parser
            )
            result = main_chain.invoke(query)

            logger.info(f"Discrete structures response generated successfully")
            return result

        except Exception as e:
            logger.error(f"Error in discrete structures query: {str(e)}")
            return "I'm having trouble accessing the Discrete Structures materials right now. Please try rephrasing your question or try again in a moment."

    @function_tool
    def answer_from_calana(query: str) -> str:
        """RAG tool for calculus and analytical geometry"""
        try:
            logger.info(f"Calculus query received: {query[:100]}...")

            parallel_chain = RunnableParallel(
                {
                    "context": st.session_state.retrievers["multiquery_retriever_cal"]
                    | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                }
            )

            prompt = PromptTemplate.from_template(
                """
                You are a helpful academic tutor helping a student with their coursework.
                You have access to relevant sections from their course textbook.

                Course Material Context:
                {context}

                Student's Question: {question}

                Instructions:
                - Answer the question using the provided course material context
                - Explain concepts step-by-step in simple terms
                - Include examples or analogies when helpful
                - If the question asks for "steps" or "method", provide a clear numbered list
                - If asking about general concepts (like "how to solve linear systems"), provide the standard method from the textbook
                - For sample problems, create appropriate examples if none are in the context
                - If the context is insufficient, provide what you can and mention what additional information might be helpful

                Provide a clear, educational response:
                """
            )

            parser = StrOutputParser()
            main_chain = (
                parallel_chain | prompt | st.session_state.retrievers["llm"] | parser
            )
            result = main_chain.invoke(query)

            logger.info(f"Calculus response generated successfully")
            return result

        except Exception as e:
            logger.error(f"Error in calculus query: {str(e)}")
            return "I'm having trouble accessing the Calculus & Analytical Geometry materials right now. Please try rephrasing your question or try again in a moment."

    logger.info("Successfully created all function tools")
    return [
        answer_from_linear_algebra,
        answer_from_discrete_structures,
        answer_from_calana,
    ]


def agent_initialization(model_name: str, _api_key: str) -> Optional[Agent]:
    """Initialize agent with the selected model"""
    try:
        logger.info(f"Initializing agent with model: {model_name}")

        # Validate API key format
        if not _api_key or len(_api_key) < 20:
            logger.error("Invalid API key provided")
            st.error(ERROR_MESSAGES["api_key_invalid"])
            return None

        # Initialize OpenAI client for Gemini
        external_client = AsyncOpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=_api_key,
        )

        model = OpenAIChatCompletionsModel(
            model=model_name,
            openai_client=external_client,
        )

        # Get function tools for the selected model
        tools = create_function_tools(model_name)

        # Initialize agent
        agent = Agent(
            name="Academic RAG Assistant",
            instructions="""
            You are an Expert Academic Assistant built on top of an Agentic RAG system. 
            Your role is to help students by answering their questions using their course books 
            through the retrieval tools provided.

            TOOLS AVAILABLE:
            - answer_from_linear_algebra(query)
            - answer_from_discrete_structures(query)
            - answer_from_calana(query) [for calculus & analytical geometry]

            WORKFLOW:
            1. When you receive a student query, first REFORMULATE and ENHANCE it into a more detailed, 
            academic-style question that will get better results from the RAG system.

            2. Identify which subject it likely belongs to:
            * Linear Algebra (matrices, vectors, systems of equations, eigenvalues, determinants, etc.)
            * Discrete Structures (logic, sets, graphs, combinatorics, proofs, etc.)
            * Calculus & Analytical Geometry (derivatives, integrals, limits, geometry, etc.)
            
            3. IMMEDIATELY use the appropriate tool with your ENHANCED query - don't ask for clarification first.
            
            4. If unsure about the subject, try the most likely tool first, then others if needed.
            
            5. Present the tool's response in a clear, student-friendly manner.

            QUERY ENHANCEMENT EXAMPLES:
            - "Tell me steps to solve a linear system" → "What are the detailed steps to solve a system of linear equations? Include methods like Gaussian elimination or substitution method with examples."
            - "derivatives" → "How do you find derivatives? What are the rules and methods for differentiation with step-by-step examples?"
            - "what is matrix" → "What is a matrix in linear algebra? Explain the definition, types, basic operations, and provide examples."
            - "prove by induction" → "How do you write a proof by mathematical induction? What are the steps and structure with examples?"

            IMPORTANT RULES:
            - Always ENHANCE short/vague queries into detailed, specific questions before using tools
            - Make queries more academic and comprehensive to get better RAG results
            - Include requests for examples, steps, definitions as appropriate
            - Don't ask for book titles or authors - the tools access the student's course books
            - Be proactive in helping with enhanced queries, not reactive in asking questions
            """,
            tools=tools,
            model=model,
        )

        logger.info("Successfully initialized agent")
        return agent
    except Exception as e:
        logger.error(f"Failed to initialize agent: {str(e)}")
        st.error(ERROR_MESSAGES["initialization_error"])
        return None


def handle_sidebar():
    """Handle sidebar configuration and return selected model and API key"""
    with st.sidebar:
        st.header("🔑 Study Assistant")

        # Create tabs
        tab1, tab2, tab3 = st.tabs(["⚙️ Config", "📚 Subjects", "📊 Info"])

        # Configuration Tab
        with tab1:
            api_key = st.text_input(
                "Your Google Gemini API Key",
                type="password",
                placeholder="Enter your API key...",
                help="Your key is kept only in your current browser session.",
                value=st.session_state.get("api_key", ""),
            )
            if api_key:
                st.session_state.api_key = api_key
                if len(api_key) < 20:
                    st.error("⚠️ This API key looks too short. Please check it.")
                elif not api_key.startswith("AIza"):
                    st.warning(
                        "⚠️ This doesn't look like a Google API key. Double-check it."
                    )
                else:
                    os.environ["GOOGLE_API_KEY"] = api_key
                    st.success("✅ API key set for this session")
                    logger.info("API key successfully set")
            else:
                st.info("💡 Enter your API key to start chatting")

            st.divider()

            selected_model = st.selectbox(
                "Generation Models",
                [
                    "gemini-2.5-pro",
                    "gemini-2.5-flash",
                    "gemini-2.5-flash-lite",
                    "gemini-2.5-flash-image-preview",
                    "gemini-live-2.5-flash-preview",
                    "gemini-2.0-flash",
                    "gemini-2.0-flash-lite",
                    "gemini-2.0-flash-001",
                    "gemini-2.0-flash-lite-001",
                    "gemini-2.0-flash-live-001",
                    "gemini-2.0-flash-live-preview-04-09",
                    "gemini-2.0-flash-preview-image-generation",
                    "gemini-1.5-flash",
                    "gemini-1.5-pro",
                ],
                index=0,
                help="Choose the Gemini model for generation",
            )
            st.session_state.model = selected_model

            # Reset agent if model changed
            if "previous_model" not in st.session_state:
                st.session_state.previous_model = selected_model
            elif st.session_state.previous_model != selected_model:
                st.session_state.agent_initialized = False
                st.session_state.previous_model = selected_model
                logger.info(f"Model changed to: {selected_model}")

        # Subjects Tab
        with tab2:
            subjects = [
                "Linear Algebra",
                "Discrete Structures",
                "Calculus & Analytical Geometry",
            ]

            st.write("**Available Subjects:**")
            for subject in subjects:
                st.markdown(
                    f'<div class="subject-badge">📖 {subject}</div>',
                    unsafe_allow_html=True,
                )

            st.divider()
            st.subheader("ℹ️ How to use")
            st.write("1. Ask questions about your course material")
            st.write("2. The assistant will search relevant textbooks")
            st.write("3. Get detailed explanations with examples")

        # Session Info Tab
        with tab3:
            message_count = (
                len(st.session_state.messages) - 1
                if st.session_state.get("messages")
                else 0
            )

            st.metric("Messages", message_count)
            st.info(f"**Current Model:**\n{selected_model}")

            if message_count > 0:
                st.divider()
                chat_text = ""
                for msg in st.session_state.messages:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    chat_text += f"{role}: {msg['content']}\n\n"

                st.download_button(
                    "📥 Download Chat",
                    chat_text,
                    f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    "text/plain",
                    use_container_width=True,
                    help="Download your conversation history",
                )

            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                logger.info("Chat history cleared")
                st.rerun()

    return selected_model, st.session_state.get("api_key")


async def main():
    """Main application function"""
    streaming_speed = 0.05
    try:
        logger.info("Starting Academic RAG Assistant application")

        # Header
        st.markdown(
            '<h1 class="main-header">🎓 Academic RAG Assistant</h1>',
            unsafe_allow_html=True,
        )

        selected_model, user_api_key = handle_sidebar()
        chat_model = None
        if user_api_key and selected_model:
            # Ensure env var is set for the underlying client
            os.environ["GOOGLE_API_KEY"] = user_api_key
            chat_model = selected_model

        # Main chat interface
        st.subheader("💬 Chat with your Academic Assistant")

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if chat_model is None:
            st.warning(
                "Please enter your Google Gemini API key in the sidebar to start chatting."
            )
            return

        # Initialize retrievers and agent
        if not st.session_state.agent_initialized:
            with st.spinner("Initializing Academic Assistant..."):
                try:
                    logger.info("Initializing retrievers and agent")
                    # Initialize retrievers with selected model
                    if initialize_retrievers_with_model(chat_model):
                        # Initialize agent with selected model
                        agent = agent_initialization(chat_model, user_api_key)
                        if agent:
                            st.session_state.agent = agent
                            st.session_state.agent_initialized = True
                            st.success("✅ Academic Assistant ready!")
                            logger.info("Academic Assistant successfully initialized")
                        else:
                            logger.error("Failed to initialize agent")
                            st.error(
                                "❌ Failed to initialize assistant. Please check your API key and try again."
                            )
                            return
                    else:
                        logger.error("Failed to initialize retrievers")
                        st.error(
                            "❌ Failed to initialize retrievers. Please refresh the page and try again."
                        )
                        return
                except Exception as e:
                    logger.error(f"Error during initialization: {str(e)}")
                    st.error(ERROR_MESSAGES["initialization_error"])
                    return

        # Chat input
        if prompt := st.chat_input(
            "Ask me anything about your courses...", disabled=chat_model is None
        ):
            try:
                # Validate input
                if not prompt.strip():
                    st.warning("Please enter a valid question.")
                    return

                if len(prompt.strip()) < 3:
                    st.warning("Please enter a more detailed question.")
                    return

                logger.info(f"User query received: {prompt[:100]}...")

                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})

                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Generate assistant response
                with st.chat_message("assistant"):
                    with st.spinner("🤔 Thinking and searching through textbooks..."):
                        message_placeholder = st.empty()
                        full_response = ""

                        try:
                            logger.info(f"Processing query: {prompt[:100]}...")

                            # Get the agent from session state
                            agent = st.session_state.agent

                            # Run the agent with streaming
                            result: RunResultStreaming = Runner.run_streamed(
                                agent, prompt, session=st.session_state.session_name
                            )

                            # Stream the response
                            async for event in result.stream_events():
                                if event.type == "raw_response_event" and isinstance(
                                    event.data, ResponseTextDeltaEvent
                                ):
                                    if (
                                        hasattr(event.data, "delta")
                                        and event.data.delta
                                    ):
                                        full_response += event.data.delta
                                        message_placeholder.markdown(
                                            full_response + "▌"
                                        )
                                        await asyncio.sleep(streaming_speed)
                                    # elif hasattr(event.data, 'content') and event.data.content:
                                    #     # Handle different event data structures
                                    #     content = event.data.content
                                    #     if isinstance(content, str):
                                    #         full_response += content
                                    #     elif hasattr(content, 'text'):
                                    #         full_response += content.text
                                    #     message_placeholder.markdown(full_response + "▌")

                            # Final response without cursor
                            if full_response and full_response.strip():
                                message_placeholder.markdown(full_response)
                                logger.info("Successfully generated streaming response")
                            else:
                                error_msg = "🚫 No response received. Please try again."
                                message_placeholder.error(error_msg)
                                full_response = error_msg
                                logger.warning("Empty response received from agent")

                        except asyncio.TimeoutError:
                            logger.error("Request timed out")
                            full_response = "⏱️ The request is taking longer than expected. Please try again with a simpler question."
                            message_placeholder.error(full_response)
                        except Exception as e:
                            logger.error(f"Error during streaming: {str(e)}")
                            full_response = ERROR_MESSAGES["processing_error"]
                            message_placeholder.error(full_response)
                        # st.rerun()

                    # Add assistant message to session state
                    st.session_state.messages.append(
                        {"role": "assistant", "content": full_response}
                    )
                    st.rerun()

            except Exception as e:
                logger.error(f"Error handling chat input: {str(e)}")
                st.error(ERROR_MESSAGES["general_error"])

    except Exception as e:
        logger.error(f"Critical error in main function: {str(e)}")
        st.error(ERROR_MESSAGES["general_error"])


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Application failed to start: {str(e)}")
        st.error(
            "⚠️ The application failed to start. Please refresh the page and try again."
        )
        st.error(
            "If the problem persists, please check your internet connection and API keys."
        )
