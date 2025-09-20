# 🎓 Academic RAG Assistant

> Transform your academic textbooks into an intelligent AI tutor using advanced RAG (Retrieval Augmented Generation) technology

![Python](https://img.shields.io/badge/python-v3.12+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-latest-red.svg)
![LangChain](https://img.shields.io/badge/langchain-latest-green.svg)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![GitHub Stars](https://img.shields.io/github/stars/ZohaibCodez/academic-rag-assistant.svg)](https://github.com/ZohaibCodez/academic-rag-assistant/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/ZohaibCodez/academic-rag-assistant.svg)](https://github.com/ZohaibCodez/academic-rag-assistant/network)
[![GitHub Watchers](https://img.shields.io/github/watchers/ZohaibCodez/academic-rag-assistant.svg)](https://github.com/ZohaibCodez/academic-rag-assistant/watchers)
[![GitHub Contributors](https://img.shields.io/github/contributors/ZohaibCodez/academic-rag-assistant.svg)](https://github.com/ZohaibCodez/academic-rag-assistant/graphs/contributors)
[![Issues](https://img.shields.io/github/issues/ZohaibCodez/academic-rag-assistant.svg)](https://github.com/ZohaibCodez/academic-rag-assistant/issues)
![Last Commit](https://img.shields.io/github/last-commit/ZohaibCodez/academic-rag-assistant)
[![Live Demo](https://img.shields.io/badge/demo-online-green.svg)](https://academic-rag-assistant.streamlit.app)
## 🎯 Overview

Academic RAG Assistant is an intelligent tutoring system that transforms your course textbooks into an interactive AI conversation partner. Using advanced agentic RAG architecture, it provides contextual answers from your specific academic materials across multiple subjects including Linear Algebra, Discrete Structures, and Calculus & Analytical Geometry.

## ✨ Features

- 📚 **Multi-Subject Expertise**: Specialized tools for Linear Algebra, Discrete Structures, and Calculus & Analytical Geometry
- 🤖 **Agentic RAG System**: Intelligent query enhancement and routing for optimal retrieval
- 🔄 **Real-time Streaming**: Live response generation with visual typing indicators
- 📱 **Modern UI**: Clean, responsive dark-themed interface with custom styling
- 💾 **Session Management**: Persistent chat history with export functionality
- 🎛️ **Model Selection**: Support for multiple Google Gemini models (2.5 Pro, Flash, 2.0 Flash, etc.)
- 🛡️ **Error Handling**: Comprehensive error management with user-friendly messages
- 📊 **Progress Tracking**: Visual feedback during initialization and processing
- 🔍 **Smart Retrieval**: MMR and multi-query retrieval strategies for better context
- 🧠 **Memory System**: SQLite-based conversation persistence

## 🛠️ Tech Stack

- **Frontend**: Streamlit with custom CSS styling
- **Backend**: Python with asyncio for concurrent operations
- **AI Framework**: OpenAI Agents SDK for agentic behavior
- **LLM Integration**: Google Gemini via OpenAI-compatible API
- **Vector Database**: Pinecone for document storage and retrieval
- **Embeddings**: HuggingFace Sentence Transformers (all-MiniLM-L6-v2)
- **Text Processing**: LangChain framework for RAG implementation
- **Memory**: SQLite for session and conversation management

## 📋 Prerequisites

- Python 3.12 or higher
- Google Gemini API Key ([Get one here](https://makersuite.google.com/app/apikey))
- Pinecone API Key ([Get one here](https://www.pinecone.io/))
- Pre-processed textbook data in Pinecone vector store

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/ZohaibCodez/academic-rag-assistant.git
cd academic-rag-assistant
```

### 2. Install Dependencies

```bash
# Using pip
pip install -r requirements.txt

# Using uv (recommended)
uv sync
```

### 3. Set Up Environment

```bash
# Create .env file
cp .env.example .env
# Edit .env with your API keys
```

### 4. Run the Application

```bash
streamlit run app.py
```

### 5. Access the App

Open your browser and navigate to `http://localhost:8501`

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

### Supported Models

- `gemini-2.5-pro` (Most capable, recommended for complex analysis)
- `gemini-2.5-flash` (Balanced performance and speed)
- `gemini-2.5-flash-lite` (Lightweight and fast)
- `gemini-2.0-flash` (Fast responses with good accuracy)
- `gemini-1.5-pro` (Reliable baseline model)
- `gemini-1.5-flash` (Quick processing)

### Configurable Parameters

```python
CHUNK_OVERLAP = 100        # Text chunk overlap for context
RETRIEVER_K_MMR = 2       # MMR retrieval count
RETRIEVER_K_SIMILARITY = 5 # Similarity search count
LAMBDA_MUL = 0.7          # MMR diversity parameter
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

## 📱 How to Use

1. **Enter API Keys**: Add your Google Gemini and Pinecone API keys in the sidebar
2. **Select Model**: Choose your preferred Gemini model from the dropdown
3. **Start Learning**: Ask questions about your coursework in natural language
4. **View Subjects**: Check available subjects in the "Subjects" tab
5. **Export History**: Download your conversation anytime from the "Info" tab

### 📚 Subjects Supported

#### Linear Algebra

- Matrix operations and properties
- Systems of linear equations (Gaussian elimination, substitution)
- Eigenvalues and eigenvectors
- Vector spaces and transformations
- Determinants and matrix inverses

#### Discrete Structures

- Mathematical logic and proof techniques
- Set theory and relations
- Graph theory and trees
- Combinatorics and counting principles
- Boolean algebra and functions

#### Calculus & Analytical Geometry

- Limits and continuity
- Differentiation techniques and applications
- Integration methods and applications
- Analytical geometry in 2D and 3D
- Sequences and series

### Example Queries

- "Explain the steps to solve a system of linear equations using Gaussian elimination"
- "What is mathematical induction and how do I write a proof?"
- "How do you find the derivative of a composite function using chain rule?"
- "What are eigenvalues and eigenvectors? Provide examples"
- "Explain the fundamental theorem of calculus with applications"

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Query Enhancer  │───▶│ Subject Router  │
│   (Natural)     │    │  (Agent System)  │    │ (Classification)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Streamlit UI   │◀───│   Agent Runner   │◀───│  Function Tools │
│   (Frontend)    │    │  (Orchestrator)  │    │ (Subject RAG)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                               │                        │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Session Store  │    │  Gemini Models   │    │ Pinecone Vector │
│   (SQLite)      │    │ (Generation AI)  │    │   Database      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                ┌─────────────────┐
                                                │  HuggingFace    │
                                                │   Embeddings    │
                                                └─────────────────┘
```

## 🐳 Docker Support

### Using Docker

```bash
# Create .env file with your API keys
echo "GOOGLE_API_KEY=your-gemini-key-here" >> .env
echo "PINECONE_API_KEY=your-pinecone-key-here" >> .env

# Build and run
docker build -t academic-rag-assistant .
docker run -p 8501:8501 --env-file .env academic-rag-assistant
```

## 📁 Project Structure

```
academic-rag-assistant/
│
├── app.py                              # Main Streamlit application
│
├── notebooks/
│   └── data_preparation_pipeline.ipynb # Complete RAG pipeline setup
│       ├── Step 1a → Multi-Document Ingestion
│       ├── Step 1b → Subject-Aware Text Splitting
│       ├── Step 2 → Retrieval System Setup
│       ├── Step 3 → Tool Definitions
│       └── Agentic RAG Final Form
│
├── logs/                               # Application logs directory
├── Dockerfile                          # Container configuration
├── requirements.txt                    # Python dependencies
├── pyproject.toml                      # Project configuration
├── uv.lock                            # UV dependency lock
├── .env.example                        # Example environment variables
├── .gitignore                          # Git ignore rules
└── README.md                           # Project documentation
```

## 📊 Performance Metrics

- **Query Processing**: ~1-3 seconds for typical academic queries
- **Memory Usage**: Optimized vector storage with Pinecone
- **Retrieval Accuracy**: High precision with multi-strategy retrieval
- **Response Quality**: Enhanced by agentic query reformulation
- **Concurrent Users**: Supports multiple simultaneous sessions
- **Streaming Speed**: Real-time response generation with 0.05s intervals

## ⚠️ Current Limitations

- **Subject Scope**: Limited to three core subjects (Linear Algebra, Discrete Structures, Calculus)
- **Language**: Optimized for English academic content
- **Data Dependency**: Requires pre-processed textbooks in Pinecone
- **API Limits**: Subject to Google Gemini and Pinecone rate limits
- **Context Window**: Limited by model context length for very long documents

## 🔄 Data Preparation Pipeline

The included Jupyter notebook (`data_preparation_pipeline.ipynb`) provides a complete walkthrough:

### Step 1a: Multi-Document Ingestion

- PDF text extraction and processing
- Document metadata handling
- Quality validation and cleanup

### Step 1b: Subject-Aware Text Splitting

- Intelligent chunking based on academic structure
- Subject-specific namespace organization
- Context preservation across chunks

### Step 2: Retrieval System Setup

- Vector store initialization with Pinecone
- Embedding model configuration
- Index creation and optimization

### Step 3: Tool Definitions

- Subject-specific RAG function tools
- Query enhancement and routing logic
- Response formatting and validation

### Agentic RAG Final Form

- Complete agent system integration
- Testing and validation procedures
- Performance optimization techniques

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Install development dependencies (`uv sync`)
4. Make your changes with comprehensive logging
5. Test across multiple Gemini models
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add comprehensive logging for new features
- Include error handling for all external API calls
- Update documentation for new functionality
- Test with multiple subjects and query types

## 📝 Future Roadmap

- [ ] Support for additional subjects (Physics, Chemistry, Statistics)
- [ ] Multi-language academic content support
- [ ] Advanced visualization tools for mathematical concepts
- [ ] Integration with learning management systems
- [ ] Collaborative study session features
- [ ] Custom textbook upload and processing
- [ ] Mobile-responsive design improvements
- [ ] Voice interaction capabilities
- [ ] Progress tracking and learning analytics
- [ ] API endpoint for programmatic access

## 🐛 Known Issues

- Large textbook corpora may require extended initialization time
- Complex mathematical notation may not render perfectly
- API rate limiting may affect performance during peak usage
- Memory usage can be high with multiple concurrent users

## 🔧 Troubleshooting

### Common Issues

**"Agent initialization failed" error:**

- Verify both Google Gemini and Pinecone API keys are valid
- Check internet connectivity and API service status
- Ensure sufficient API quota remaining

**"Vector store connection failed":**

- Confirm Pinecone API key and index configuration
- Verify the "semester-books" index exists with correct namespaces
- Check Pinecone service status and regional settings

**Slow response times:**

- Try switching to gemini-2.5-flash for faster responses
- Check your network connection stability
- Consider using a different model if quota limits are reached

**Memory errors:**

- Restart the Streamlit application
- Clear browser cache and session storage
- For Docker: increase memory allocation limits

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenAI Agents SDK](https://github.com/openai/agents) for the agentic framework
- [Streamlit](https://streamlit.io/) for the incredible web framework
- [LangChain](https://langchain.com/) for comprehensive RAG implementation
- [Google AI](https://ai.google.dev/) for Gemini API access
- [Pinecone](https://www.pinecone.io/) for scalable vector database services
- [HuggingFace](https://huggingface.co/) for open-source embedding models
- Academic community for inspiration and feedback

## 📞 Support

If you encounter any issues or have questions:

- Open an [Issue](https://github.com/ZohaibCodez/academic-rag-assistant/issues)
- Check existing issues for solutions
- Review the troubleshooting section above
- Contact: [itxlevicodez@gmail.com](mailto:itxlevicodez@gmail.com)

---

⭐ **Star this repository if you found it helpful for your academic journey!**

Built with 🖤 for students by [@ZohaibCodez](https://github.com/ZohaibCodez) using Google Gemini AI and advanced RAG techniques
