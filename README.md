# Personal RAG Assistant 🤖📚

A powerful, agentic Retrieval-Augmented Generation (RAG) system built with **Streamlit**, **LangChain**, and **LangGraph**. It combines the speed of **Groq** with the robustness of **GitHub Models (Azure AI)** to provide accurate answers from your documents and the web.

## 🚀 Key Features

- **⚡ Blazing Fast AI**: Uses **Groq** (Llama 3.1 8B) as the primary engine for near-instant responses.
- **🛡️ Robust Fallback System**: 
  - **Initialization Fallback**: Automatically switches to **GitHub Models** if Groq fails during startup
  - **Runtime Fallback**: Automatically switches to GitHub Models if Groq fails during API calls (rate limits, errors, etc.)
- **🧠 Semantic Search**: Uses **FAISS** vector database with cosine similarity for accurate semantic document retrieval
- **🔍 Hybrid Search**: Combines **FAISS** (Semantic Vector Search) and **BM25** (Keyword Search) using an `EnsembleRetriever` for superior retrieval accuracy
- **🛠️ Agentic Capabilities**:
  - **Document Search**: Uses FAISS semantic similarity to find relevant document chunks
  - **Web Search**: Uses DuckDuckGo to find real-time information (Wikipedia errors suppressed)
  - **Calculator**: Handles mathematical queries precisely
- **📄 Multi-Format Support**: Upload PDF, TXT, and DOCX files
- **🔗 Source Citations**: Displays exact source chunks used to answer questions with metadata (file name, page number)
- **✨ Smart Document Processing**: 
  - Automatic chunking with overlap
  - Content validation
  - Progress indicators
  - Error handling with detailed feedback

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Orchestration**: LangChain & LangGraph
- **LLMs**: 
  - Primary: Groq (`llama-3.1-8b-instant`)
  - Fallback: GitHub Models (`meta-llama/Llama-3.2-3B-Instruct`)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (CPU-only for compatibility)
- **Vector Store**: FAISS (semantic similarity search)
- **Keyword Search**: BM25 (via rank_bm25)
- **Search Engine**: DuckDuckGo (with error suppression for non-critical failures)

## 📋 Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd Personal-RAG-Assistant
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# Activate:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory:
```env
# Primary LLM (Groq) - Required
GROQ_API_KEY=gsk_...

# Secondary/Fallback LLM (GitHub Models / Azure AI) - Optional but recommended
GITHUB_TOKEN=github_pat_...
GITHUB_MODEL=meta-llama/Llama-3.2-3B-Instruct
GITHUB_BASE_URL=https://models.inference.ai.azure.com
```

**Note**: You can use either one or both API keys. If both are provided, Groq will be used first with automatic fallback to GitHub Models if Groq fails.

### 5. Run the Application
```bash
streamlit run app.py
```
The app will open at `http://localhost:8501`.

## 💡 Usage Guide

### Document Upload & Processing
1. **Upload Documents**: Use the sidebar to upload your PDF, TXT, or DOCX files
2. **Process Documents**: Click "🚀 Process & Index Documents"
   - Documents are automatically chunked (1000 chars with 200 char overlap)
   - Embedded using sentence transformers
   - Indexed in FAISS vector database
   - Status indicators show progress and results
3. **View Status**: Check the sidebar for document status and chunk count

### Asking Questions
The system follows this workflow:
1. **Document Search First**: Always searches uploaded documents using semantic similarity
2. **Answer from Documents**: Uses document content to answer your question
3. **Web Search Fallback**: Only uses web search if documents don't contain the answer

**Example Queries**:
- *"What are the phases of project submission?"* (Uses Document Search)
- *"Summarize the key points from my uploaded document"* (Uses Document Search)
- *"What is the current stock price of Apple?"* (Uses Web Search - current events)
- *"Calculate 15% of 8500"* (Uses Calculator)

### Viewing Sources
- Expand the "📚 Document Sources" section below answers to see:
  - Exact document chunks used
  - Source file names
  - Page numbers (for PDFs)
  - Similarity scores

## 🔧 How It Works

### Document Processing Pipeline
1. **Load**: Documents are loaded based on file type (PDF/TXT/DOCX)
2. **Split**: Documents are split into chunks (1000 chars, 200 overlap)
3. **Embed**: Chunks are converted to embeddings using sentence transformers
4. **Store**: Embeddings are stored in FAISS vector database
5. **Index**: BM25 keyword index is created for hybrid search

### Search Process
1. **Query Embedding**: User query is converted to embedding
2. **Semantic Search**: FAISS finds most similar document chunks using cosine similarity
3. **Relevance Filtering**: Results are filtered by similarity score (< 1.5 threshold)
4. **Hybrid Enhancement**: BM25 provides keyword-based results (if available)
5. **Result Ranking**: Ensemble retriever combines semantic + keyword results

### LLM Response Generation
1. **Tool Selection**: LLM decides which tool to use (Document_Search, Web_Search, Calculator)
2. **Tool Execution**: Selected tool is executed with the query
3. **Context Building**: Tool results are added to the conversation context
4. **Answer Generation**: LLM generates answer based on tool results
5. **Source Attribution**: Sources are extracted and displayed

## 📂 Project Structure

```
Personal-RAG-Assistant/
├── app.py              # Main Streamlit application with UI
├── graph_agent.py      # Core RAG logic, LangGraph setup, tool definitions
├── prompt.py           # System prompts for LLM
├── requirements.txt    # Project dependencies
├── .env                # API keys (not committed - create your own)
└── README.md           # This file
```

## 🎯 Key Components

### `app.py`
- Streamlit UI and user interface
- Document upload and processing
- Chat interface
- Source display
- Session state management

### `graph_agent.py`
- `DocumentRAG`: Handles document loading, chunking, embedding, and vector store creation
- `RAGAssistant`: Main agent with LangGraph workflow
- Tool definitions: Document_Search, Web_Search, Calculator
- API fallback logic (Groq → GitHub)
- Error handling and logging

### `prompt.py`
- System prompt that guides LLM behavior
- Emphasizes document-first approach
- Tool usage instructions

## ⚠️ Troubleshooting

### Common Issues

1. **"Retriever not initialized" Warning**
   - **Cause**: Documents haven't been uploaded and processed
   - **Solution**: Upload documents and click "Process & Index Documents"

2. **"No relevant information found in uploaded documents"**
   - **Cause**: Query doesn't match document content semantically
   - **Solution**: Try rephrasing your question or check if documents contain the information

3. **Groq API Errors**
   - **Cause**: Rate limits, API key issues, or service unavailability
   - **Solution**: System automatically falls back to GitHub Models if configured

4. **Import Errors**
   - **Cause**: Missing dependencies
   - **Solution**: Run `pip install -r requirements.txt`

5. **Embedding Errors**
   - **Cause**: GPU/CUDA issues
   - **Solution**: The system enforces CPU-only execution automatically

6. **DuckDuckGo Wikipedia Errors**
   - **Cause**: DNS issues with Wikipedia (non-critical)
   - **Solution**: These are automatically suppressed - search still works using other engines

### API Configuration

- **Groq API Key**: Get from [console.groq.com](https://console.groq.com)
- **GitHub Token**: Get from [github.com/settings/tokens](https://github.com/settings/tokens)
- **Both APIs**: Recommended for reliability - automatic fallback ensures continuous service

## 🔒 Security Notes

- Never commit your `.env` file
- API keys are loaded from environment variables
- Documents are processed locally (not sent to external services except LLM APIs)
- Temporary files are automatically cleaned up

## 📝 License

[Add your license here]

## 🤝 Contributing

[Add contribution guidelines here]

## 🙏 Acknowledgments

- LangChain & LangGraph for the agent framework
- Groq for fast inference
- GitHub Models (Azure AI) for fallback support
- Streamlit for the UI framework
- FAISS for efficient vector search
