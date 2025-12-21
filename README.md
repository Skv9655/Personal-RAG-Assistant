# Personal RAG Assistant 🤖📚

A powerful, agentic Retrieval-Augmented Generation (RAG) system built with **Streamlit**, **LangChain**, and **LangGraph**. It combines the speed of **Groq** with the robustness of **GitHub Models (Azure AI)** to provide accurate answers from your documents and the web.

## 🚀 Key Features

- **⚡ Blazing Fast AI**: Uses **Groq** (Llama 3.1 8B) as the primary engine for near-instant responses.
- **🛡️ Robust Fallback**: Automatically switches to **GitHub Models** (Llama 3.1 405B) if Groq is unavailable.
- **🧠 Hybrid Search**: Combines **FAISS** (Semantic Vector Search) and **BM25** (Keyword Search) using an `EnsembleRetriever` for superior retrieval accuracy.
- **🛠️ Agentic Capabilities**:
  - **Document Search**: Intelligently queries uploaded files.
  - **Web Search**: Uses DuckDuckGo to find real-time information.
  - **Calculator**: Handles mathematical queries precisely.
- **📄 Multi-Format Support**: Upload PDF, TXT, and DOCX files.
- **🔗 Source Citations**: Displays exact source chunks used to answer questions completely with a "Sources" expander.

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Orchestration**: LangChain & LangGraph
- **LLMs**: Groq (`llama-3.1-8b-instant`), GitHub Models (`meta-llama-3.1-405b-instruct`)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (Running specifically on CPU for compatibility)
- **Vector Store**: FAISS
- **Keyword Search**: rank_bm25

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
# Primary LLM (Groq)
GROQ_API_KEY=gsk_...

# Secondary/Fallback LLM (GitHub Models / Azure AI)
GITHUB_TOKEN=github_pat_...
GITHUB_MODEL=meta-llama-3.1-405b-instruct
GITHUB_BASE_URL=https://models.inference.ai.azure.com
```

### 5. Run the Application
```bash
streamlit run app.py
```
The app will open at `http://localhost:8501`.

## 💡 Usage Guide

1.  **Upload Docs**: Use the sidebar to upload your PDF/TXT/DOCX files.
2.  **Process**: Click "Process Documents". The system will indexing them using Hybrid Search (Vectors + Keywords).
3.  **Chat**: Ask questions like:
    - *"Summarize the uploaded marketing plan"* (Uses Document Search)
    - *"What is the stock price of Apple today?"* (Uses Web Search)
    - *"Calculate 15% of 8500"* (Uses Calculator)
4.  **View Sources**: Expand the "📚 Sources" section below the answer to see exactly where the information came from.

## 📂 Project Structure

```
Personal-RAG-Assistant/
├── app.py              # Main Streamlit application
├── graph_agent.py      # Core RAG logic, Tool definitions, and LangGraph setup
├── prompt.py           # System prompts
├── requirements.txt    # Project dependencies
├── .env                # API keys (not committed)
└── README.md           # Documentation
```

## ⚠️ Troubleshooting

-   **Import Errors**: If you see `ModuleNotFoundError: No module named 'langchain.retrievers'`, run `pip install -r requirements.txt` to ensure `rank_bm25` and strict LangChain versions are installed.
-   **Embedding Crash**: If the app crashes on start with `MetaTensor` errors, ensure you are running the latest version of `graph_agent.py` which enforces CPU execution for embeddings.
