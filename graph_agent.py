from __future__ import annotations
# graph_agent.py
import os
import re
import uuid
import logging
from datetime import datetime
from typing import List, Any, Sequence, TypedDict, Annotated
from pydantic.v1.fields import FieldInfo as FieldInfoV1
from dotenv import load_dotenv

# LangChain core
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import Tool, StructuredTool
from pydantic import BaseModel, Field

# LangChain integrations
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# Updated to non-deprecated embeddings (requires: pip install langchain-huggingface)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
# Robust import for EnsembleRetriever
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    try:
        from langchain.retrievers.ensemble import EnsembleRetriever
    except ImportError:
        # Fallback implementation if module is missing
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.callbacks import CallbackManagerForRetrieverRun
        from langchain_core.documents import Document
        from typing import List
        
        class EnsembleRetriever(BaseRetriever):
            retrievers: List[BaseRetriever]
            weights: List[float] = None
            
            def _get_relevant_documents(
                self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
            ) -> List[Document]:
                # Simple fallback: collect unique docs from all retrievers
                results = []
                seen = set()
                # Gather from all sources
                for r in self.retrievers:
                    try:
                        docs = r.invoke(query)
                        for doc in docs:
                            if doc.page_content not in seen:
                                results.append(doc)
                                seen.add(doc.page_content)
                    except Exception:
                        continue
                return results[:5]

# LangGraph
from langgraph.graph import StateGraph, END

# Prompt
from prompt import SYSTEM_PROMPT

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress DuckDuckGo Wikipedia DNS errors (non-critical - DuckDuckGo falls back to other engines)
# These errors occur because Wikipedia DNS lookup fails, but DuckDuckGo uses multiple engines
# and will successfully return results from other engines (Google, Brave, etc.)
ddgs_logger = logging.getLogger("ddgs.ddgs")
ddgs_logger.setLevel(logging.ERROR)  # Suppress INFO/WARNING level Wikipedia DNS errors

# Also suppress primp logger which may show similar non-critical errors
primp_logger = logging.getLogger("primp")
primp_logger.setLevel(logging.WARNING)  # Only show warnings/errors, not INFO level messages


# ---------------------------
# LangGraph State
# ---------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]


# ---------------------------
# Calculator Tool
# ---------------------------
class CalculatorTool:
    name: str = "calculator"
    description: str = "Useful for performing mathematical calculations"

    def _run(self, query):
        try:
            # Clean the query to only allow safe characters
            query = re.sub(r"[^0-9+\-*/().\s]", "", query)
            result = eval(query)
            return f"Calculation result: {result}"
        except Exception as e:
            return f"Error in calculation: {str(e)}"

    async def _arun(self, query):
        raise NotImplementedError


# ---------------------------
# Document RAG
# ---------------------------
class DocumentRAG:
    def __init__(self):
        # Initialize embeddings with strict CPU enforcement
        try:
            # Direct SentenceTransformer loading to avoid meta tensor issues
            from sentence_transformers import SentenceTransformer
            # Force CPU usage
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
            
            # Wrap in HuggingFaceEmbeddings using the loaded model
            # Note: We pass the model name string but the actual loading happened above to verify/cache
            # For the actual LangChain wrapper, we use strict kwargs
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("Embeddings initialized successfully on CPU")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            # Last resort fallback - if embeddings fail, app shouldn't crash
            logger.warning("Falling back to FakeEmbeddings to prevent crash")
            from langchain_community.embeddings import FakeEmbeddings
            self.embeddings = FakeEmbeddings(size=384)
        
        
        self.vector_store = None
        self.retriever = None
        self.llm = None

        # Try Groq first, then fallback to GitHub Models
        groq_api_key = os.getenv("GROQ_API_KEY")
        github_token = os.getenv("GITHUB_TOKEN")
        
        if groq_api_key:
            logger.info("Attempting to use Groq")
            try:
                # Try to import ChatGroq
                from langchain_groq import ChatGroq
                logger.info("Successfully imported ChatGroq")
                self.llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    groq_api_key=groq_api_key,
                    temperature=0.7,
                    max_tokens=1024,
                    # Moved top_p to model_kwargs to fix the warning
                    model_kwargs={"top_p": 1.0},
                )
                logger.info("Groq LLM initialized successfully")
            except ImportError as e:
                logger.warning(f"langchain_groq not available: {e}")
                if github_token:
                    logger.info("Falling back to GitHub Models")
                    self._setup_github_llm(github_token)
            except Exception as e:
                logger.error(f"Failed to initialize Groq: {e}")
                if github_token:
                    logger.info("Falling back to GitHub Models due to error")
                    self._setup_github_llm(github_token)
        elif github_token:
            logger.info("Using GitHub Models (Azure AI)")
            self._setup_github_llm(github_token)
        else:
            logger.warning("No GROQ_API_KEY or GITHUB_TOKEN found in environment")
            # Create a dummy LLM for testing (won't actually work but allows app to start)
            self.llm = None
    
    def _setup_github_llm(self, github_token):
        """Setup GitHub Models LLM"""
        try:
            self.llm = ChatOpenAI(
                model=os.getenv("GITHUB_MODEL", "meta-llama/Llama-3.2-3B-Instruct"),
                api_key=github_token,
                base_url=os.getenv("GITHUB_BASE_URL", "https://models.inference.ai.azure.com"),
                temperature=0.7,
                top_p=0.9,
                max_tokens=1024,
            )
            logger.info("GitHub Models LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize GitHub Models: {e}")
            self.llm = None

    def load_document(self, file_path: str, file_type: str):
        try:
            if file_type == "pdf":
                loader = PyPDFLoader(file_path)
            elif file_type == "txt":
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_type == "docx":
                loader = Docx2txtLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} documents from {file_path}")
            return documents
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            raise

    def split_documents(self, documents, chunk_size=1000, chunk_overlap=200):
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            chunks = splitter.split_documents(documents)
            logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            raise

    def create_vector_store(self, documents):
        """
        Create FAISS vector store from document chunks.
        
        Process:
        1. Validates documents have content
        2. Creates FAISS vector store with embeddings (semantic similarity)
        3. Creates BM25 retriever (keyword search)
        4. Combines into Ensemble retriever (hybrid search)
        5. Falls back to FAISS-only if BM25 fails
        
        The FAISS vector store enables semantic similarity search using cosine similarity
        between query embeddings and document chunk embeddings.
        """
        try:
            if not documents:
                logger.warning("No documents to create vector store from")
                return False
            
            logger.info(f"Creating vector store with {len(documents)} document chunks")
            logger.info(f"Using embeddings model: {type(self.embeddings)}")
            
            # Validate documents have content
            valid_docs = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
            if not valid_docs:
                logger.error("No valid documents with content found")
                return False
            
            logger.info(f"Processing {len(valid_docs)} valid documents out of {len(documents)} total")
            
            # 1. Create FAISS Vector Store (Dense)
            try:
                logger.info("Creating FAISS vector store...")
                self.vector_store = FAISS.from_documents(valid_docs, self.embeddings)
                faiss_retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
                logger.info(f"FAISS vector store created: {self.vector_store.index.ntotal} vectors")
            except Exception as e:
                logger.error(f"Failed to create FAISS vector store: {e}", exc_info=True)
                return False
            
            # 2. Create BM25 Retriever (Sparse)
            try:
                logger.info("Creating BM25 retriever...")
                bm25_retriever = BM25Retriever.from_documents(valid_docs)
                bm25_retriever.k = 5
                logger.info("BM25 retriever created successfully")
            except Exception as e:
                logger.error(f"Failed to create BM25 retriever: {e}", exc_info=True)
                # Fallback to FAISS only if BM25 fails
                logger.warning("Falling back to FAISS-only retriever")
                self.retriever = faiss_retriever
                return True
            
            # 3. Create Ensemble Retriever (Hybrid)
            try:
                logger.info("Creating Ensemble retriever...")
                # Weights: 0.5 for semantic (FAISS), 0.5 for keyword (BM25)
                self.retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, faiss_retriever],
                    weights=[0.5, 0.5]
                )
                logger.info("Ensemble retriever created successfully")
            except Exception as e:
                logger.error(f"Failed to create Ensemble retriever: {e}", exc_info=True)
                # Fallback to FAISS only
                logger.warning("Falling back to FAISS-only retriever")
                self.retriever = faiss_retriever
                return True
            
            # Verify initialization
            if self.vector_store and self.retriever:
                logger.info(f"Hybrid vector store created successfully. Index size: {self.vector_store.index.ntotal}")
                return True
            else:
                logger.error("Vector store or retriever object is None after creation")
                return False
        except Exception as e:
            logger.error(f"Error creating vector store: {e}", exc_info=True)
            return False

    def search_documents(self, query, k=5):
        """
        Search documents using FAISS semantic similarity search.
        Uses cosine similarity to find the most relevant document chunks.
        """
        try:
            # First, try using FAISS vector store directly for semantic similarity
            if self.vector_store:
                logger.info(f"Searching documents using FAISS semantic similarity for: '{query}'")
                # Use similarity_search_with_score to get relevance scores
                results_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
                
                # Filter results by relevance (score < 1.0 means good similarity)
                # Lower score = more similar
                filtered_results = []
                for doc, score in results_with_scores:
                    if score < 1.5:  # Threshold for semantic similarity
                        filtered_results.append(doc)
                        logger.info(f"Found relevant chunk (similarity score: {score:.3f})")
                
                if filtered_results:
                    logger.info(f"Found {len(filtered_results)} relevant documents via FAISS semantic search")
                    return filtered_results
                elif results_with_scores:
                    # If no results pass threshold, return top results anyway
                    logger.warning(f"No results passed similarity threshold, returning top {len(results_with_scores)} results")
                    return [doc for doc, score in results_with_scores]
            
            # Fallback to retriever if vector_store not available
            if self.retriever:
                logger.info(f"Using retriever fallback for: '{query}'")
                results = self.retriever.invoke(query)
                logger.info(f"Found {len(results)} relevant documents via retriever")
                return results
            
            logger.warning("Neither vector_store nor retriever is initialized")
            return []
        except Exception as e:
            logger.error(f"Error searching documents: {e}", exc_info=True)
            return []


# ---------------------------
# Tool Schemas
# ---------------------------
class SearchSchema(BaseModel):
    query: str = Field(description="The search or question to look up")

class CalculatorSchema(BaseModel):
    expression: str = Field(description="The mathematical expression to evaluate (e.g., '2+2')")

# ---------------------------
# RAG Assistant
# ---------------------------
class RAGAssistant:
    def __init__(self):
        logger.info("Initializing RAGAssistant")
        
        # Store API keys for runtime fallback
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.current_api = None  # Track which API is currently active
        self.fallback_attempted = False  # Prevent multiple fallback attempts
        
        # Initialize DocumentRAG
        self.document_rag = DocumentRAG()
        
        # Determine which API was initialized
        if self.document_rag.llm:
            llm_model_name = getattr(self.document_rag.llm, "model_name", 
                                    getattr(self.document_rag.llm, "model", "unknown"))
            if "groq" in str(type(self.document_rag.llm)).lower() or "groq" in str(llm_model_name).lower():
                self.current_api = "groq"
            else:
                self.current_api = "github"
            logger.info(f"Initialized with {self.current_api.upper()} API")
        
        # Initialize tools
        from langchain_community.tools import DuckDuckGoSearchRun
        # Note: DuckDuckGo Wikipedia errors are suppressed at module level
        self.search_tool = DuckDuckGoSearchRun()
        self.calculator_tool = CalculatorTool()

        # Define tools using StructuredTool with proper schemas for Groq compatibility
        # Groq's API requires structured schemas, but we use wrapper functions to handle variations
        self.tools = [
            StructuredTool.from_function(
                func=self._search_documents_wrapper,
                name="Document_Search",
                description="""Search uploaded documents using FAISS semantic similarity. 
                This tool uses vector embeddings to find document chunks that are semantically similar to your query.
                ALWAYS use this tool FIRST for any question - it searches through the user's uploaded documents.
                REQUIRED parameter: 'query' (string) - the search query or question to find in documents.""",
                args_schema=SearchSchema
            ),
            StructuredTool.from_function(
                func=self._web_search_wrapper,
                name="Web_Search",
                description="Search the web for current information. Use this for recent events or general knowledge. REQUIRED parameter: 'query' (string) - the search query.",
                args_schema=SearchSchema
            ),
            StructuredTool.from_function(
                func=self._calculate_wrapper,
                name="Calculator",
                description="Perform mathematical calculations. REQUIRED parameter: 'expression' (string) - the mathematical expression like '2+2'.",
                args_schema=CalculatorSchema
            )
        ]
        

        # Initialize LLM with tools if available
        if self.document_rag.llm:
            logger.info("LLM available, attempting to bind tools")
            try:
                # Robust model name access (ChatGroq uses 'model_name')
                # Use getattr with a default to avoid AttributeError
                llm_model_name = getattr(self.document_rag.llm, "model_name", 
                                        getattr(self.document_rag.llm, "model", "unknown"))
                logger.info(f"LLM model identified: {llm_model_name}")
                
                # Check if it's a known model that supports tools, or just try to bind
                # Groq models that support tool calling
                groq_tool_supporting_models = [
                    "llama-3.3-70b-versatile",
                    "llama-3.1-70b-versatile",
                    "llama-3.1-8b-instant",
                    "llama-3.2-90b-vision-preview",
                    "mixtral-8x7b-32768",
                    "gemma2-9b-it"
                ]
                
                # Robust string matching for model support
                model_str = str(llm_model_name).lower()
                is_gemini = "gemini" in model_str
                is_supported_groq = any(m.lower() in model_str for m in groq_tool_supporting_models)
                
                if is_gemini or is_supported_groq:
                    # For Groq, bind tools directly
                    # Note: Groq's API may have quirks with tool schema validation
                    self.llm_with_tools = self.document_rag.llm.bind_tools(self.tools)
                    logger.info("Tools successfully bound to LLM")
                else:
                    # Try to bind anyway if it looks like it might support it
                    try:
                        self.llm_with_tools = self.document_rag.llm.bind_tools(self.tools)
                        logger.info("Tools bound by default to LLM")
                    except Exception:
                        logger.warning(f"Model {llm_model_name} might not support tool calling. Using basic mode.")
                        self.llm_with_tools = self.document_rag.llm
            except Exception as e:
                logger.warning(f"Could not bind tools to LLM: {e}")
                self.llm_with_tools = self.document_rag.llm
        else:
            logger.warning("No LLM available")
            self.llm_with_tools = None

        # Build the graph
        self.graph = self._build_graph()
        logger.info("RAGAssistant initialized successfully")
    
    def _initialize_github_fallback(self):
        """Initialize GitHub LLM as fallback when Groq fails"""
        if not self.github_token:
            logger.warning("GitHub token not available for fallback")
            return False
        
        try:
            logger.info("Initializing GitHub LLM as fallback...")
            from langchain_openai import ChatOpenAI
            
            github_llm = ChatOpenAI(
                model=os.getenv("GITHUB_MODEL", "meta-llama/Llama-3.2-3B-Instruct"),
                api_key=self.github_token,
                base_url=os.getenv("GITHUB_BASE_URL", "https://models.inference.ai.azure.com"),
                temperature=0.7,
                top_p=0.9,
                max_tokens=1024,
            )
            
            # Update DocumentRAG's LLM
            self.document_rag.llm = github_llm
            
            # Try to bind tools
            try:
                self.llm_with_tools = github_llm.bind_tools(self.tools)
                logger.info("GitHub LLM initialized and tools bound successfully")
            except Exception as tool_error:
                logger.warning(f"Could not bind tools to GitHub LLM: {tool_error}")
                self.llm_with_tools = github_llm
            
            self.current_api = "github"
            return True
        except Exception as e:
            logger.error(f"Failed to initialize GitHub LLM fallback: {e}")
            return False

    # ---------------------------
    # LangGraph
    # ---------------------------
    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", self._execute_tools)

        workflow.set_entry_point("agent")

        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {"continue": "tools", "end": END},
        )

        workflow.add_edge("tools", "agent")

        return workflow.compile()

    def _should_continue(self, state: AgentState):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "continue"
        return "end"

    def _execute_tools(self, state: AgentState):
        """Execute tools manually"""
        from langchain_core.messages import ToolMessage
        
        messages = state["messages"]
        last_message = messages[-1]
        tool_calls = last_message.tool_calls if hasattr(last_message, "tool_calls") else []
        
        tool_messages = []
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_input = tool_call["args"]
            
            # Find and execute the tool
            tool_output = "Tool not found"
            for tool in self.tools:
                if tool.name == tool_name:
                    try:
                        # For StructuredTool, use invoke() which handles schema validation
                        # tool_input is already a dict from tool_calls["args"]
                        logger.info(f"Executing tool {tool_name} with input: {tool_input}")
                        tool_output = tool.invoke(tool_input)
                    except Exception as e:
                        tool_output = f"Error executing tool {tool_name}: {str(e)}"
                        logger.error(f"Tool execution error: {e}", exc_info=True)
                    break
            
            tool_messages.append(
                ToolMessage(
                    content=str(tool_output),
                    tool_call_id=tool_call["id"],
                    name=tool_name
                )
            )
        
        return {"messages": tool_messages}

    def _call_model(self, state: AgentState):
        if not self.llm_with_tools:
            return {
                "messages": [
                    AIMessage(content="No API keys configured. Please set GROQ_API_KEY or GITHUB_TOKEN in .env file.")
                ]
            }

        messages = state["messages"]
        
        # Add system message if not present
        if not any(isinstance(m, SystemMessage) for m in messages):
            system_msg = SYSTEM_PROMPT.format(
                current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            # Convert messages to correct format
            formatted_messages = [SystemMessage(content=system_msg)]
            formatted_messages.extend(messages)
        else:
            formatted_messages = list(messages)

        try:
            logger.info(f"Calling LLM ({self.current_api.upper() if self.current_api else 'unknown'})...")
            response = self.llm_with_tools.invoke(formatted_messages)
            logger.info("LLM response received")
            return {"messages": [response]}
        except Exception as e:
            error_str = str(e)
            logger.error(f"Error calling LLM ({self.current_api.upper() if self.current_api else 'unknown'}): {e}")
            
            # If using Groq and it fails, try to fallback to GitHub (only once)
            if self.current_api == "groq" and self.github_token and not self.fallback_attempted:
                logger.info("Groq API failed, attempting to fallback to GitHub API...")
                self.fallback_attempted = True  # Mark that we've attempted fallback
                if self._initialize_github_fallback():
                    try:
                        logger.info("Retrying with GitHub API...")
                        response = self.llm_with_tools.invoke(formatted_messages)
                        logger.info("GitHub API response received successfully")
                        return {"messages": [response]}
                    except Exception as github_error:
                        logger.error(f"GitHub API also failed: {github_error}")
                        # Continue to final fallback
            
            # Check if it's a Groq tool validation error (even after fallback attempt)
            if "__arg1" in error_str and "tool call validation" in error_str.lower():
                # This is a known Groq API quirk - try retrying without tools as fallback
                logger.warning("Tool validation error detected. Retrying without tools...")
                try:
                    # Retry without tools
                    logger.info("Retrying without tools...")
                    response = self.document_rag.llm.invoke(formatted_messages)
                    return {"messages": [response]}
                except Exception as retry_error:
                    logger.error(f"Retry without tools also failed: {retry_error}")
            
            # Final fallback response
            api_info = f" (Current API: {self.current_api.upper()})" if self.current_api else ""
            return {
                "messages": [
                    AIMessage(content=f"I encountered an error: {str(e)[:200]}. Please try again or check your API configuration.{api_info}")
                ]
            }

    # ---------------------------
    # Tools
    # ---------------------------
    def _search_documents_wrapper(self, query: str = None, **kwargs):
        """
        Search uploaded documents using FAISS semantic similarity.
        Returns relevant document chunks that match the query semantically.
        """
        # StructuredTool passes 'query' as keyword argument
        # But we also handle cases where it might come as dict or other formats
        if query:
            query_str = str(query)
        elif kwargs:
            # Try to extract from kwargs or dict input
            query = (kwargs.get("query") or 
                    kwargs.get("search query") or 
                    kwargs.get("q") or
                    kwargs.get("search_query") or
                    kwargs.get("__arg1") or
                    list(kwargs.values())[0] if kwargs else "")
            query_str = str(query) if query else ""
        else:
            query_str = ""
        
        if not query_str:
            return "No search query provided."
        
        logger.info(f"Searching documents using semantic similarity for: '{query_str}'")
        
        # Check if vector store is initialized
        if not self.document_rag.vector_store:
            logger.warning("Vector store not initialized. Please upload and process documents first.")
            return "No documents have been uploaded yet. Please upload documents first using the sidebar."
        
        # Search using FAISS semantic similarity
        docs = self.document_rag.search_documents(query_str, k=5)
        
        if not docs:
            logger.info("No relevant documents found via semantic search")
            return "No relevant information found in uploaded documents for this query. Try rephrasing your question or check if the documents contain this information."
        
        # Format the results with more context
        results = []
        results.append(f"Found {len(docs)} relevant document chunk(s) for: '{query_str}'\n")
        
        for i, doc in enumerate(docs, 1):
            # Include metadata if available
            metadata_info = ""
            if hasattr(doc, 'metadata') and doc.metadata:
                source = doc.metadata.get('source', '')
                page = doc.metadata.get('page', '')
                if source:
                    metadata_info = f" [Source: {source.split('/')[-1]}]"
                if page:
                    metadata_info += f" [Page: {page}]"
            
            # Return more content (up to 800 chars) for better context
            content = doc.page_content.strip()
            if len(content) > 800:
                content = content[:800] + "..."
            
            results.append(f"--- Document Chunk {i}{metadata_info} ---\n{content}\n")
        
        return "\n".join(results)
    
    def _web_search_wrapper(self, query: str = None, **kwargs):
        """
        Web search wrapper that handles schema parameters.
        Note: DuckDuckGo Wikipedia DNS errors are suppressed at module level (non-critical).
        """
        # StructuredTool passes 'query' as keyword argument
        if query:
            query_str = str(query)
        elif kwargs:
            # Try to extract from kwargs
            query = (kwargs.get("query") or 
                    kwargs.get("search query") or 
                    kwargs.get("q") or
                    kwargs.get("search_query") or
                    kwargs.get("__arg1") or
                    list(kwargs.values())[0] if kwargs else "")
            query_str = str(query) if query else ""
        else:
            query_str = ""
        
        if not query_str:
            return "No search query provided."
        
        try:
            # DuckDuckGo Wikipedia errors are suppressed at module level
            # The search will still work using other engines (Google, Brave, etc.)
            result = self.search_tool.run(query_str)
            return result
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return f"Error performing web search: {str(e)[:200]}"
    
    def _calculate_wrapper(self, expression: str = None, **kwargs):
        """Wrapper that handles schema parameters and variations from Groq"""
        # StructuredTool passes 'expression' as keyword argument
        if expression:
            expr_str = str(expression)
        elif kwargs:
            # Try to extract from kwargs
            expression = (kwargs.get("expression") or 
                         kwargs.get("expr") or 
                         kwargs.get("calc") or
                         kwargs.get("__arg1") or
                         list(kwargs.values())[0] if kwargs else "")
            expr_str = str(expression) if expression else ""
        else:
            expr_str = ""
        
        return self.calculator_tool._run(expr_str)
    
    def _normalize_tool_input(self, tool_name: str, tool_input):
        """Normalize tool input to handle parameter name variations from LLM"""
        if not isinstance(tool_input, dict):
            return tool_input
        
        # Handle search tools (Document_Search, Web_Search)
        if tool_name in ["Document_Search", "Web_Search"]:
            # Try common variations of query parameter, including Groq's __arg1 format
            query = (tool_input.get("query") or 
                    tool_input.get("search query") or 
                    tool_input.get("q") or 
                    tool_input.get("search_query") or
                    tool_input.get("__arg1"))  # Groq format
            if query:
                return {"query": query}
            # If no query found, try to extract from any string value
            for key, value in tool_input.items():
                if isinstance(value, str):
                    return {"query": value}
            # Fallback: return as-is
            return tool_input
        
        # Handle Calculator tool
        elif tool_name == "Calculator":
            expression = (tool_input.get("expression") or 
                         tool_input.get("expr") or 
                         tool_input.get("calc") or
                         tool_input.get("__arg1"))  # Groq format
            if expression:
                return {"expression": expression}
            # If no expression found, try to extract from any string value
            for key, value in tool_input.items():
                if isinstance(value, str):
                    return {"expression": value}
            # Fallback: return as-is
            return tool_input
        
        return tool_input

    def process_query(self, query: str):
        try:
            logger.info(f"Processing query: {query}")
            result = self.graph.invoke(
                {"messages": [HumanMessage(content=query)]}
            )
            # Return the full result so we can extract sources
            return result
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Error processing your query: {str(e)[:200]}"