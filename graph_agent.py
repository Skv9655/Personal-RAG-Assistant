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
        try:
            if not documents:
                logger.warning("No documents to create vector store from")
                return False
            
            logger.info(f"Creating vector store with {len(documents)} document chunks")
            logger.info(f"Using embeddings model: {type(self.embeddings)}")
            
            # 1. Create FAISS Vector Store (Dense)
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            faiss_retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
            
            # 2. Create BM25 Retriever (Sparse)
            bm25_retriever = BM25Retriever.from_documents(documents)
            bm25_retriever.k = 5
            
            # 3. Create Ensemble Retriever (Hybrid)
            # Weights: 0.5 for semantic (FAISS), 0.5 for keyword (BM25)
            self.retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever],
                weights=[0.5, 0.5]
            )
            
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
        try:
            if not self.retriever:
                logger.warning("Retriever not initialized")
                return []
            
            # Update k for retrievers if needed (though Ensemble doesn't easily support dynamic k prop)
            # We rely on the configured k in create_vector_store
            
            logger.info(f"Searching documents with Hybrid Search for: {query}")
            results = self.retriever.invoke(query)
            logger.info(f"Found {len(results)} relevant documents via Hybrid Search")
            return results
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
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
        
        # Initialize DocumentRAG
        self.document_rag = DocumentRAG()
        
        # Initialize tools
        from langchain_community.tools import DuckDuckGoSearchRun
        self.search_tool = DuckDuckGoSearchRun()
        self.calculator_tool = CalculatorTool()

        # Define tools using standard Tool for maximum compatibility
        # This allows the model to just send a string or a simple JSON
        self.tools = [
            Tool(
                name="Document_Search",
                func=self._search_documents,
                description="Search uploaded documents for relevant information. Input should be the search query."
            ),
            Tool(
                name="Web_Search", 
                func=self.search_tool.run,
                description="Search the web for current information. Use this for recent events or general knowledge. Input should be the search query."
            ),
            Tool(
                name="Calculator",
                func=self.calculator_tool._run,
                description="Perform mathematical calculations. Input should be a mathematical expression like '2+2'."
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
                        # Handle different input formats
                        if isinstance(tool_input, dict):
                            query = tool_input.get("query", tool_input.get("q", str(tool_input)))
                        else:
                            query = str(tool_input)
                        logger.info(f"Executing tool {tool_name} with input: {query[:100]}...")
                        tool_output = tool.func(query)
                    except Exception as e:
                        tool_output = f"Error executing tool {tool_name}: {str(e)}"
                        logger.error(f"Tool execution error: {e}")
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
            logger.info("Calling LLM...")
            response = self.llm_with_tools.invoke(formatted_messages)
            logger.info("LLM response received")
            return {"messages": [response]}
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            # Fallback response
            return {
                "messages": [
                    AIMessage(content=f"I encountered an error: {str(e)[:200]}. Please try again or check your API configuration.")
                ]
            }

    # ---------------------------
    # Tools
    # ---------------------------
    def _search_documents(self, query: str):
        logger.info(f"Searching documents for: {query}")
        docs = self.document_rag.search_documents(query)
        if not docs:
            return "No relevant information found in uploaded documents."
        
        # Format the results
        results = []
        for i, doc in enumerate(docs[:3], 1):
            results.append(f"Result {i}: {doc.page_content[:500]}...")
        
        return "\n\n".join(results)

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