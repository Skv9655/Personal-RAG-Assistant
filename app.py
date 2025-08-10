import streamlit as st
import os
import tempfile
import logging
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime
import json

# Document processing
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# LLM and Agents
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType, Tool
# Memory is now handled with simple session state
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import BaseTool
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# Utilities
import requests
from dotenv import load_dotenv
import re

# Load environment variables
try:
    load_dotenv()
except Exception as e:
    print(f"Warning: Could not load .env file: {e}")

# Check if API key is available
if not os.getenv("GOOGLE_API_KEY"):
    print("❌ GOOGLE_API_KEY not found in environment variables. Please check your .env file.")
    # Don't stop here, let the application continue for testing

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Personal RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# Always reset memory to ensure it's the correct format
st.session_state.memory = {"conversations": [], "max_conversations": 10}

class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = "Useful for performing mathematical calculations"
    
    def _run(self, query: str) -> str:
        try:
            # Remove any non-mathematical text and evaluate
            query = re.sub(r'[^0-9+\-*/().\s]', '', query)
            result = eval(query)
            return f"Calculation result: {result}"
        except Exception as e:
            return f"Error in calculation: {str(e)}"
    
    def _arun(self, query: str):
        raise NotImplementedError("Async not implemented")

class DocumentRAG:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.vector_store = None
        self.llm = ChatGoogleGenerativeAI(
            model="gemma-3n-e2b-it",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1,
            max_tokens=2048
        )
        
    def load_document(self, file_path: str, file_type: str) -> List[str]:
        """Load document based on file type"""
        try:
            if file_type == "pdf":
                loader = PyPDFLoader(file_path)
            elif file_type == "txt":
                loader = TextLoader(file_path)
            elif file_type == "docx":
                loader = Docx2txtLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            documents = loader.load()
            return documents
        except Exception as e:
            logger.error(f"Error loading document: {e}")
            return []
    
    def split_documents(self, documents: List, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        return text_splitter.split_documents(documents)
    
    def create_vector_store(self, documents: List):
        """Create vector store from documents"""
        try:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            return True
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            return False
    
    def search_documents(self, query: str, k: int = 5):
        """Search documents for relevant information"""
        if not self.vector_store:
            return []
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            return docs
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

class RAGAssistant:
    def __init__(self):
        self.document_rag = DocumentRAG()
        self.search_tool = DuckDuckGoSearchRun()
        self.calculator_tool = CalculatorTool()
        # Use simple memory system
        self.memory = st.session_state.memory
        
        # Initialize agent with tools
        self.tools = [
            Tool(
                name="Document Search",
                func=self._search_documents,
                description="Search uploaded documents for relevant information"
            ),
            Tool(
                name="Web Search",
                func=self.search_tool.run,
                description="Search the web for current information"
            ),
            Tool(
                name="Calculator",
                func=self.calculator_tool._run,
                description="Perform mathematical calculations"
            )
        ]
        
        # Create agent without memory first, then add memory
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.document_rag.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def _search_documents(self, query: str) -> str:
        """Search documents and return relevant information"""
        docs = self.document_rag.search_documents(query)
        if not docs:
            return "No relevant information found in uploaded documents."
        
        context = "\n\n".join([doc.page_content for doc in docs])
        return f"Relevant information from documents:\n{context}"
    
    def process_query(self, query: str) -> str:
        """Process user query using RAG and agents"""
        try:
            # First try to find information in documents
            doc_info = self._search_documents(query)
            
            if "No relevant information found" in doc_info:
                # If no document info, use web search or direct LLM response
                try:
                    # Try agent first
                    response = self.agent.invoke({"input": query})
                    if isinstance(response, dict):
                        response = response.get("output", str(response))
                    else:
                        response = str(response)
                except Exception as agent_error:
                    logger.warning(f"Agent failed, using direct LLM: {agent_error}")
                    # Fallback to direct LLM response
                    response = self.document_rag.llm.invoke(query).content
            else:
                # Combine document info with LLM response
                prompt = f"""
                Based on the following information from uploaded documents:
                {doc_info}
                
                Please answer the user's question: {query}
                
                If the document information is sufficient, provide a detailed answer.
                If additional information is needed, mention what's missing and suggest using web search.
                """
                response = self.document_rag.llm.invoke(prompt).content
            
            return response
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Sorry, I encountered an error: {str(e)}"

def main():
    st.title("📚 Personal RAG Assistant")
    st.markdown("Upload documents and ask questions! I can also search the web and perform calculations.")
    
    # Initialize RAG assistant
    if 'rag_assistant' not in st.session_state:
        st.session_state.rag_assistant = RAGAssistant()
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            help="Upload PDF, TXT, or DOCX files"
        )
        
        if uploaded_files:
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    all_documents = []
                    
                    for uploaded_file in uploaded_files:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name
                        
                        try:
                            # Load and process document
                            file_type = uploaded_file.name.split('.')[-1].lower()
                            documents = st.session_state.rag_assistant.document_rag.load_document(tmp_file_path, file_type)
                            
                            if documents:
                                # Split documents
                                split_docs = st.session_state.rag_assistant.document_rag.split_documents(documents)
                                all_documents.extend(split_docs)
                                
                                st.success(f"✅ Processed {uploaded_file.name}")
                            
                            # Clean up temporary file
                            os.unlink(tmp_file_path)
                            
                        except Exception as e:
                            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                    
                    if all_documents:
                        # Create vector store
                        success = st.session_state.rag_assistant.document_rag.create_vector_store(all_documents)
                        if success:
                            st.session_state.vector_store = st.session_state.rag_assistant.document_rag.vector_store
                            st.success(f"✅ Vector store created with {len(all_documents)} document chunks")
                        else:
                            st.error("❌ Failed to create vector store")
        
        # Display uploaded files
        if st.session_state.vector_store:
            st.success("📚 Documents loaded and ready for queries!")
        
        # Memory info
        st.header("🧠 Memory")
        st.info(f"Remembering last {st.session_state.memory['max_conversations']} conversations")
        
        if st.button("Clear Memory"):
            st.session_state.memory["conversations"] = []
            st.session_state.chat_history = []
            st.success("Memory cleared!")

    # Main chat interface
    st.header("💬 Chat Interface")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents or anything else..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag_assistant.process_query(prompt)
                st.write(response)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Update memory
        conversation = {"user": prompt, "assistant": response}
        st.session_state.memory["conversations"].append(conversation)
        
        # Keep only the last N conversations
        if len(st.session_state.memory["conversations"]) > st.session_state.memory["max_conversations"]:
            st.session_state.memory["conversations"] = st.session_state.memory["conversations"][-st.session_state.memory["max_conversations"]:]
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <p>Powered by Gemma 1.5 Flash | Document RAG | Web Search | Calculator</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()