import streamlit as st
import os
import tempfile
import logging
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.messages import ToolMessage

from graph_agent import RAGAssistant

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Personal RAG Assistant",
    page_icon="📚",
    layout="wide",
)

# ---------------------------
# SESSION STATE INITIALIZATION
# ---------------------------
def initialize_session_state():
    """Initialize ALL session state variables safely"""
    defaults = {
        "rag_assistant": None,
        "vector_store_ready": False,
        "processed_files": [],
        "total_chunks": 0,
        "chat_history": [],
        "documents_uploaded": False
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            if key == "rag_assistant":
                st.session_state[key] = RAGAssistant()
            else:
                st.session_state[key] = default_value

# Initialize session state immediately
initialize_session_state()


# ---------------------------
# MAIN APP
# ---------------------------
def main():
    st.title("📚 Personal RAG Assistant")
    st.markdown("Upload documents and ask questions!")
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("📁 Document Management")
        
        # Status display - SAFE access with .get()
        if st.session_state.get("vector_store_ready", False):
            st.success(f"✅ Documents Ready!")
            st.info(f"• {st.session_state.get('total_chunks', 0)} document chunks indexed")
            processed_files = st.session_state.get("processed_files", [])
            if processed_files:
                st.info(f"• {len(processed_files)} file(s) loaded")
        else:
            st.warning("⚠️ No documents loaded")
            st.info("Upload PDF, TXT, or DOCX files to enable document search")
        
        st.divider()
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            help="Upload multiple files. Supported: PDF, TXT, DOCX"
        )
        
        # Process button
        if uploaded_files and st.button("🚀 Process & Index Documents", type="primary"):
            process_documents(uploaded_files)
        
        st.divider()
        
        # Clear button
        if st.session_state.get("vector_store_ready", False):
            if st.button("🗑️ Clear All Documents"):
                clear_documents()
        
        # API status
        st.divider()
        st.header("⚙️ System Status")
        assistant = st.session_state.get("rag_assistant")
        if assistant and assistant.document_rag.llm:
            llm_type = type(assistant.document_rag.llm).__name__
            st.success(f"• LLM: {llm_type}")
        else:
            st.error("• LLM: Not configured")
        
        if assistant and assistant.document_rag.vector_store:
            st.success(f"• Vector Store: Ready")
        else:
            st.warning("• Vector Store: Not ready")
    
    # Main chat interface
    display_chat_history()
    
    # Chat input
    if prompt := st.chat_input("Ask about your documents..."):
        handle_user_query(prompt)


# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def process_documents(uploaded_files):
    """Process uploaded documents"""
    with st.spinner("🔄 Processing documents..."):
        all_chunks = []
        processed_count = 0
        errors = []
        
        for file in uploaded_files:
            try:
                # Save to temp file
                suffix = file.name.split(".")[-1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
                    tmp.write(file.getvalue())
                    tmp_path = tmp.name
                
                # Load document
                st.info(f"Loading {file.name}...")
                documents = st.session_state.rag_assistant.document_rag.load_document(
                    tmp_path, suffix
                )
                
                if not documents:
                    errors.append(f"❌ Failed to load {file.name}")
                    continue
                
                # Split documents
                st.info(f"Splitting {file.name}...")
                chunks = st.session_state.rag_assistant.document_rag.split_documents(documents)
                
                if chunks:
                    all_chunks.extend(chunks)
                    processed_count += 1
                    st.session_state.processed_files.append(file.name)
                    st.success(f"✅ {file.name}: {len(chunks)} chunks")
                else:
                    errors.append(f"⚠️ No content in {file.name}")
                
                # Clean up temp file
                os.unlink(tmp_path)
                
            except Exception as e:
                errors.append(f"❌ Error processing {file.name}: {str(e)[:100]}")
                logger.error(f"Error with {file.name}: {e}", exc_info=True)
        
        # Create vector store if we have chunks
        if all_chunks:
            st.info(f"Creating vector index with {len(all_chunks)} chunks...")
            success = st.session_state.rag_assistant.document_rag.create_vector_store(all_chunks)
            
            if success:
                st.session_state.vector_store_ready = True
                st.session_state.total_chunks = len(all_chunks)
                st.session_state.documents_uploaded = True
                
                st.balloons()
                st.success(f"🎉 Successfully processed {processed_count}/{len(uploaded_files)} files!")
                st.success(f"📊 {len(all_chunks)} document chunks indexed and ready for search!")
            else:
                st.error("❌ Failed to create vector store. Check logs for details.")
        else:
            st.error("❌ No valid document content found in any uploaded files.")
        
        # Show errors if any
        if errors:
            with st.expander("⚠️ Processing Errors", expanded=True):
                for error in errors:
                    st.error(error)


def clear_documents():
    """Clear all documents from memory"""
    st.session_state.rag_assistant.document_rag.vector_store = None
    st.session_state.rag_assistant.document_rag.retriever = None
    st.session_state.vector_store_ready = False
    st.session_state.processed_files = []
    st.session_state.total_chunks = 0
    st.session_state.documents_uploaded = False
    st.rerun()


def display_chat_history():
    """Display chat history"""
    chat_history = st.session_state.get("chat_history", [])
    for msg in chat_history:
        with st.chat_message(msg.get("role", "user")):
            st.write(msg.get("content", ""))
            
            # Show sources if available
            if msg.get("sources"):
                with st.expander("📚 View Sources"):
                    for source in msg.get("sources", []):
                        st.write(source)


def handle_user_query(prompt):
    """Handle user query and display response"""
    # Ensure chat_history exists
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Add user message to chat
    st.session_state.chat_history.append({
        "role": "user", 
        "content": prompt
    })
    
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            # Check if documents are available
            if not st.session_state.get("documents_uploaded", False):
                st.warning("⚠️ No documents uploaded yet. I'll use general knowledge only.")
            
            # Process query
            result = st.session_state.rag_assistant.process_query(prompt)
            
            # Extract response
            if isinstance(result, dict) and "messages" in result:
                messages = result["messages"]
                if messages:
                    last_msg = messages[-1]
                    response_text = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
                    
                    # Display response
                    st.write(response_text)
                    
                    # Extract sources from tool messages
                    sources = []
                    for msg in messages:
                        # Check if it's a ToolMessage by checking for 'name' attribute
                        if hasattr(msg, 'name') and msg.name == "Document_Search":
                            sources.append(msg.content)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response_text,
                        "sources": sources if sources else None
                    })
                    
                    # Display sources if available
                    if sources and "No relevant information" not in sources[0]:
                        with st.expander("📚 Document Sources"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**Source {i}:**")
                                st.write(source)
                else:
                    st.error("No response from assistant")
            else:
                error_msg = str(result) if result else "Unknown error"
                st.error(f"Error: {error_msg[:200]}")
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_msg
                })


# ---------------------------
# MAIN EXECUTION
# ---------------------------
if __name__ == "__main__":
    # One more safety check
    if "rag_assistant" not in st.session_state:
        st.session_state.rag_assistant = RAGAssistant()
    
    main()