# app.py
import streamlit as st
import os
import tempfile
import logging
from dotenv import load_dotenv

from graph_agent import RAGAssistant

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Personal RAG Assistant",
    page_icon="📚",
    layout="wide",
)

if "rag_assistant" not in st.session_state:
    st.session_state.rag_assistant = RAGAssistant()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def main():
    st.title("📚 Personal RAG Assistant")
    st.markdown("Upload documents and ask questions!")

    with st.sidebar:
        st.header("📁 Upload Documents")
        files = st.file_uploader(
            "Upload files",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
        )

        if files and st.button("Process Documents"):
            docs = []
            for file in files:
                suffix = file.name.split(".")[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
                    tmp.write(file.getvalue())
                    path = tmp.name

                loaded = st.session_state.rag_assistant.document_rag.load_document(
                    path, suffix
                )
                docs.extend(
                    st.session_state.rag_assistant.document_rag.split_documents(
                        loaded
                    )
                )
                os.unlink(path)

            st.session_state.rag_assistant.document_rag.create_vector_store(docs)
            st.success("Documents indexed successfully!")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask something..."):
        st.session_state.chat_history.append(
            {"role": "user", "content": prompt}
        )
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.rag_assistant.process_query(prompt)
                
                # Check for errors
                if isinstance(result, dict) and "error" in result:
                    response_text = result["messages"][0].content
                    st.error(response_text)
                else:
                    # Extract response content
                    response_text = result["messages"][-1].content
                    st.write(response_text)
                    
                    # Extract sources from tool messages
                    sources = []
                    for msg in result["messages"]:
                        # check for tool messages
                        if hasattr(msg, "name") and msg.name == "Document_Search":
                            sources.append(msg.content)
                    
                    # Display sources if any
                    if sources:
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**Source {i}:**\n{source}")

        st.session_state.chat_history.append(
            {"role": "assistant", "content": response_text}
        )


if __name__ == "__main__":
    main()
