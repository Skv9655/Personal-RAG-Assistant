
import sys
import pkg_resources

print(f"Python version: {sys.version}")

try:
    import langchain
    print(f"LangChain version: {langchain.__version__}")
    print(f"LangChain path: {langchain.__file__}")
except ImportError:
    print("LangChain not installed")

try:
    from langchain.retrievers import EnsembleRetriever
    print("SUCCESS: from langchain.retrievers import EnsembleRetriever")
except ImportError as e:
    print(f"FAILED: from langchain.retrievers import EnsembleRetriever ({e})")

try:
    from langchain.retrievers.ensemble import EnsembleRetriever
    print("SUCCESS: from langchain.retrievers.ensemble import EnsembleRetriever")
except ImportError as e:
    print(f"FAILED: from langchain.retrievers.ensemble import EnsembleRetriever ({e})")

try:
    from langchain_community.retrievers import EnsembleRetriever
    print("SUCCESS: from langchain_community.retrievers import EnsembleRetriever")
except ImportError as e:
    print(f"FAILED: from langchain_community.retrievers import EnsembleRetriever ({e})")
