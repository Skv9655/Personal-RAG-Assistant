# Personal RAG Assistant

A comprehensive Retrieval-Augmented Generation (RAG) system built with Streamlit that allows you to upload documents and ask questions about them. The system also includes web search and calculator capabilities.

## Features

- 📚 **Document Upload**: Support for PDF, TXT, and DOCX files
- 🤖 **Gemma 1.5 Flash Model**: Powered by Google's latest language model
- 🔍 **RAG System**: Retrieve relevant information from uploaded documents
- 🌐 **Web Search**: Search the web for additional information when needed
- 🧮 **Calculator**: Perform mathematical calculations
- 🧠 **Memory**: Remember the last 10 conversations
- 💬 **Chat Interface**: Modern chat-based UI with Streamlit
- 📱 **Responsive Design**: Works on desktop and mobile devices

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd Mini_RAG
```

### 2. Create Virtual Environment
```bash
python -m venv venv
```

### 3. Activate Virtual Environment
**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Set Up Environment Variables
Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_google_api_key_here
```

**Important Security Note:** 
- Never commit your `.env` file to version control
- The `.env` file is already added to `.gitignore` for security
- Replace `your_google_api_key_here` with your actual Google API key

### 6. Run the Application
```bash
streamlit run Rag_Personal_Project.py
```

The application will open in your browser at `http://localhost:8501`

## Usage Guide

### 1. Upload Documents
- Use the sidebar to upload PDF, TXT, or DOCX files
- Click "Process Documents" to create the vector store
- Wait for the processing to complete

### 2. Ask Questions
- Type your questions in the chat interface
- The system will:
  - First search uploaded documents for relevant information
  - If information is found, provide detailed answers
  - If not found, use web search for additional information
  - Use calculator for mathematical queries

### 3. Example Questions
For novels:
- "What is the moral of the story?"
- "Who is the main character?"
- "Who is the villain and why?"
- "What are the key themes?"

For legal documents:
- "What are the main legal implications?"
- "What are the key terms and conditions?"
- "What are the penalties mentioned?"

### 4. Memory Management
- The system remembers the last 10 conversations
- Use "Clear Memory" button to reset conversation history

## Project Structure

```
Mini_RAG/
├── Rag_Personal_Project.py    # Main application file
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── .env                      # Environment variables (create this)
└── venv/                     # Virtual environment
```

## Technical Details

### Models Used
- **LLM**: Gemma 3 27B (Google)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: FAISS

### Tools Available
1. **Document Search**: Searches uploaded documents
2. **Web Search**: DuckDuckGo search for current information
3. **Calculator**: Mathematical calculations

### Memory System
- Uses ConversationBufferWindowMemory
- Keeps last 10 conversations
- Persists during session

## Deployment

### Streamlit Cloud
1. Push your code to GitHub
2. Connect your repository to Streamlit Cloud
3. Set environment variables in Streamlit Cloud dashboard
4. Deploy

### Local Deployment
```bash
streamlit run Rag_Personal_Project.py --server.port 8501 --server.address 0.0.0.0
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **API Key Issues**: Verify your Google API key is correct and has proper permissions

3. **Memory Issues**: For large documents, consider reducing chunk size in the code

4. **Performance**: The system works best with documents under 100MB

### Performance Tips
- Use smaller chunk sizes for better retrieval
- Limit document size for faster processing
- Clear memory periodically for better performance

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is for personal use. Please respect the terms of service for all APIs used.

## Security

### API Key Management
- API keys are stored in `.env` file (not in code)
- `.env` file is excluded from version control via `.gitignore`
- Never share your API keys publicly
- For deployment, set environment variables in your hosting platform

### Best Practices
- Use environment variables for all sensitive data
- Regularly rotate your API keys
- Monitor API usage to prevent unexpected charges
- Keep your dependencies updated

## Support

For issues and questions, please check the troubleshooting section or create an issue in the repository.
