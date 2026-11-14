# Multi-PDF RAG Chatbot

A Streamlit-based Retrieval-Augmented Generation (RAG) application that lets you upload multiple PDF documents, process them into embeddings, and ask natural-language questions about their content.

This app uses:
- Python and Streamlit for the web interface
- LangChain for document processing and prompt orchestration
- PyPDFLoader to read PDF content
- RecursiveCharacterTextSplitter to chunk documents
- HuggingFace embeddings for vectorization
- FAISS as the vector database
- OpenAI GPT models for answer generation

## Features

- Upload multiple PDF files at once
- Automatically extract text from uploaded PDFs
- Split documents into manageable chunks
- Store embeddings in a FAISS vector store
- Search the most relevant chunks for a user question
- Answer questions using only the uploaded document context
- Display source document names and page references for traceability
- Clear chat history or remove all loaded documents as needed

## Project Structure

```text
Day9_RAG/
├── rag_app.py              # Main Streamlit app
├── requirements.txt         # Python dependencies
├── .env                    # Environment variables (not committed in some setups)
├── rag_vector_store/       # Saved FAISS vector store data
└── README.md               # Project documentation
```

## Tech Stack

- Python 3.11+
- Streamlit
- LangChain
- LangChain Community
- LangChain OpenAI
- HuggingFace Embeddings
- FAISS CPU
- PyPDF
- python-dotenv

## Prerequisites

Before running the app, make sure you have:

- Python installed
- A valid OpenAI API key
- Access to the required Python packages

## Environment Setup

1. Create a `.env` file in the project folder.
2. Add your OpenAI API key:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

## Installation

From the project directory:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run rag_app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## How It Works

1. Upload one or more PDF files in the sidebar.
2. Click "Process New PDFs" to extract and split the content.
3. The app creates vector embeddings for each text chunk.
4. When you ask a question, it retrieves the most relevant chunks from the FAISS index.
5. The LLM answers using only the retrieved PDF context.
6. Relevant source files and pages are displayed in the response.

## Usage Example

- Upload policy documents or research PDFs
- Ask: "What are the key risks mentioned in this document?"
- Ask: "Summarize the steps for implementation in the uploaded files."

## Notes

- The app requires a valid OpenAI API key to generate responses.
- It works best with text-rich PDFs.
- If a file is already processed, the app avoids re-embedding it.
- The vector store persists under `rag_vector_store/` for reuse across sessions.

## License

This project is for learning and development purposes.

## Author

Built as a GenAI learning project for experimenting with multi-document RAG workflows.
