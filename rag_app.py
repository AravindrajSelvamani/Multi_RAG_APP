
import os
import hashlib
import streamlit as st

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from tempfile import NamedTemporaryFile

# ==========================================
# 1. ENVIRONMENT CONFIGURATION
# ==========================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(
    page_title="Multi-PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Multi-PDF RAG Chatbot")
st.caption("Upload multiple PDFs and ask questions using AI")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is missing in .env file")
    st.stop()


# ==========================================
# 2. INITIALIZE SESSION STATE
# ==========================================

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


# ==========================================
# 3. LOAD EMBEDDING MODEL
# ==========================================

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_llm():
    return ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0
    )


embeddings = load_embeddings()
llm = load_llm()


# ==========================================
# 4. PDF PROCESSING FUNCTION
# ==========================================

def process_pdfs(uploaded_pdfs):

    all_documents = []
    new_files = {}

    for uploaded_file in uploaded_pdfs:

        file_bytes = uploaded_file.getvalue()

        # Generate unique hash for PDF
        file_hash = hashlib.sha256(
            file_bytes
        ).hexdigest()

        # Skip already processed PDFs
        if file_hash in st.session_state.uploaded_files:
            st.info(
                f"{uploaded_file.name} already processed."
            )
            continue

        temp_path = None

        try:
            # Save uploaded PDF temporarily
            with NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(file_bytes)
                temp_path = temp_file.name

            # Load PDF
            loader = PyPDFLoader(temp_path)
            documents = loader.load()

            # Store original PDF filename
            for doc in documents:
                doc.metadata["source"] = uploaded_file.name

            all_documents.extend(documents)

            new_files[file_hash] = uploaded_file.name

        except Exception as e:
            st.error(
                f"Error processing {uploaded_file.name}: {e}"
            )

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    if not all_documents:
        return 0

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(all_documents)

    if not chunks:
        st.warning("No extractable text found in PDFs.")
        return 0

    # Create embeddings and FAISS index
    if st.session_state.vector_store is None:

        st.session_state.vector_store = (
            FAISS.from_documents(
                chunks,
                embeddings
            )
        )

    else:

        # Add new PDFs to existing vector store
        st.session_state.vector_store.add_documents(
            chunks
        )

    # Mark files as successfully processed
    st.session_state.uploaded_files.update(new_files)

    return len(chunks)


# ==========================================
# 5. RAG RESPONSE FUNCTION
# ==========================================

def generate_response(query):

    vector_store = st.session_state.vector_store

    # Retrieve relevant documents
    retrieved_docs = vector_store.similarity_search(
        query,
        k=4
    )

    # Build context with source information
    context = "\n\n".join(
        [
            f"Source: {doc.metadata.get('source', 'Unknown')}\n"
            f"Page: {doc.metadata.get('page', 0) + 1}\n"
            f"Content: {doc.page_content}"
            for doc in retrieved_docs
        ]
    )

    # Create prompt template
    prompt_template = ChatPromptTemplate.from_template(
        """
        You are a helpful AI assistant.

        Answer the user's question using only
        the provided PDF context.

        If the answer is not present in the context,
        say "I could not find this information
        in the uploaded PDFs."

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    )

    # LangChain pipeline
    chain = prompt_template | llm

    response = chain.invoke({
        "context": context,
        "question": query
    })

    # Extract source details
    sources = []

    for doc in retrieved_docs:

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        page = doc.metadata.get("page", 0) + 1

        source_text = f"{source} - Page {page}"

        if source_text not in sources:
            sources.append(source_text)

    return response.content, sources


# ==========================================
# 6. SIDEBAR - MULTI PDF UPLOAD
# ==========================================

with st.sidebar:

    st.header("📁 Upload PDF Documents")

    uploaded_pdfs = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"pdf_uploader_{st.session_state.uploader_key}"
    )

    if st.button(
        "Process New PDFs",
        use_container_width=True
    ):

        if uploaded_pdfs:

            with st.spinner(
                "Processing PDFs and creating embeddings..."
            ):

                try:
                    total_chunks = process_pdfs(
                        uploaded_pdfs
                    )

                    if total_chunks > 0:
                        st.success(
                            f"{total_chunks} chunks added!"
                        )

                        # Reset uploader to allow new files
                        st.session_state.uploader_key += 1

                        st.rerun()

                except Exception as e:
                    st.error(str(e))

        else:
            st.warning("Please upload PDF files.")

    st.divider()

    st.subheader("📚 Document Library")

    if st.session_state.uploaded_files:

        for filename in st.session_state.uploaded_files.values():
            st.write(f"📄 {filename}")

        st.success(
            f"Total PDFs: "
            f"{len(st.session_state.uploaded_files)}"
        )

    else:
        st.info("No PDFs uploaded yet.")

    st.divider()

    if st.button(
        "Clear Chat History",
        use_container_width=True
    ):
        st.session_state.chat_history = []
        st.rerun()

    if st.button(
        "Clear All Documents",
        use_container_width=True
    ):
        st.session_state.vector_store = None
        st.session_state.uploaded_files = {}
        st.session_state.chat_history = []
        st.session_state.uploader_key += 1
        st.rerun()


# ==========================================
# 7. DISPLAY CHAT HISTORY
# ==========================================

for chat in st.session_state.chat_history:

    with st.chat_message(chat["role"]):

        st.markdown(chat["content"])

        if chat["role"] == "assistant":

            if chat.get("sources"):

                with st.expander("📖 Sources"):

                    for source in chat["sources"]:
                        st.write(source)


# ==========================================
# 8. CHAT INPUT
# ==========================================

user_query = st.chat_input(
    "Ask questions about your PDFs..."
)

if user_query:

    if st.session_state.vector_store is None:

        st.warning(
            "Please upload and process PDFs first."
        )

    else:

        # Save user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_query
        })

        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate AI response
        with st.chat_message("assistant"):

            with st.spinner("Searching PDFs..."):

                try:
                    response, sources = generate_response(
                        user_query
                    )

                    st.markdown(response)

                    with st.expander("📖 Sources"):

                        for source in sources:
                            st.write(source)

                    # Save assistant response
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response,
                        "sources": sources
                    })

                except Exception as e:
                    st.error(f"Error: {e}")
