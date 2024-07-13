import streamlit as st
from io import StringIO
import fitz  # PyMuPDF
from llm_prompt import get_prompt_embeddingllm, get_text_chunks_langchain, get_answer
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

RAG_PROMPT_TEMPLATE, embedding_llm = get_prompt_embeddingllm()

# Set page title and layout
st.set_page_config(page_title="Q&A App", layout="wide")

# Title of the app
st.title("Document & Text Q&A App")

# Sidebar for document upload and text input
st.sidebar.header("Upload Document or Paste Text")

# Option to upload a document
uploaded_file = st.sidebar.file_uploader("Choose a document...", type=["txt", "pdf"])

# Option to paste text
pasted_text = st.sidebar.text_area("Paste your text here")

# Display the uploaded document or pasted text
document_content = None

if uploaded_file is not None:
    st.subheader("Uploaded Document")
    if uploaded_file.type == "text/plain":
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        document_content = stringio.read()
        st.text_area("Document content:", document_content, height=300)
        docs = get_text_chunks_langchain(document_content)
    elif uploaded_file.type == "application/pdf":
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        document_content = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            document_content += page.get_text()
        st.text_area("Document content:", document_content, height=300)
        docs = get_text_chunks_langchain(document_content)
    else:
        st.warning("Only .txt and .pdf files are supported at this time.")
elif pasted_text:
    st.subheader("Pasted Text")
    st.text_area("Pasted content:", pasted_text, height=300)
    docs = get_text_chunks_langchain(pasted_text)
else:
    docs = None

# Input for user's question
st.subheader("Ask a Question")
question = st.text_input("Enter your question:")

# Placeholder for the answer
if question:
    st.subheader("Answer")
    if docs:
        final_answer = get_answer(embedding_llm, RAG_PROMPT_TEMPLATE, docs, question)
        st.write(final_answer)
    else:
        st.warning("Please upload a document or paste text before asking a question.")

# Instructions or notes
st.sidebar.info(
    """
    Please upload a document or enter the content you want to ask questions about.
    Then, enter your question in the main area.
    """
)

# Add a footer
st.markdown("""
    <style>
        footer {
            visibility: hidden;
        }
        footer:after {
            content:'Q&A App by YourName';
            visibility: visible;
            display: block;
            position: relative;
            padding: 10px;
            top: 2px;
            font-size: 14px;
            text-align: center;
        }
        .css-18e3th9 {
            padding: 1rem 2rem 2rem 2rem;
        }
        .css-1d391kg input {
            padding: 10px;
            font-size: 16px;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
    </style>
    """, unsafe_allow_html=True)
