import streamlit as st
from io import StringIO
import fitz  # PyMuPDF
from llm_prompt import  *
from langchain.document_loaders import UnstructuredURLLoader

READER_LLM, RAG_PROMPT_TEMPLATE = get_llm(READER_MODEL_NAME = "stabilityai/stablelm-2-zephyr-1_6b")


# Set page title and layout
st.set_page_config(page_title="Q&A App", layout="wide")

# Title of the app
st.title("Document & URL Q&A App")

# Sidebar for document upload and URL input
st.sidebar.header("Upload Document or Enter URL")

# Option to upload a document
uploaded_file = st.sidebar.file_uploader("Choose a document...", type=["txt", "pdf"])

# Option to input a URL
url = st.sidebar.text_input("Enter a URL")

# Display the uploaded document or URL
document_content = None

if uploaded_file is not None:
    st.subheader("Uploaded Document")
    if uploaded_file.type == "text/plain":
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        document_content = stringio.read()
        st.text_area("Document content:", document_content, height=300)

        docs = get_text_chunks_langchain(document_content)
    elif uploaded_file.type == "application/pdf":
        # Read PDF file
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        document_content = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            document_content += page.get_text()
        st.text_area("Document content:", document_content, height=300)

        docs = get_text_chunks_langchain(document_content)
    else:
        st.warning("Only .txt and .pdf files are supported at this time.")
elif url:
    st.subheader("Entered URL")
    st.write(f"URL: {url}")

    llm_loader = UnstructuredURLLoader(urls=[url])
    docs = llm_loader.load()

else:
  docs = None

# Input for user's question
st.subheader("Ask a Question")
question = st.text_input("Enter your question:")

# Placeholder for the answer
if question:
    st.subheader("Answer")
    # Placeholder where you will handle the answer logic
    if docs != None:
      final_answer = get_answer(READER_LLM, RAG_PROMPT_TEMPLATE, docs, question)
      st.write(final_answer)

# Instructions or notes
st.sidebar.info(
    """
    Please upload a document or enter a URL for the content you want to ask questions about.
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
        }
    </style>
    """, unsafe_allow_html=True)
