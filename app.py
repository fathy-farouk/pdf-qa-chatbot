import streamlit as st
import os
import shutil
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from PIL import Image

# Set Tesseract path manually (Windows users)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Ensure Poppler is set for pdf2image (Windows users)
poppler_path = r"C:\path\to\poppler-xx\bin"  # Update with your actual Poppler path
os.environ["PATH"] += os.pathsep + poppler_path

# Load API key from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Streamlit UI Setup
st.set_page_config(page_title="AI PDF Q&A Chatbot", page_icon="📄")
st.title("📄 AI PDF Q&A Chatbot")
st.markdown("Upload a PDF and ask anything about its content!")

# File uploader
uploaded_file = st.file_uploader("📤 Upload your PDF file", type=["pdf"])

# Function to extract text from scanned PDF using OCR
def extract_text_from_scanned_pdf(pdf_path):
    text = ""
    try:
        images = convert_from_path(pdf_path)
        for image in images:
            text += pytesseract.image_to_string(image, lang="eng") + "\n"
    except Exception as e:
        st.error(f"❌ OCR Processing Failed: {str(e)}")
        return None
    return text.strip()

# Process PDF and update FAISS index
if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        # Save file temporarily
        pdf_path = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)
        
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Check if the PDF is scanned (has no selectable text)
        doc = fitz.open(pdf_path)
        is_scanned = all([not page.get_text("text") for page in doc])

        if is_scanned:
            st.warning("🔍 Scanned PDF detected! Applying OCR for text extraction.")
            extracted_text = extract_text_from_scanned_pdf(pdf_path)
            if not extracted_text:
                st.error("❌ OCR failed to extract any text. Try another file.")
                st.stop()
            else:
                documents = [{"page_content": extracted_text, "metadata": {"source": pdf_path}}]
        else:
            st.success("✅ Selectable text found! Processing without OCR.")
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()

        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(documents)

        # Convert to embeddings and store in FAISS
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local("faiss_index")

        st.success("✅ PDF Processed Successfully! Now you can ask questions.")

# Load FAISS Index
if os.path.exists("faiss_index"):
    vector_store = FAISS.load_local(
        "faiss_index", 
        OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY), 
        allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever()
    llm = ChatOpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # User Query Input
    query = st.text_input("🔍 Ask a question about the uploaded PDF:")
    
    if query:
        with st.spinner("Searching..."):
            response = qa_chain.run(query)
        
        st.subheader("📌 Answer:")
        st.write(response)

# Cleanup temp files
if os.path.exists("temp"):
    shutil.rmtree("temp")

import time

# Cleanup temp files
if os.path.exists("temp"):
    try:
        time.sleep(1)  # Small delay to ensure all files are closed
        shutil.rmtree("temp")
    except PermissionError:
        st.warning("⚠️ Could not delete temp folder immediately. Try again later.")

