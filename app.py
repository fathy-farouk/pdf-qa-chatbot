import streamlit as st
import os
import shutil
import fitz  # PyMuPDF
import pytesseract
import cv2
import numpy as np
from pdf2image import convert_from_path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from PIL import Image

# Set Tesseract path (Windows users)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load API key from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Streamlit UI Setup
st.set_page_config(page_title="AI PDF Q&A Chatbot", page_icon="📄")
st.title("📄 AI PDF Q&A Chatbot")
st.markdown("Upload a PDF and ask anything about its content!")

# File uploader
uploaded_file = st.file_uploader("📤 Upload your PDF file", type=["pdf"])

# Function to extract text from scanned PDFs using OCR
def extract_text_from_scanned_pdf(pdf_path):
    text = ""
    images = convert_from_path(pdf_path)  # Convert PDF pages to images
    
    for image in images:
        # Convert PIL image to OpenCV format
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to enhance text visibility
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # Perform OCR for Arabic text
        text += pytesseract.image_to_string(thresh, lang="ara") + "\n"
    
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
