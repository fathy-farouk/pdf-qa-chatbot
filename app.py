import streamlit as st
import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load API key from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Streamlit UI Setup
st.set_page_config(page_title="AI PDF Q&A Chatbot", page_icon="📄")
st.title("📄 AI PDF Q&A Chatbot")
st.markdown("Upload a PDF and ask anything about its content!")

# File uploader
uploaded_file = st.file_uploader("📤 Upload your PDF file", type=["pdf"])

# Process PDF and update FAISS index
if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        # Save file temporarily
        pdf_path = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)
        
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load and split PDF
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        
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
