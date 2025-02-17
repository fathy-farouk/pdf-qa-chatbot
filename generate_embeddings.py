from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load API key from environment variables (or manually set it)
load_dotenv()  
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Ensure you have this in your .env file

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Load the PDF file
pdf_path = r"G:\PDF files\My updated CV v.10.pdf"  # Replace with your actual file name
loader = PyMuPDFLoader(pdf_path)
documents = loader.load()

# Split into smaller chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(documents)

# Convert text chunks into vector embeddings
vector_store = FAISS.from_documents(chunks, embeddings)

# Save FAISS database locally
vector_store.save_local("faiss_index")

print("✅ Embeddings generated and saved successfully!")
