# from langchain.document_loaders import PyMuPDFLoader

# # Load the PDF file
# pdf_path = r"G:\PDF files\My updated CV v.10.pdf"  # Replace with your actual file name
# loader = PyMuPDFLoader(pdf_path)
# documents = loader.load()

# # Print extracted text
# for doc in documents[:5]:  # Print first 5 extracted pages
#     print(f"Page {doc.metadata['page']}: {doc.page_content}\n")


from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load the PDF file
pdf_path = r"G:\PDF files\My updated CV v.10.pdf"  # Replace with actual file name
loader = PyMuPDFLoader(pdf_path)
documents = loader.load()

# Split into smaller chunks (each around 1000 characters, with some overlap)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(documents)

# Print first 3 chunks as a sample
for i, chunk in enumerate(chunks[:3]):
    print(f"Chunk {i+1}:\n{chunk.page_content}\n{'-'*50}")
