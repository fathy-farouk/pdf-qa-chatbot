from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OpenAI API key not found! Make sure .env is set up correctly.")

# Load the FAISS index
vector_store = FAISS.load_local(
    "faiss_index", 
    OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY), 
    allow_dangerous_deserialization=True
)


# Initialize GPT-4 model
llm = ChatOpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY)

# Create a retriever to fetch relevant chunks
retriever = vector_store.as_retriever()

# Define RAG-based QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Function to get answers from the PDF
def ask_pdf(query):
    response = qa_chain(query)
    answer = response["result"]
    sources = response["source_documents"]

    print("\n🔍 **Answer:**")
    print(answer)

    print("\n📌 **Sources:**")
    for i, source in enumerate(sources[:3]):
        print(f"📄 Chunk {i+1}: {source.page_content[:200]}...")  # Show first 200 chars of each source

# Run a test query
if __name__ == "__main__":
    query = input("Ask a question about the PDF: ")
    ask_pdf(query)
