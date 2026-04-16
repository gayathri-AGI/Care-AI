import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from src.helper import download_hugging_face_embeddings

load_dotenv()

# Load API Key
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Load embeddings
embeddings = download_hugging_face_embeddings()

data_path = "data"
documents = []

print("📚 Loading documents...")

# Load all PDFs correctly
for file in os.listdir(data_path):
    if file.endswith(".pdf"):
        print(f"Loading {file}...")
        
        file_path = os.path.join(data_path, file)
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()

        # Add metadata (important)
        for doc in docs:
            doc.metadata["source"] = file

        documents.extend(docs)

print(f"✅ Total pages loaded: {len(documents)}")

print("✂ Splitting documents...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=150
)

docs = text_splitter.split_documents(documents)

print(f"✅ Total chunks created: {len(docs)}")

print("🧠 Creating embeddings and uploading to Pinecone...")

index_name = "medical-heart"

batch_size = 300  # good for 8GB RAM

vectorstore = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings
)

for i in range(0, len(docs), batch_size):
    batch = docs[i:i+batch_size]
    vectorstore.add_documents(batch)
    print(f"Uploaded {i + len(batch)} / {len(docs)}")

print("✅ All new medical books added successfully!")