import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

# ==========================
# 1) Load your CSV manually
# ==========================
file_path = "data/translated_articles2.csv"
df = pd.read_csv(file_path)
print("=================================")
print(f"Loaded {len(df)} rows from {file_path}")
print("=================================")

# Combine title + article_content into one field
df["combined_text"] = df["title"].fillna("") + " " + df["translated_content"].fillna("")

# Convert to list of dicts (metadata)
documents = []
for _, row in df.iterrows():
    documents.append({
        "page_content": row["combined_text"],
        "metadata": {
            "id": row["id"],
            "slug": row["slug"],
            "title": row["title"]
        }
    })

print("=================================")
print('file loaded successfully')
print(f"Loaded {len(documents)} documents.")
print("=================================")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,     # MiniLM can handle small to medium chunks
    chunk_overlap=200
)

print("=================================")
print("Text splitter initialized.")
print(f"Before splitting: {len(documents)} documents")
print("=================================")


docs = text_splitter.create_documents(
    [doc["page_content"] for doc in documents],
    metadatas=[doc["metadata"] for doc in documents]
)

print(f"After splitting: {len(docs)} chunks")

# ==========================
# 3) Setup embeddings
# ==========================
HF_TOKEN = os.getenv("HAGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not set.")

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction",
    huggingfacehub_api_token=HF_TOKEN
)

print("=================================")
print("Embeddings model initialized.")

# ==========================
# 4) Build FAISS vectorstore
# ==========================
db = FAISS.from_documents(docs, embeddings)
print("=================================")
print("FAISS DB built with", len(docs), "chunks.")

# ==========================
# 5) Save locally
# ==========================
db.save_local("faiss_index2")
print("FAISS index saved locally.")
