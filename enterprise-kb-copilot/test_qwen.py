import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv(".env")
api_key = os.getenv("EMBEDDING_API_KEY")
base_url = os.getenv("EMBEDDING_BASE_URL")
model = os.getenv("EMBEDDING_MODEL_NAME")

print(f"Key: {api_key}, URL: {base_url}, Model: {model}")

print("Initializing embeddings...")
embeddings = OpenAIEmbeddings(
    api_key=api_key,
    base_url=base_url,
    model=model,
    chunk_size=25,
)

print("Calling embed_documents...")
texts = ["第一首古诗", "第二首古诗"]
res = embeddings.embed_documents(texts)
print(f"Success! Embedded {len(res)} items.")
