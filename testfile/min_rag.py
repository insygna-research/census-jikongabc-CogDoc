from langchain_ollama import OllamaLLM
import chromadb
from sentence_transformers import SentenceTransformer

llm = OllamaLLM(model="qwen2.5:7b")

encoder = SentenceTransformer(
    "BAAI/bge-small-zh-v1.5"
)

client = chromadb.Client()

collection = client.create_collection("demo")


docs = [
    "RAG是一种检索增强生成技术",
    "LangChain是一个LLM应用开发框架",
    "ChromaDB是一个向量数据库"
]

embeddings = encoder.encode(docs).tolist()

collection.add(
    documents=docs,
    embeddings=embeddings,
    ids=["1", "2", "3"]
)

query = "什么是RAG"

query_embedding = encoder.encode([query]).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=2
)

context = "\n".join(results["documents"][0])

prompt = f"""
请根据以下内容回答问题：

{context}

问题：
{query}
"""

response = llm.invoke(prompt)

print(context)

print("-------------")

print(response)


