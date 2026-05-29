from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="qwen2.5:7b")

resp = llm.invoke("一句话解释什么是RAG")

print(resp)