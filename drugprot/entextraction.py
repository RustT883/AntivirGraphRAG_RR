from langchain_ollama import ChatOllama
import os
from langchain_core.messages import AIMessage

#os.environ['http_proxy'] = 'socks://127.0.0.1:10808/'
#os.environ['https_proxy'] = 'socks://127.0.0.1:10808/'

llm = ChatOllama(
    base_url = "http://localhost:11434",
    model="qwen2.5:latest", 
    timeout=600, 
    temperature = 0.0,
    disable_streaming = True,
    num_ctx=4096
)

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)

print()
