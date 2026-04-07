import requests
response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "llama3.2",
        "messages": [{"role": "user", "content": "Hi!"}],
        "stream": False
    }
)
print(response.json()["message"]["content"])
