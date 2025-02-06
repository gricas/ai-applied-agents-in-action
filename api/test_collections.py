import requests

response = requests.get("http://localhost:8000/list_collections/")
print(response.json())
