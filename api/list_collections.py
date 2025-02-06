import requests

response = requests.get("http://localhost:8000/list-collections/")
print(response.json())
