import requests

response = requests.delete("http://localhost:8000/clear-db/")
print(response.json())
