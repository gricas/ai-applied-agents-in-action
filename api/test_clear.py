import requests

response = requests.delete("http://localhost:8000/clear_db/")
print(response.json())
