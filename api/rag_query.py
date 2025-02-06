import requests


def test_search():
    url = "http://localhost:8000/rag-query/"
    params = {"query": "What kind of beans should I use?"}
    response = requests.post(url, params=params)
    print("Status Code:", response.status_code)
    print("\nSearch Results:")
    print(response.json())


if __name__ == "__main__":
    test_search()
