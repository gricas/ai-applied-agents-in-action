import requests


def test_search():
    url = "http://localhost:8000/search/"
    params = {"query": "what is coffee?"}
    response = requests.get(url, params=params)
    print("Status Code:", response.status_code)
    print("\nSearch Results:")
    print(response.json())


if __name__ == "__main__":
    test_search()
