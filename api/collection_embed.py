
import requests
import os


def test_document_processing():
    url = "http://localhost:8000/process-documents-by-collection"

    response = requests.post(url)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")


if __name__ == "__main__":
    test_document_processing()
