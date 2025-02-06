import requests
import os


def test_document_processing():
    url = "http://localhost:8000/process_documents"

    txt_files = []
    for filename in os.listdir("docs"):
        if filename.endswith(".txt"):
            file_path = os.path.join("docs", filename)
            txt_files.append(("files", (filename, open(file_path, "rb"), "text/plain")))

    if not txt_files:
        print("No .txt files found in docs directory")
        return

    response = requests.post(url, files=txt_files)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

    for _, (_, file_obj, _) in txt_files:
        file_obj.close()


if __name__ == "__main__":
    test_document_processing()
