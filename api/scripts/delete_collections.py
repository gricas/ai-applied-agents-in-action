import os
import sys

import chromadb

chroma_client = chromadb.PersistentClient(path="./chroma_db")


def list_collections():
    """
    Retrieve and return information about all collections in the ChromaDB.
    """
    collection_names = chroma_client.list_collections()
    collections_info = []
    for name in collection_names:
        collection = chroma_client.get_collection(name)
        collections_info.append({"name": name, "count": collection.count()})
    return collections_info


def delete_collections():
    """
    Delete all collections in the ChromaDB.
    """
    collection_names = chroma_client.list_collections()
    for name in collection_names:
        chroma_client.delete_collection(name=name)
    return len(collection_names)


def main():
    try:
        collections_info = list_collections()

        if not collections_info:
            print("No collections found in the ChromaDB.")
            return

        print("Current collections in the ChromaDB:")
        for info in collections_info:
            print(f" - {info['name']}: {info['count']} documents")
        print(f"Total collections: {len(collections_info)}\n")

        user_input = (
            input("Do you want to delete all these collections? (y/n): ")
            .strip()
            .lower()
        )
        if user_input == "y":
            num_deleted = delete_collections()
            print(f"Successfully deleted {num_deleted} collections.")
        else:
            print("No collections were deleted.")

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
