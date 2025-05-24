import json

import requests

BASE_URL = "http://localhost:8001"

def test_classification():
    print("\nTesting Classification Endpoint:")
    data = {
        "prompt": "Write a function to calculate the fibonacci sequence",
        "possible_categories": ["reasoning", "function-calling", "text-to-text"],
        "multi_label": False
    }

    response = requests.post(f"{BASE_URL}/classify", json=data)
    print(f"Status Code: {response.status_code}")
    print("Response:")
    print(json.dumps(response.json(), indent=2))

def test_compression():
    print("\nTesting Compression Endpoint:")
    data = {
        "text": "This is a test text that needs to be compressed. It contains multiple sentences and should be reduced to a smaller size while maintaining the main meaning.",
        "target_token": 10
    }

    response = requests.post(f"{BASE_URL}/compress", json=data)
    print(f"Status Code: {response.status_code}")
    print("Response:")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    print("Starting Python Service Tests...")
    test_classification()
    test_compression()
