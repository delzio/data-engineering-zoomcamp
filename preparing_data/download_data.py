import requests
import os

def download_file(url, filepath):
    response = requests.get(url)

    directory = os.path.dirname(filepath)
    os.makedirs(directory, exist_ok=True)

    if response.status_code == 200:
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"File downloaded successfully to '{filepath}'")
    else:
        print(f"Failed to download file. Status Code: {response.status_code}")