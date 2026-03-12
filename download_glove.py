"""Download GloVe embeddings from Stanford NLP."""
import urllib.request
import zipfile
import os

GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
DATA_DIR = "data"
TARGET_FILE = "glove.6B.50d.txt"


def download_glove():
    os.makedirs(DATA_DIR, exist_ok=True)

    target_path = os.path.join(DATA_DIR, TARGET_FILE)
    zip_path = os.path.join(DATA_DIR, "glove.6B.zip")

    if os.path.exists(target_path):
        print("GloVe embeddings already present.")
        return

    print("Downloading GloVe embeddings (~862MB)...")
    urllib.request.urlretrieve(GLOVE_URL, zip_path)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extract(TARGET_FILE, DATA_DIR)

    os.remove(zip_path)
    print(f"Done! Saved to {target_path}")


if __name__ == "__main__":
    download_glove()
