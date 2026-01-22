from datasets import load_dataset
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def download_squad():
    dataset = load_dataset("squad")
    dataset.save_to_disk(os.path.join(BASE_DIR, "../data/squad"))
    print("SQuAD dataset downloaded and saved.")

if __name__ == "__main__":
    download_squad()