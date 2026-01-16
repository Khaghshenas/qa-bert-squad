from datasets import load_dataset

def download_squad():
    dataset = load_dataset("squad")
    dataset.save_to_disk("../data/squad")
    print("SQuAD dataset downloaded and saved.")

if __name__ == "__main__":
    download_squad()