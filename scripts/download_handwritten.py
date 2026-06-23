from datasets import load_dataset
from PIL import Image
import pandas as pd
import os

os.makedirs("datasets/handwritten", exist_ok=True)

dataset = load_dataset("Teklia/IAM-line", split="test")

records = []

for idx in range(50):
    sample = dataset[idx]

    image = sample["image"]
    text = sample["text"]

    filename = f"h_{idx}.png"

    image.save(f"datasets/handwritten/{filename}")

    records.append({
        "filename": filename,
        "text": text
    })

pd.DataFrame(records).to_csv(
    "datasets/handwritten/ground_truth.csv",
    index=False
)

print("Saved 50 handwritten samples")