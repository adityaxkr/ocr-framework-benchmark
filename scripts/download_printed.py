from datasets import load_dataset
import pandas as pd
import os

os.makedirs("datasets/printed", exist_ok=True)

dataset = load_dataset(
    "priyank-m/MJSynth_text_recognition",
    split="test",
    streaming=True
)

records = []

for idx, sample in enumerate(dataset):

    image = sample["image"]
    text = sample["label"]

    filename = f"p_{idx}.png"

    image.save(f"datasets/printed/{filename}")

    records.append({
        "filename": filename,
        "text": text
    })

    if idx == 49:   # save 50 samples
        break

pd.DataFrame(records).to_csv(
    "datasets/printed/ground_truth.csv",
    index=False
)

print("Saved 50 printed samples")