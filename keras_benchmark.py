import os
import time
import pandas as pd
import keras_ocr

print("Loading Keras-OCR...")

pipeline = keras_ocr.pipeline.Pipeline()

dataset_path = "dataset"

results = []

for image_file in os.listdir(dataset_path):

    image_path = os.path.join(
        dataset_path,
        image_file
    )

    print(f"\nProcessing {image_file}")

    start = time.time()

    prediction_groups = pipeline.recognize(
        [image_path]
    )

    runtime = round(
        time.time() - start,
        3
    )

    words = []

    for prediction in prediction_groups[0]:

        words.append(
            prediction[0]
        )

    extracted_text = " ".join(words)

    results.append({
        "image": image_file,
        "framework": "KerasOCR",
        "runtime_sec": runtime,
        "confidence": "N/A",
        "text": extracted_text
    })

df = pd.DataFrame(results)

os.makedirs(
    "results",
    exist_ok=True
)

df.to_csv(
    "results/keras_results.csv",
    index=False
)

print("\nDone")
print(df)