import os
import time
import pandas as pd

from PIL import Image
from jiwer import wer, cer

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel
)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate_dataset(
    model_name,
    dataset_dir,
    gt_csv,
    output_csv
):
    print(f"\nLoading {model_name}...")

    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    gt = pd.read_csv(gt_csv)

    rows = []

    total_time = 0

    for _, row in gt.iterrows():

        image_path = os.path.join(
            dataset_dir,
            row["filename"]
        )

        image = Image.open(image_path).convert("RGB")

        start = time.time()

        pixel_values = processor(
            images=image,
            return_tensors="pt"
        ).pixel_values

        generated_ids = model.generate(pixel_values)

        prediction = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        elapsed = time.time() - start

        total_time += elapsed

        rows.append({
            "filename": row["filename"],
            "ground_truth": str(row["text"]),
            "prediction": prediction
        })

    results = pd.DataFrame(rows)

    results.to_csv(output_csv, index=False)

    gt_text = " ".join(results["ground_truth"])
    pred_text = " ".join(results["prediction"])

    metrics = {
        "model": model_name,
        "dataset": dataset_dir,
        "CER": cer(gt_text, pred_text),
        "WER": wer(gt_text, pred_text),
        "Avg_Inference_Time": total_time / len(results)
    }

    return metrics


all_metrics = []

experiments = [

    (
        "microsoft/trocr-base-printed",
        "datasets/printed",
        "datasets/printed/ground_truth.csv",
        "results/e1_printed_on_printed.csv"
    ),

    (
        "microsoft/trocr-base-printed",
        "datasets/handwritten",
        "datasets/handwritten/ground_truth.csv",
        "results/e2_printed_on_handwritten.csv"
    ),

    (
        "microsoft/trocr-base-handwritten",
        "datasets/printed",
        "datasets/printed/ground_truth.csv",
        "results/e3_handwritten_on_printed.csv"
    ),

    (
        "microsoft/trocr-base-handwritten",
        "datasets/handwritten",
        "datasets/handwritten/ground_truth.csv",
        "results/e4_handwritten_on_handwritten.csv"
    )

]

for exp in experiments:

    metrics = evaluate_dataset(*exp)

    all_metrics.append(metrics)

summary = pd.DataFrame(all_metrics)

summary.to_csv(
    "results/trocr_summary.csv",
    index=False
)

print("\nDone.")
print(summary)