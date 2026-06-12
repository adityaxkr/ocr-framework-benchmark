
import os
import time
import pandas as pd
from PIL import Image
import pytesseract
import easyocr
from paddleocr import PaddleOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

# ==========================================================
# TESSERACT CONFIG
# ==========================================================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# ==========================================================
# LOAD OCR MODELS
# ==========================================================

print("Loading EasyOCR...")
easy_reader = easyocr.Reader(['en'])

print("Loading PaddleOCR...")
paddle_ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en'
)

print("Loading TrOCR Printed Model...")
trocr_printed_processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-base-printed"
)

trocr_printed_model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-base-printed"
)

trocr_printed_model.eval()

print("Loading TrOCR Handwritten Model...")
trocr_hand_processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-base-handwritten"
)

trocr_hand_model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-base-handwritten"
)

trocr_hand_model.eval()

print("\nAll OCR Models Loaded Successfully\n")

# ==========================================================
# DATASET
# ==========================================================

dataset_path = "dataset"
results = []

# ==========================================================
# PROCESS IMAGES
# ==========================================================

for image_file in os.listdir(dataset_path):

    image_path = os.path.join(dataset_path, image_file)

    print("\n" + "=" * 70)
    print(f"Processing: {image_file}")
    print("=" * 70)

    # ======================================================
    # TESSERACT
    # ======================================================

    try:

        start = time.time()

        text = pytesseract.image_to_string(
            Image.open(image_path)
        )

        runtime = round(time.time() - start, 3)

        results.append({
            "image": image_file,
            "framework": "Tesseract",
            "runtime_sec": runtime,
            "confidence": "N/A",
            "text": text.strip()
        })

    except Exception as e:

        results.append({
            "image": image_file,
            "framework": "Tesseract",
            "runtime_sec": "ERROR",
            "confidence": "N/A",
            "text": str(e)
        })

    # ======================================================
    # EASYOCR
    # ======================================================

    try:

        start = time.time()

        output = easy_reader.readtext(image_path)

        runtime = round(time.time() - start, 3)

        extracted_text = " ".join(
            [item[1] for item in output]
        )

        confidence = round(
            sum(item[2] for item in output) /
            max(len(output), 1),
            4
        )

        results.append({
            "image": image_file,
            "framework": "EasyOCR",
            "runtime_sec": runtime,
            "confidence": confidence,
            "text": extracted_text
        })

    except Exception as e:

        results.append({
            "image": image_file,
            "framework": "EasyOCR",
            "runtime_sec": "ERROR",
            "confidence": "N/A",
            "text": str(e)
        })

    # ======================================================
    # PADDLEOCR
    # ======================================================

    try:

        start = time.time()

        result = paddle_ocr.ocr(
            image_path,
            cls=True
        )

        runtime = round(
            time.time() - start,
            3
        )

        texts = []
        confidences = []

        if result and result[0]:

            for line in result[0]:

                texts.append(
                    line[1][0]
                )

                confidences.append(
                    line[1][1]
                )

        extracted_text = " ".join(texts)

        avg_confidence = round(
            sum(confidences) / len(confidences),
            4
        ) if confidences else 0

        results.append({
            "image": image_file,
            "framework": "PaddleOCR",
            "runtime_sec": runtime,
            "confidence": avg_confidence,
            "text": extracted_text
        })

    except Exception as e:

        results.append({
            "image": image_file,
            "framework": "PaddleOCR",
            "runtime_sec": "ERROR",
            "confidence": "N/A",
            "text": str(e)
        })

    # ======================================================
    # TrOCR
    # ======================================================

    try:

        print(f"Running TrOCR on {image_file}")

        start = time.time()

        image = Image.open(
            image_path
        ).convert("RGB")

        if "handwritten" in image_file.lower():

            processor = trocr_hand_processor
            model = trocr_hand_model

        else:

            processor = trocr_printed_processor
            model = trocr_printed_model

        pixel_values = processor(
            images=image,
            return_tensors="pt"
        ).pixel_values

        with torch.no_grad():

            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=100
            )

        extracted_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        runtime = round(
            time.time() - start,
            3
        )

        print("TrOCR Output:", extracted_text)

        results.append({
            "image": image_file,
            "framework": "TrOCR",
            "runtime_sec": runtime,
            "confidence": "N/A",
            "text": extracted_text
        })

    except Exception as e:

        print(f"TrOCR Error on {image_file}: {e}")

        results.append({
            "image": image_file,
            "framework": "TrOCR",
            "runtime_sec": "ERROR",
            "confidence": "N/A",
            "text": str(e)
        })

# ==========================================================
# SAVE RESULTS
# ==========================================================

os.makedirs("results", exist_ok=True)

df = pd.DataFrame(results)

df.to_csv(
    "results/results.csv",
    index=False
)

summary = df[
    [
        "image",
        "framework",
        "runtime_sec",
        "confidence"
    ]
]

summary.to_csv(
    "results/summary.csv",
    index=False
)

print("\n" + "=" * 70)
print("BENCHMARK COMPLETE")
print("=" * 70)

print(df)

print("\nResults saved to results/results.csv")
print("Summary saved to results/summary.csv")

