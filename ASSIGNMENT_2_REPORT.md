# Assignment 2: Evaluation of TrOCR Printed and Handwritten Models

## Objective

The objective of this experiment was to evaluate the performance of two pretrained TrOCR models on printed and handwritten text datasets. The models evaluated were:

* TrOCR Base Printed (`microsoft/trocr-base-printed`)
* TrOCR Base Handwritten (`microsoft/trocr-base-handwritten`)

The evaluation was performed on two publicly available datasets:

* IAM Handwriting Dataset (handwritten text)
* MJSynth Text Recognition Dataset (printed text)

Four experiments were conducted to study both in-domain and cross-domain performance.

## Experimental Setup

| Experiment | Model             | Dataset             |
| ---------- | ----------------- | ------------------- |
| E1         | TrOCR Printed     | Printed Dataset     |
| E2         | TrOCR Printed     | Handwritten Dataset |
| E3         | TrOCR Handwritten | Printed Dataset     |
| E4         | TrOCR Handwritten | Handwritten Dataset |

Evaluation metrics used:

* Character Error Rate (CER)
* Word Error Rate (WER)
* Average Inference Time

## Results

| Model             | Dataset     | CER   | WER   | Avg Inference Time (s) |
| ----------------- | ----------- | ----- | ----- | ---------------------- |
| TrOCR Printed     | Printed     | 0.644 | 0.920 | 0.997                  |
| TrOCR Printed     | Handwritten | 0.784 | 0.990 | 2.537                  |
| TrOCR Handwritten | Printed     | 0.609 | 1.140 | 0.839                  |
| TrOCR Handwritten | Handwritten | 0.076 | 0.190 | 2.075                  |

## Analysis

The results demonstrate that model specialization significantly impacts OCR performance.

The TrOCR Handwritten model achieved the best performance on the IAM handwritten dataset with a CER of 0.076 and a WER of 0.190. Visual inspection of the predictions showed that most handwritten text lines were recognized accurately, with only minor errors involving characters, punctuation, and proper nouns.

Cross-domain evaluation showed a decline in performance when models were applied to datasets outside their intended domain. The TrOCR Printed model performed poorly on handwritten text, while the TrOCR Handwritten model also experienced reduced accuracy when evaluated on printed word images.

Qualitative analysis of the printed dataset revealed that many predictions were semantically correct but differed in capitalization or formatting. Examples include:

* "slinking" → "SLINKING"
* "Impeaching" → "IMPEACHING"
* "underpays" → "UNDERPAYS"

This indicates that some reported errors were caused by case-sensitive evaluation rather than complete recognition failure.

The experiment confirms that pretrained TrOCR models are optimized for the domains on which they were trained and perform best when evaluated on matching data distributions.

## Conclusion

The TrOCR Handwritten model achieved the strongest performance on handwritten text and demonstrated effective recognition of long text sequences. The results support the use of domain-specific OCR models and highlight the importance of selecting a model that matches the target document type. Future work may include normalization of predictions before evaluation and testing on larger benchmark datasets.
