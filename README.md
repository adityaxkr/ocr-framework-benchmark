\# OCR Framework Benchmarking



\## Overview



This project benchmarks multiple OCR (Optical Character Recognition) frameworks across different image conditions.



The objective is to evaluate OCR performance on:



\* Clean printed text

\* Blurry text

\* Handwritten text

\* Non-English text

\* Rotated text



\## OCR Frameworks Evaluated



1\. Tesseract OCR

2\. EasyOCR

3\. PaddleOCR

4\. TrOCR

5\. Keras-OCR (additional evaluation)



\## Project Structure



```

ocr-framework-benchmark/

│

├── dataset/

│   ├── clean.jpg

│   ├── blurry.jpg

│   ├── handwritten.jpeg

│   ├── non\_english.jpg

│   └── rotated.jpg

│

├── results/

│   ├── results.csv

│   └── summary.csv

│

├── benchmark.py

├── requirements.txt

├── README.md

└── .gitignore

```



\## Installation



\### Create Virtual Environment



```bash

python -m venv venv

venv\\Scripts\\activate

```



\### Install Dependencies



```bash

pip install -r requirements.txt

```



\### Install Tesseract OCR



Download and install Tesseract OCR:



https://github.com/tesseract-ocr/tesseract



Update the path in `benchmark.py` if necessary.



\## Running the Benchmark



```bash

python benchmark.py

```



Results will be saved to:



```

results/results.csv

results/summary.csv

```



\## Evaluation Methodology



The OCR frameworks were evaluated using:



\* OCR Accuracy

\* Runtime Performance

\* Robustness to Blur

\* Handwriting Recognition Capability

\* Rotated Text Recognition

\* Language Support

\* Ease of Deployment



\## Results Summary



\### Runtime



| Framework | Approx Runtime  |

| --------- | --------------- |

| Tesseract | Fastest (\~0.3s) |

| PaddleOCR | \~1s             |

| TrOCR     | \~9s             |

| EasyOCR   | \~12s            |



\### Key Findings



\#### Tesseract



Strengths:



\* Fastest OCR framework

\* Excellent for clean printed text



Weaknesses:



\* Poor handwritten recognition

\* Failed on rotated text



\#### EasyOCR



Strengths:



\* Good printed text recognition

\* Confidence scores available



Weaknesses:



\* Slow CPU inference

\* Poor handwritten performance



\#### PaddleOCR



Strengths:



\* Best overall accuracy

\* Best rotated text handling

\* Best handwritten recognition

\* Strong confidence estimates



Weaknesses:



\* Larger dependency footprint



\#### TrOCR



Strengths:



\* Transformer-based OCR architecture

\* Suitable for text-line recognition tasks



Weaknesses:



\* Poor performance on full-document OCR

\* Slow inference



\## Final Recommendation



| Use Case                | Recommended Framework |

| ----------------------- | --------------------- |

| Clean Printed Documents | Tesseract             |

| Fast CPU OCR            | Tesseract             |

| Handwritten Text        | PaddleOCR             |

| Rotated Text            | PaddleOCR             |

| General Purpose OCR     | PaddleOCR             |

| Production Deployment   | PaddleOCR             |

| Research Experiments    | TrOCR                 |



\## Conclusion



Among all evaluated frameworks, PaddleOCR provided the best balance between accuracy, robustness, and runtime performance. It consistently achieved the strongest results across multiple challenging OCR scenarios and is recommended as the primary OCR framework for production applications.



