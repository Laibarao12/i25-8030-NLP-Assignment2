# DS5007 - BBC Urdu NLP Assignment 2

**Student ID:** 25i8030

This repository contains the PyTorch source for three experiments on 300 BBC Urdu articles: TF-IDF/PPMI/Skip-gram word representations, BiLSTM-CRF POS and NER tagging, and a four-block Transformer for news-topic classification.

## Contents
- `i25-8030_Assignment2_DS-X.ipynb`: main notebook (public code-only copy).
- `src/`: Urdu preprocessing, embeddings, annotations, sequence models, Transformer, report generation.
- `assets/`: analogy tests and MRR reference pairs.
- `tests/`: component-level checks.
- `results/summary.json`: aggregate model results.
- `data/`, `models/`, `embeddings/`: documentation for private course deliverables.

## Run in Google Colab
1. Start a GPU runtime and install dependencies from `requirements.txt`.
2. Obtain the coursework-authorized BBC Urdu 300-article dataset. In the notebook upload its ZIP containing `raw.txt`, `cleaned.txt`, `Metadata.json`.
3. Run cells in order. The notebook writes each `src/` module and then executes Part 1, Part 2 and Part 3.
4. Source files and test helpers can also be inspected directly in the `src/` and `tests/` folders.

## Saved results
| Metric | Score |
| --- | ---: |
| POS token accuracy | 89.17% |
| NER entity precision / recall / F1 | 90.00% / 87.80% / 88.89% |
| Transformer accuracy | 75.56% |
| BiLSTM topic accuracy | 46.67% |

NER scores are computed against automatically generated gazetteer annotations, not expert gold labels.

## Notes on data and weights
The full article text, derived CoNLL files, `.npy` matrices and `.pt` checkpoints are provided in the **private course submission**, not in this public repository. The dense PPMI array is roughly 400 MB and cannot be pushed to ordinary GitHub without LFS.

Repository: https://github.com/Laibarao12/i25-8030-NLP-Assignment2
