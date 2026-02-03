# Formality-based Human vs Machine Text Detection

NLP project: detect human-written vs machine-generated text using **formality features** (contractions, informal words, sentence length stats, punctuation, capitalization).

## Setup

```bash
cd CSE188NLP_Project
pip install -r requirements.txt
```

Download NLTK data (optional; used if you switch to NLTK sentence tokenizer later):

```bash
python -c "import nltk; nltk.download('punkt')"
```

## Run

From the project root:

```bash
python run_train.py
```

This will:

1. Load **HC3** (Hello-SimpleAI/HC3) — human vs ChatGPT answers — subset `finance`, up to 1500 samples per class.
2. Extract formality features (contraction rate, informal word rate, sentence length mean/std, exclamation/question rate, all-caps ratio, etc.).
3. Train a **Logistic Regression** classifier (with standardized features).
4. Print test **accuracy**, **F1**, **AUROC**, classification report, and **feature means** (human vs machine) on the test set.

## Project layout

- `src/formality_features.py` — formality feature extraction.
- `src/data_loader.py` — load HC3 (human vs ChatGPT) and train/val/test split.
- `src/train.py` — feature extraction, scaling, training, evaluation.
- `run_train.py` — entry point to run the pipeline.
- `requirements.txt` — dependencies.

## Data

Uses **ahmadreza13/human-vs-Ai-generated-dataset** by default (balanced human vs AI; 400 per class = 800 total). The ziq dataset is heavily imbalanced (1375 human, 3 machine) and not recommended. In `run_train.py`:

```python
run_pipeline(max_samples=400, dataset_name="ahmadreza13")  # default, ~87% test accuracy
run_pipeline(max_samples=1000, dataset_name="ahmadreza13")  # larger run
```

## Hypothesis

In informal domains, machine-generated text tends to be more formally consistent (fewer contractions, less slang, more uniform sentence structure). Formality features can separate human vs machine text; the script prints feature means per class to check this.

## Report

For your report: use the printed **feature means** (human vs machine) as an analysis table; use **accuracy/F1/AUROC** as main results; add ablations (e.g. drop one feature at a time) if you extend the code.
