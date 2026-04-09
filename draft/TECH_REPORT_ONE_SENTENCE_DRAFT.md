# Tech Report One-Sentence Draft

## Methods and Libraries Used (Concise)

- **Classifier:** `LogisticRegression` from **scikit-learn** (`sklearn.linear_model`)
- **Feature scaling:** `StandardScaler` from **scikit-learn** (`sklearn.preprocessing`)
- **Evaluation metrics:** `accuracy_score`, `f1_score`, `roc_auc_score`, `classification_report` from **scikit-learn** (`sklearn.metrics`)
- **Data split:** stratified train/validation/test using `train_test_split` from **scikit-learn** (`sklearn.model_selection`)
- **Feature extraction method (custom, rule-based):**
  - `contraction_rate`
  - `informal_word_rate`
  - `sentence_length_mean`
  - `sentence_length_std`
  - `exclamation_rate`
  - `question_rate`
  - `all_caps_ratio`
  - `avg_word_length`
  - `num_sentences`
  - `num_words`
- **Dataset loading:** **Hugging Face Datasets** (`datasets.load_dataset`), mainly `ahmadreza13/human-vs-Ai-generated-dataset` (with optional `ziq/ai-generated-text-classification`)
- **Core numeric library:** **NumPy**
- **Pipeline entry point:** `run_train.py` -> `src/train.py` (`run_pipeline`)

## Quick Project Summary (Edit Freely)

This project builds a lightweight NLP classifier to detect whether text is human-written or machine-generated using formality features (like contractions, slang rate, sentence style, punctuation, and capitalization).  
It is useful because AI-generated text is now common, so simple and interpretable detection tools can help with transparency, education, and content analysis.  
Our baseline pipeline uses feature extraction + logistic regression and reports Accuracy, F1, and AUROC, with a default run around the high-80% range in this project setup.  
These results matter because they show that even simple stylistic signals can provide meaningful separation between human and machine text.  
The work is directly related to NLP and LLMs since it studies language patterns in text produced by humans versus modern language models.  

External sources used in this project (data + open-source tools):
- Dataset: **ahmadreza13/human-vs-Ai-generated-dataset** (Hugging Face) by user **ahmadreza13**  
  Link: https://huggingface.co/datasets/ahmadreza13/human-vs-Ai-generated-dataset
- Dataset (alternative, noted as imbalanced): **ziq/ai-generated-text-classification** (Hugging Face) by user **ziq**  
  Link: https://huggingface.co/datasets/ziq/ai-generated-text-classification
- Library: **Hugging Face Datasets** (for dataset loading/streaming), by Hugging Face contributors  
  Link: https://github.com/huggingface/datasets
- Library: **scikit-learn** (Logistic Regression, metrics, scaling), by Pedregosa et al. and open-source contributors  
  Link: https://github.com/scikit-learn/scikit-learn
- Library: **NumPy** (feature arrays and numeric operations), by NumPy contributors  
  Link: https://github.com/numpy/numpy

---

## How to Use This File

- Add **one sentence** under each section first.
- Keep it short now.
- I can expand each sentence into a full ACL-style section later.

---

## Title

Description: Project name (clear, specific, and technical).

Sentence: Formality-Based Human vs. Machine Text Detection

## Abstract

Description: One-sentence overview of problem, method, data, and main result.

Sentence: using a dataset by ahmadreza13 and others on github, we use this dataset to then run our lightweight NLP classifier made with tools from scikit-learn to detect whether the text is human or AI generated, using human langugage patterns vs generated ones.

## 1. Introduction

Description: Why this problem matters and what this project tries to solve.

Sentence: In this day and age, many online text contents can easily be seen as human written, but in the age of AI, this may not be true. Our lightweight NLP classifier helps distinguish text between ai and human.

## 2. Related Work

Description: Prior research/tools related to AI-text detection and how this project differs.

Sentence: not sure what to add, in terms of my related work, ive worked on using AI to analyze Resumes, this project differs as we use lots of data 

## 3. Method

Description: What methods are used overall in this project.

Sentence: We use datasets related to hugging face, and tools from scikit-learn for the NLP classifier

## 3.1 Features

Description: What text features are extracted and why they are useful.

Sentence: we extract certain features from text such as informalities, question rate, avg word length, and so on, to compare against ai generated answers.

## 3.2 Model

Description: What model is used and how it is trained.

Sentence:  we use Logisitc regression, standard scaler, and accuracy score, from scikit learn

## 4. Experimental Setup

Description: How the experiments are designed and run.

Sentence: the pipeline is, we get large amounts of text data from sources such as reddit, twitter, etc, and run them through the pipeline where it then goes through text extraction.

## 4.1 Data

Description: What dataset is used, sample sizes, and split setup.

Sentence: datasets were used by (list users we used dataset from), sample sizes were (answer) and data split was using train_test_split from ski-kit learn

## 4.2 Metrics

Description: Which evaluation metrics are used and why.

Sentence: Evaluation metrics used were `accuracy_score`, `f1_score`, `roc_auc_score`, `classification_report` from scikit-learn

## 4.3 Baselines

Description: What baseline methods are compared against.

Sentence: We compare our logistic-regression model against a majority-class baseline that always predicts the most common training label (human or machine), then evaluate both with Accuracy, F1, and AUROC.

## 5. Results

Description: Main quantitative outcomes and key performance numbers.

Sentence: ________________________________________________

## 6. Analysis

Description: What deeper analysis is done beyond headline metrics.

Sentence: ________________________________________________

## 6.1 Ablation Study

Description: What happens when specific features/components are removed.

Sentence: We run one-group-at-a-time ablations (e.g., remove contraction, punctuation, sentence statistics, or capitalization features) and report metric drops to measure each group’s contribution.

## 6.2 Error Analysis

Description: Common failure cases and why the model gets them wrong.

Sentence: We export false-positive and false-negative examples with model confidence and short text snippets, then summarize recurring failure patterns in style overlap and ambiguous language.

## 7. Discussion

Description: What the results mean in practice and what insights were learned.

Sentence: ________________________________________________

## 8. Limitations

Description: Current weaknesses, constraints, and risks of this approach.

Sentence: ________________________________________________

## 9. Conclusion

Description: Final takeaway and the next step for future work.

Sentence: ________________________________________________

## References

Description: External papers, datasets, libraries, and tools cited in the report.

Sentence: ________________________________________________

