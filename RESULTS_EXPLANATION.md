# Explanation of Test Results

This document explains the test-set results from the formality-based **human vs machine text** classifier (Logistic Regression on formality features).

---

## 1. Overall Performance

| Metric          | Value  |
| --------------- | ------ |
| **Accuracy**    | 86.78% |
| **F1 (binary)** | 0.8491 |
| **AUROC**       | 0.8393 |

- **Accuracy (86.78%)**: On the 121 test documents, the model predicts the correct label (human or machine) about 87% of the time.
- **F1 (0.85)**: Balance between precision and recall; the model is reasonably good at both finding machine text and not over-flagging human text.
- **AUROC (0.84)**: The model ranks a random machine text above a random human text about 84% of the time; values above 0.8 indicate useful discrimination.

**Conclusion:** Formality features alone give solid separation between human- and machine-written text on this dataset, without using a large language model.

---

## 2. Classification Report

| Class       | Precision | Recall | F1-score | Support |
| ----------- | --------- | ------ | -------- | ------- |
| **human**   | 0.80      | 0.98   | 0.88     | 61      |
| **machine** | 0.98      | 0.75   | 0.85     | 60      |

- **Human (0.80 precision, 0.98 recall)**

  - Almost all human texts are correctly identified as human (high recall).
  - Some machine texts are wrongly labeled as human, so precision is a bit lower (0.80).

- **Machine (0.98 precision, 0.75 recall)**
  - When the model says “machine,” it is almost always right (high precision).
  - It misses about 25% of machine texts (recall 0.75), so some machine texts look more “human-like” in terms of formality.

**Conclusion:** The classifier is conservative on machine text (few false positives) but misses some machine-written samples; human text is detected very reliably.

---

## 3. Feature Means (Human vs Machine)

Means are computed on the **test set** only. They show how human and machine text differ on each formality-related feature.

| Feature                  | Human  | Machine | Interpretation                                                                      |
| ------------------------ | ------ | ------- | ----------------------------------------------------------------------------------- |
| **contraction_rate**     | 0.0078 | 0.0092  | Slightly more contractions in machine text here; small difference.                  |
| **informal_word_rate**   | 0.0001 | 0.0003  | Very low for both; machine slightly higher.                                         |
| **sentence_length_mean** | 20.03  | 19.01   | Similar; humans slightly longer sentences on average.                               |
| **sentence_length_std**  | 9.81   | 8.17    | **Human text has more variation** in sentence length; machine text is more uniform. |
| **exclamation_rate**     | 0.00   | 0.0071  | Machine text uses more exclamation marks per sentence.                              |
| **question_rate**        | 0.00   | 0.0088  | Machine text uses more question marks per sentence.                                 |
| **all_caps_ratio**       | 0.0114 | 0.0032  | **Human text uses more ALL CAPS** (emphasis, informal style).                       |
| **avg_word_length**      | 5.09   | 5.06    | Nearly the same.                                                                    |
| **num_sentences**        | 7.98   | 13.65   | **Machine text has more sentences** per document (longer, more structured).         |
| **num_words**            | 148.93 | 229.83  | **Machine text is longer** in word count on average.                                |

**Main takeaways:**

1. **Length and structure:** Machine text tends to be longer (more words, more sentences) and more uniform in sentence length; human text is shorter and more variable.
2. **Informality cues:** Human text uses more ALL CAPS; machine text uses more exclamation and question marks per sentence.
3. **Formality hypothesis:** The “formality” signal is partly about **length/structure** (longer, more uniform machine text) and **punctuation/emphasis** (more caps in human, more !/? in machine), not only contractions or slang in this dataset.

---

## 4. Summary

- The classifier reaches **~87% accuracy** and **AUROC ~0.84** using only formality-style features.
- It is very good at recognizing human text (high recall) and very confident when it predicts machine (high precision), but it misses some machine text (lower recall for machine).
- Feature means show that **sentence-length variation**, **document length** (words/sentences), **ALL CAPS**, and **exclamation/question rate** are useful for telling human and machine text apart in this setup.

These results support using formality (and related stylistic features) as a simple, interpretable signal for human vs machine text detection, and give concrete numbers and feature interpretations you can use in a report or paper.
