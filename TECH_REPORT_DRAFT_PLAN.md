# Technical Report Draft Plan (ACL-Style)

This file is a writing plan for the final project report.

It follows:
- Course rules in `COURSEWORK_INSTRUCTIONS.md`
- ACL long paper structure (using your `2024.acl-long.3v1.pdf` as style reference)

---

## 1) Submission Targets (Must-Have)

- **Format:** 2025 ACL long paper template
- **Deadline:** before **May 1st, 2026**
- **Filename:** `Last Name, First Name.pdf`
- **Primary graded item:** final project report PDF

---

## 2) Report Goal (One-Sentence)

Show that formality-based features can separate human-written text from machine-generated text, and evaluate this claim with clear experiments, ablations, and error analysis.

---

## 3) ACL-Style Paper Skeleton (What to Draft)

Use this section order for your draft.

## Title
- Keep it specific and technical.
- Example direction: *Formality-Based Human vs Machine Text Detection*

## Abstract (150–250 words)
- Problem
- Method
- Dataset
- Key metrics (Accuracy/F1/AUROC)
- Main takeaway

## 1. Introduction
- Why this problem matters (LLM-generated text is common)
- What your project studies
- Your main hypothesis
- 2–3 contributions (small bullet list)

## 2. Related Work
- Prior AI-text detection approaches (briefly)
- Where your approach is different (focus on simple formality features)

## 3. Method
### 3.1 Features
- Contraction rate
- Informal word rate
- Sentence length mean/std
- Punctuation ratios (e.g., exclamation, question)
- All-caps ratio

### 3.2 Model
- Logistic Regression
- Feature scaling
- Train/validation/test setup

## 4. Experimental Setup
### 4.1 Data
- Dataset choice and rationale
- Sample counts (human vs machine)
- Split details

### 4.2 Metrics
- Accuracy
- F1
- AUROC

### 4.3 Baselines
- Add simple baseline(s), for example:
  - Majority class baseline
  - Single-feature threshold baseline

## 5. Results
- Main result table (Accuracy/F1/AUROC)
- Short interpretation (what improved and why it matters)

## 6. Analysis
### 6.1 Ablation Study (required by rubric)
- Remove one feature group at a time
- Show metric drop and explain what it means

### 6.2 Error Analysis (required by rubric)
- Show 5–10 failure examples
- Group common failure patterns
- Explain likely reasons

## 7. Discussion
- What worked well
- What failed
- Practical implications

## 8. Limitations
- Dataset/domain limits
- Generalization limits
- Ethical/careful-use note

## 9. Conclusion
- 3–5 sentence summary
- Key final claim and next steps

## References
- ACL style bibliography (`.bib`)
- Include key detection papers and tools used

---

## 4) Figure/Table Plan (Concrete)

Add these visuals for clarity and excitement:

1. **Pipeline figure**
   - Data -> Feature extraction -> Logistic Regression -> Metrics

2. **Main results table**
   - Model/baseline vs Accuracy/F1/AUROC

3. **Ablation table**
   - Full model vs feature-removed variants

4. **Error analysis table**
   - Example text, predicted label, true label, error note

5. **Optional chart**
   - Feature importance or per-feature class mean differences

---

## 5) Rubric Mapping Checklist (Soundness + Excitement)

Use this before final submission.

### Soundness
- [ ] Baselines are included and justified.
- [ ] Data size and splits are clearly reported.
- [ ] Ablation study is present and interpretable.
- [ ] Error analysis includes qualitative examples.
- [ ] Metrics are aligned to task goals (Accuracy/F1/AUROC).
- [ ] Claims match evidence in tables/figures.

### Excitement
- [ ] There is a clear insight beyond “just ran a model.”
- [ ] Writing is clear and structured in ACL format.
- [ ] Figures/tables are clean and readable.
- [ ] Task framing explains why this is interesting/useful.

---

## 6) Drafting Workflow (Fast Path)

1. Write Methods + Experimental Setup first (easiest, factual).
2. Run experiments and collect final numbers.
3. Build Main Results + Ablation + Error Analysis tables.
4. Write Discussion + Limitations.
5. Write Introduction and Abstract last.
6. Final pass for ACL formatting, references, and naming rule.

---

## 7) Suggested File Outputs for Report Writing

- `report/main.tex` (or Overleaf equivalent)
- `report/custom.bib`
- `report/figures/` (all figures)
- `report/tables/` (optional source tables)
- Final export: `Last Name, First Name.pdf`

---

## 8) Optional Poster Alignment (If You Participate)

Keep poster content consistent with the report:
- Problem
- Method pipeline
- Main results table
- One ablation result
- One error-analysis insight
- Key takeaway

