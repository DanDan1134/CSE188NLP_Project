"""
Train formality-based classifier: extract features, train Logistic Regression, evaluate.
"""

from typing import Optional, Dict, List, Tuple
import os
import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler

from .formality_features import extract_all_features, feature_names
from .data_loader import load_hc3, train_val_test_split


def texts_to_feature_matrix(texts: list) -> np.ndarray:
    """Convert list of documents to (n_samples, n_features) array."""
    names = feature_names()
    rows = []
    for t in texts:
        feats = extract_all_features(t)
        rows.append([feats[n] for n in names])
    return np.array(rows, dtype=np.float64)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """Compute core binary classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auroc = roc_auc_score(y_true, y_proba)
    except Exception:
        auroc = 0.5
    return {"accuracy": acc, "f1": f1, "auroc": auroc}


def _train_and_eval_lr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> Dict[str, float]:
    """Train a logistic regression model and evaluate on test data."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    y_proba = clf.predict_proba(X_test_s)[:, 1]
    return _compute_metrics(y_test, y_pred, y_proba)


def _run_ablation_study(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    names: List[str],
    seed: int,
) -> Dict[str, Dict[str, float]]:
    """Run feature-group ablation by removing one group at a time."""
    groups = {
        "full_model": [],
        "minus_contraction": ["contraction_rate"],
        "minus_informal": ["informal_word_rate"],
        "minus_sentence_stats": ["sentence_length_mean", "sentence_length_std", "num_sentences", "num_words"],
        "minus_punctuation": ["exclamation_rate", "question_rate"],
        "minus_caps": ["all_caps_ratio"],
        "minus_word_length": ["avg_word_length"],
    }

    name_to_idx = {n: i for i, n in enumerate(names)}
    results: Dict[str, Dict[str, float]] = {}

    for setting, drop_names in groups.items():
        keep_indices = [i for i, n in enumerate(names) if n not in drop_names]
        X_train_sel = X_train_raw[:, keep_indices]
        X_test_sel = X_test_raw[:, keep_indices]
        results[setting] = _train_and_eval_lr(X_train_sel, y_train, X_test_sel, y_test, seed)

    return results


def _write_error_analysis(
    out_path: str,
    texts: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    max_rows: int = 10,
) -> None:
    """Write a compact error-analysis CSV with false positives and false negatives."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows: List[Tuple[str, int, int, float, str]] = []

    # False positives: human(0) predicted as machine(1)
    fp_idx = np.where((y_true == 0) & (y_pred == 1))[0][:max_rows]
    for i in fp_idx:
        rows.append(("false_positive", int(y_true[i]), int(y_pred[i]), float(y_proba[i]), texts[i][:300]))

    # False negatives: machine(1) predicted as human(0)
    fn_idx = np.where((y_true == 1) & (y_pred == 0))[0][:max_rows]
    for i in fn_idx:
        rows.append(("false_negative", int(y_true[i]), int(y_pred[i]), float(y_proba[i]), texts[i][:300]))

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["error_type", "true_label", "pred_label", "pred_prob_machine", "text_snippet"])
        writer.writerows(rows)


def run_pipeline(
    subset: str = "finance",
    max_samples: Optional[int] = 700,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    dataset_name: str = "ziq",
):
    """Load data, extract formality features, train LR, evaluate."""
    print("Loading data...")
    texts, labels = load_hc3(
        subset=subset,
        max_samples_per_class=max_samples,
        seed=seed,
        dataset_name=dataset_name,
    )
    print(f"  Total: {len(texts)} documents ({np.sum(labels==0)} human, {np.sum(labels==1)} machine)")

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
        texts, labels, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
    )
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    print("Extracting formality features...")
    X_train_raw = texts_to_feature_matrix(X_train)
    X_val_f = texts_to_feature_matrix(X_val)
    X_test_raw = texts_to_feature_matrix(X_test)

    scaler = StandardScaler()
    X_train_f = scaler.fit_transform(X_train_raw)
    X_val_f = scaler.transform(X_val_f)
    X_test_f = scaler.transform(X_test_raw)

    print("Training Logistic Regression...")
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X_train_f, y_train)

    # Baseline: always predict the majority class from training labels
    majority_class = int(np.mean(y_train) >= 0.5)
    majority_rate = float(np.mean(y_train))

    # Validation
    y_val_pred = clf.predict(X_val_f)
    print(f"\nValidation: accuracy={accuracy_score(y_val, y_val_pred):.4f}, F1={f1_score(y_val, y_val_pred):.4f}")

    # Test
    y_base_pred = np.full_like(y_test, majority_class)
    y_base_proba = np.full(shape=len(y_test), fill_value=majority_rate, dtype=np.float64)
    base_acc = accuracy_score(y_test, y_base_pred)
    base_f1 = f1_score(y_test, y_base_pred)
    try:
        base_auroc = roc_auc_score(y_test, y_base_proba)
    except Exception:
        base_auroc = 0.5

    y_test_pred = clf.predict(X_test_f)
    y_test_proba = clf.predict_proba(X_test_f)[:, 1]
    metrics = _compute_metrics(y_test, y_test_pred, y_test_proba)
    acc, f1, auroc = metrics["accuracy"], metrics["f1"], metrics["auroc"]
    print("\n--- Baseline (majority class) ---")
    print(f"Majority class from train: {majority_class}")
    print(f"Accuracy: {base_acc:.4f}")
    print(f"F1 (binary): {base_f1:.4f}")
    print(f"AUROC: {base_auroc:.4f}")

    print("\n--- Logistic Regression test set ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 (binary): {f1:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_test_pred, target_names=["human", "machine"]))

    # Ablation study: drop one feature group at a time
    names = feature_names()
    ablation_results = _run_ablation_study(X_train_raw, y_train, X_test_raw, y_test, names, seed)
    print("\n--- Ablation study (test set) ---")
    print(f"{'Setting':<24} {'Accuracy':>10} {'F1':>10} {'AUROC':>10}")
    for setting, m in ablation_results.items():
        print(f"{setting:<24} {m['accuracy']:>10.4f} {m['f1']:>10.4f} {m['auroc']:>10.4f}")

    # Error analysis: write FP/FN examples for report
    error_path = os.path.join("outputs", "error_analysis.csv")
    _write_error_analysis(error_path, X_test, y_test, y_test_pred, y_test_proba, max_rows=10)
    print(f"\nSaved error analysis samples to: {error_path}")

    # Feature analysis: mean per class on test set (before scaling we need raw features)
    human_means = X_test_raw[y_test == 0].mean(axis=0)
    machine_means = X_test_raw[y_test == 1].mean(axis=0)
    print("\n--- Feature means (test set) ---")
    print(f"{'Feature':<25} {'Human':>10} {'Machine':>10}")
    for i, n in enumerate(names):
        print(f"{n:<25} {human_means[i]:>10.4f} {machine_means[i]:>10.4f}")

    return {
        "clf": clf,
        "scaler": scaler,
        "baseline_majority_accuracy": base_acc,
        "baseline_majority_f1": base_f1,
        "baseline_majority_auroc": base_auroc,
        "ablation_results": ablation_results,
        "error_analysis_path": error_path,
        "test_accuracy": acc,
        "test_f1": f1,
        "test_auroc": auroc,
    }


if __name__ == "__main__":
    run_pipeline(subset="finance", max_samples=1500)
