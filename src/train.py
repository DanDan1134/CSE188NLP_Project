"""
Train formality-based classifier: extract features, train Logistic Regression, evaluate.
"""

from typing import Optional
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
    X_train_f = texts_to_feature_matrix(X_train)
    X_val_f = texts_to_feature_matrix(X_val)
    X_test_f = texts_to_feature_matrix(X_test)

    scaler = StandardScaler()
    X_train_f = scaler.fit_transform(X_train_f)
    X_val_f = scaler.transform(X_val_f)
    X_test_f = scaler.transform(X_test_f)

    print("Training Logistic Regression...")
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X_train_f, y_train)

    # Validation
    y_val_pred = clf.predict(X_val_f)
    print(f"\nValidation: accuracy={accuracy_score(y_val, y_val_pred):.4f}, F1={f1_score(y_val, y_val_pred):.4f}")

    # Test
    y_test_pred = clf.predict(X_test_f)
    y_test_proba = clf.predict_proba(X_test_f)[:, 1]
    acc = accuracy_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    try:
        auroc = roc_auc_score(y_test, y_test_proba)
    except Exception:
        auroc = 0.5
    print("\n--- Test set ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 (binary): {f1:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_test_pred, target_names=["human", "machine"]))

    # Feature analysis: mean per class on test set (before scaling we need raw features)
    X_test_raw = texts_to_feature_matrix(X_test)
    names = feature_names()
    human_means = X_test_raw[y_test == 0].mean(axis=0)
    machine_means = X_test_raw[y_test == 1].mean(axis=0)
    print("\n--- Feature means (test set) ---")
    print(f"{'Feature':<25} {'Human':>10} {'Machine':>10}")
    for i, n in enumerate(names):
        print(f"{n:<25} {human_means[i]:>10.4f} {machine_means[i]:>10.4f}")

    return {
        "clf": clf,
        "scaler": scaler,
        "test_accuracy": acc,
        "test_f1": f1,
        "test_auroc": auroc,
    }


if __name__ == "__main__":
    run_pipeline(subset="finance", max_samples=1500)
