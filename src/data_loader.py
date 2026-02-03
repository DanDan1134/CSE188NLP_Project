"""
Load human vs machine text for formality-based detection.
Supports: 'ziq/ai-generated-text-classification' (small, fast) or
          'ahmadreza13/human-vs-Ai-generated-dataset' (large).
"""

from typing import List, Tuple, Optional
import numpy as np


def load_hc3(
    subset: str = "finance",
    max_samples_per_class: Optional[int] = 2000,
    min_chars: int = 50,
    seed: int = 42,
    dataset_name: str = "ziq",
) -> Tuple[List[str], np.ndarray]:
    """
    Load human vs AI text dataset. Labels: 0 = human, 1 = machine.
    dataset_name: 'ziq' (small, ~1.4k rows) or 'ahmadreza13' (large, ~3.6M).
    subset is ignored (kept for API compatibility).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install: pip install datasets")

    if dataset_name == "ziq":
        # ziq: heavily imbalanced (1375 human, 3 machine) — not recommended for this task
        dataset = load_dataset("ziq/ai-generated-text-classification", split="train")
        text_key, label_key = "text", "generated"
        stream = False
    else:
        # ahmadreza13: use streaming to avoid loading 3.6M rows
        text_key, label_key = "data", "generated"
        stream = max_samples_per_class is not None

    human_texts: List[str] = []
    machine_texts: List[str] = []

    if stream and dataset_name == "ahmadreza13":
        # Stream and take first N per class
        try:
            dataset = load_dataset(
                "ahmadreza13/human-vs-Ai-generated-dataset",
                split="train",
                streaming=True,
                trust_remote_code=False,
            )
        except Exception:
            dataset = load_dataset("ahmadreza13/human-vs-Ai-generated-dataset", split="train")
            stream = False
        if stream:
            n_per_class = max_samples_per_class or 2000
            for row in dataset:
                if len(human_texts) >= n_per_class and len(machine_texts) >= n_per_class:
                    break
                text = (row.get(text_key) or "").strip()
                if not text or len(text) < min_chars:
                    continue
                label = row.get(label_key, -1)
                is_human = label in (0, False, "0") or label == 0.0
                is_machine = label in (1, True, "1") or label == 1.0
                if is_human and len(human_texts) < n_per_class:
                    human_texts.append(text)
                elif is_machine and len(machine_texts) < n_per_class:
                    machine_texts.append(text)
        else:
            dataset = load_dataset("ahmadreza13/human-vs-Ai-generated-dataset", split="train")
            stream = False

    if not stream or dataset_name == "ziq":
        if dataset_name == "ziq":
            pass  # dataset already loaded above
        elif not stream:
            dataset = load_dataset("ahmadreza13/human-vs-Ai-generated-dataset", split="train")
        n_total = len(dataset)
        for i in range(n_total):
            row = dataset[i]
            text = (row.get(text_key) or row.get("text") or "").strip()
            if not text or len(text) < min_chars:
                continue
            label = row.get(label_key, -1)
            is_human = label in (0, False, "0") or label == 0.0
            is_machine = label in (1, True, "1") or label == 1.0
            if is_human:
                human_texts.append(text)
            elif is_machine:
                machine_texts.append(text)

    np.random.seed(seed)
    if max_samples_per_class is not None and len(human_texts) > 0 and len(machine_texts) > 0:
        n = min(max_samples_per_class, len(human_texts), len(machine_texts))
        if n < len(human_texts) or n < len(machine_texts):
            human_idx = np.random.choice(len(human_texts), size=n, replace=False)
            machine_idx = np.random.choice(len(machine_texts), size=n, replace=False)
            human_texts = [human_texts[i] for i in human_idx]
            machine_texts = [machine_texts[i] for i in machine_idx]

    texts = human_texts + machine_texts
    labels = [0] * len(human_texts) + [1] * len(machine_texts)
    perm = np.random.permutation(len(texts))
    texts = [texts[i] for i in perm]
    labels = np.array([labels[i] for i in perm], dtype=np.int64)
    return texts, labels


def train_val_test_split(
    texts: List[str],
    labels: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], np.ndarray, List[str], np.ndarray, List[str], np.ndarray]:
    """Stratified split into train / val / test."""
    from sklearn.model_selection import train_test_split

    X_train, X_rest, y_train, y_rest = train_test_split(
        texts, labels, train_size=train_ratio, stratify=labels, random_state=seed
    )
    val_ratio_rest = val_ratio / (1 - train_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, train_size=val_ratio_rest, stratify=y_rest, random_state=seed
    )
    return X_train, y_train, X_val, y_val, X_test, y_test
