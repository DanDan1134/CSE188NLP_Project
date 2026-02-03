"""
Formality feature extraction for human vs machine text detection.
Features: contractions, informal words, sentence length stats, punctuation, capitalization.
"""

import re
from typing import List, Dict, Any

# Common contractions (machine text often expands these)
CONTRACTIONS = {
    "i'm", "i've", "i'll", "i'd", "you're", "you've", "you'll", "you'd",
    "he's", "he'll", "he'd", "she's", "she'll", "she'd", "it's", "it'll", "it'd",
    "we're", "we've", "we'll", "we'd", "they're", "they've", "they'll", "they'd",
    "that's", "that'll", "that'd", "what's", "what're", "what'll", "what'd",
    "who's", "who're", "who'll", "who'd", "where's", "where'd", "when's", "when'd",
    "why's", "why'd", "how's", "how'd", "don't", "doesn't", "didn't", "won't",
    "wouldn't", "couldn't", "shouldn't", "can't", "cannot", "isn't", "aren't",
    "wasn't", "weren't", "haven't", "hasn't", "hadn't", "ain't", "'s", "'re", "'ve", "'ll", "'d",
}

# Informal / slang words (humans use more in casual text)
INFORMAL_WORDS = {
    "gonna", "wanna", "gotta", "kinda", "sorta", "oughta", "lotsa", "dunno",
    "yeah", "yep", "nope", "nah", "yup", "uh", "um", "hmm", "lol", "lmao",
    "omg", "idk", "imo", "tbh", "btw", "imo", "imo", "jk", "smh", "imo",
    "awesome", "cool", "nice", "sucks", "kinda", "pretty", "really", "super",
    "stuff", "thing", "things", "guys", "guy", "dude", "buddy", "folks",
    "gonna", "wanna", "kinda", "gotta", "coulda", "shoulda", "woulda", "musta",
}

# Punctuation that tends to appear more in informal human text
EXCLAMATION = "!"
QUESTION = "?"
ELLIPSIS = "..."
DOTS = ".."


def _get_sentences(text: str) -> List[str]:
    """Split text into sentences (simple regex; use nltk for better results)."""
    if not text or not str(text).strip():
        return [""]
    # Split on . ! ? and filter empty
    sentences = re.split(r"[.!?]+", str(text))
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences if sentences else [str(text)]


def _get_words(text: str) -> List[str]:
    """Lowercased words (letters + optional apostrophe)."""
    return re.findall(r"[a-z']+", str(text).lower())


def contraction_rate(text: str) -> float:
    """Proportion of tokens that are contractions (or contain apostrophe contraction)."""
    words = _get_words(text)
    if not words:
        return 0.0
    count = sum(1 for w in words if w in CONTRACTIONS or (len(w) > 1 and "'" in w))
    return count / len(words)


def informal_word_rate(text: str) -> float:
    """Proportion of tokens that are in informal/slang list."""
    words = _get_words(text)
    if not words:
        return 0.0
    count = sum(1 for w in words if w in INFORMAL_WORDS)
    return count / len(words)


def sentence_length_mean_std(text: str) -> tuple:
    """(mean word count per sentence, std). Machine text often has lower std."""
    sentences = _get_sentences(text)
    if not sentences:
        return 0.0, 0.0
    lengths = [len(_get_words(s)) for s in sentences]
    mean = sum(lengths) / len(lengths)
    variance = sum((x - mean) ** 2 for x in lengths) / len(lengths) if len(lengths) > 1 else 0
    std = variance ** 0.5
    return mean, std


def exclamation_rate(text: str) -> float:
    """Exclamation marks per sentence (informal human text often higher)."""
    sentences = _get_sentences(text)
    if not sentences:
        return 0.0
    count = sum(1 for s in text if s == EXCLAMATION)
    return count / len(sentences)


def question_rate(text: str) -> float:
    """Question marks per sentence."""
    sentences = _get_sentences(text)
    if not sentences:
        return 0.0
    count = text.count(QUESTION)
    return count / len(sentences)


def all_caps_ratio(text: str) -> float:
    """Proportion of words that are all-caps (e.g. I HATE THIS)."""
    words = re.findall(r"\b[A-Za-z]+\b", str(text))
    if not words:
        return 0.0
    caps = sum(1 for w in words if len(w) > 1 and w.isupper())
    return caps / len(words)


def avg_word_length(text: str) -> float:
    """Average word length (informal sometimes shorter words)."""
    words = _get_words(text)
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def extract_all_features(text: str) -> Dict[str, float]:
    """Extract all formality-related features for one document."""
    mean_len, std_len = sentence_length_mean_std(text)
    return {
        "contraction_rate": contraction_rate(text),
        "informal_word_rate": informal_word_rate(text),
        "sentence_length_mean": mean_len,
        "sentence_length_std": std_len,
        "exclamation_rate": exclamation_rate(text),
        "question_rate": question_rate(text),
        "all_caps_ratio": all_caps_ratio(text),
        "avg_word_length": avg_word_length(text),
        "num_sentences": float(len(_get_sentences(text))),
        "num_words": float(len(_get_words(text))),
    }


def feature_names() -> List[str]:
    """Ordered list of feature names for sklearn."""
    return list(extract_all_features("dummy").keys())
