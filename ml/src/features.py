"""Feature extraction: turn a sentence into a numeric vector for the model.

Each feature is a signal for either IMPORTANCE (is this worth reading?) or
CLARITY (is this easy to read?).
"""

import re
from pathlib import Path

import nltk
import syllables

# Anchor data paths to this file so the module works from any cwd.
_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Top 5000 common English words — presence signals "easy to read".
# Sourced from: https://github.com/first20hours/google-10000-english
with open(_DATA_DIR / "common_words.txt", encoding="utf-8") as f:
    COMMON_WORDS = set(f.read().splitlines())

CLAIM_WORDS = {
    "shows", "found", "reveals", "causes", "leads", "results",
    "because", "therefore", "however", "despite", "although",
    "first", "most", "key", "main", "primary", "significant",
}


def extract_features(sentence: str) -> dict:
    """
    Turn a sentence into a vector of numbers the model can learn from.
    Returns None for sentences too short to be meaningful.
    """
    words = [w for w in nltk.word_tokenize(sentence) if w.isalpha()]

    if len(words) < 3:
        return None

    n = len(words)
    avg_word_len = sum(len(w) for w in words) / n
    avg_syllables = sum(syllables.estimate(w) for w in words) / n
    rare_ratio = sum(1 for w in words if w.lower() not in COMMON_WORDS) / n
    claim_ratio = sum(1 for w in words if w.lower() in CLAIM_WORDS) / n

    # Numbers = signals of specificity and importance.
    has_number = int(bool(re.search(r"\d", sentence)))

    word_count = n

    # Sentences in the 10-25 word range tend to be most readable; score
    # falls off for very short (no info) or very long (hard to parse) ones.
    length_score = 1.0 if 10 <= n <= 25 else max(0, 1 - abs(n - 17) / 20)

    return {
        "word_count": word_count,
        "avg_word_len": avg_word_len,
        "avg_syllables": avg_syllables,
        "rare_word_ratio": rare_ratio,
        "claim_ratio": claim_ratio,
        "has_number": has_number,
        "length_score": length_score,
    }
