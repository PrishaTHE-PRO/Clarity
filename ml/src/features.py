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

# Penn Treebank POS tag prefixes we care about.
PROPER_NOUN_TAGS = {"NNP", "NNPS"}          # named entities -> specificity
VERB_TAGS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}  # active claims
NUMBER_TAG = "CD"                            # cardinal numbers -> specifics


def extract_features(sentence: str) -> dict:
    """
    Turn a sentence into a vector of numbers the model can learn from.
    Returns None for sentences too short to be meaningful.

    NOTE: features are content-only (computed from the sentence in isolation).
    We deliberately do NOT include sentence position: the training labels are
    derived from position (lead/lede vs body), so a position feature would leak
    the label and inflate accuracy without learning about the text itself.
    """
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    words = [w for w in tokens if w.isalpha()]

    if len(words) < 3:
        return None

    n = len(words)
    avg_word_len = sum(len(w) for w in words) / n
    avg_syllables = sum(syllables.estimate(w) for w in words) / n
    rare_ratio = sum(1 for w in words if w.lower() not in COMMON_WORDS) / n
    claim_ratio = sum(1 for w in words if w.lower() in CLAIM_WORDS) / n

    # POS-based content signals (ratios over alphabetic word count).
    proper_noun_ratio = sum(1 for _, t in tagged if t in PROPER_NOUN_TAGS) / n
    verb_ratio = sum(1 for _, t in tagged if t in VERB_TAGS) / n

    # Numeric specificity: both a flag and a count of number tokens.
    has_number = int(bool(re.search(r"\d", sentence)))
    number_count = sum(1 for _, t in tagged if t == NUMBER_TAG)

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
        "proper_noun_ratio": proper_noun_ratio,
        "verb_ratio": verb_ratio,
        "has_number": has_number,
        "number_count": number_count,
        "length_score": length_score,
    }
