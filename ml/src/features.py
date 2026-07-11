"""Feature extraction for the clarity model.

IMPORTANT — browser portability:
The extension recomputes these features in JavaScript at runtime, so every
feature here MUST be reproducible in JS *exactly*. That is why we deliberately
avoid NLTK tokenization and the `syllables` library (which have no faithful JS
equivalent) and instead use:
  - regex word tokenization   ->  matches JS  text.match(/[a-z]+/g)
  - a vowel-group syllable heuristic  ->  reimplemented identically in JS

Keep features.py and extension/content.js in lock-step: any change here must be
mirrored there, or Python training and JS inference will disagree.
"""

import re
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Top 5000 common English words — presence signals "easy to read".
# The same list is bundled into the extension as common_words.js.
with open(_DATA_DIR / "common_words.txt", encoding="utf-8") as f:
    COMMON_WORDS = set(f.read().splitlines())

_WORD_RE = re.compile(r"[a-z]+")
_VOWELS = set("aeiouy")


def tokenize(sentence: str):
    """Lowercase alphabetic word tokens. Mirror of JS: text.match(/[a-z]+/g)."""
    return _WORD_RE.findall(sentence.lower())


def count_syllables(word: str) -> int:
    """Vowel-group heuristic. Must stay identical to countSyllables() in JS."""
    count, prev_vowel = 0, False
    for ch in word:
        is_vowel = ch in _VOWELS
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:  # silent trailing 'e'
        count -= 1
    return max(1, count)


def extract_features(sentence: str) -> dict:
    """
    Turn a sentence into the 5 readability features the model learns from.
    Returns None for sentences too short to be meaningful.
    """
    words = tokenize(sentence)
    if len(words) < 3:
        return None

    n = len(words)
    avg_word_len = sum(len(w) for w in words) / n
    avg_syllables = sum(count_syllables(w) for w in words) / n
    rare_ratio = sum(1 for w in words if w not in COMMON_WORDS) / n

    # Sentences in the 10-25 word range tend to be most readable; score
    # falls off for very short (no info) or very long (hard to parse) ones.
    length_score = 1.0 if 10 <= n <= 25 else max(0, 1 - abs(n - 17) / 20)

    return {
        "word_count": n,
        "avg_word_len": avg_word_len,
        "avg_syllables": avg_syllables,
        "rare_word_ratio": rare_ratio,
        "length_score": length_score,
    }
