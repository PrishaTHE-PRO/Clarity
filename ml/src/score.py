"""The full 'worth reading' decision = CLARITY model x IMPORTANCE gate.

The ONNX/scikit model answers only "is this written clearly?" (the fuzzy,
perceptual part ML is good at). Importance is a crisp logical property, so we
handle it with a small transparent rule layer. This is the reference for the
JavaScript version in the extension — keep the two in lock-step.

Definition:
  worth_reading = clarity_prob            IF the sentence is substantive
                  clarity_prob * PENALTY  otherwise

"Substantive" = the sentence names enough concrete things. We approximate that
with a portable rule (no POS tagger, so JS can match it): count words that are
neither function words (the, is, and, ...) nor vague words (thing, stuff, ...).
Trivial filler like "he did the stuff and it was fine" has almost none.
"""

import json
import pickle
from pathlib import Path

import pandas as pd

from features import extract_features, tokenize

MODEL_DIR = Path(__file__).resolve().parents[1] / "model"

with open(MODEL_DIR / "model.pkl", "rb") as f:
    _MODEL = pickle.load(f)
with open(MODEL_DIR / "feature_names.json") as f:
    _FEATURES = json.load(f)

# Importance-gate tunables.
MIN_CONTENT_WORDS = 3
FILLER_PENALTY = 0.35

# Function words carry grammar, not content. (Kept in sync with content.js.)
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "so", "of", "to",
    "in", "on", "at", "by", "for", "with", "from", "as", "into", "onto",
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does",
    "did", "have", "has", "had", "will", "would", "shall", "should", "can",
    "could", "may", "might", "must", "i", "you", "he", "she", "it", "we",
    "they", "me", "him", "her", "us", "them", "my", "your", "his", "its",
    "our", "their", "this", "that", "these", "those", "who", "whom", "which",
    "what", "there", "here", "when", "where", "why", "how", "not", "no",
    "yes", "very", "just", "also", "too", "more", "most", "some", "any",
    "all", "each", "than", "about", "up", "down", "out", "off", "over",
    "again", "once", "get", "got", "go", "went", "make", "made", "said",
}

# Words that are grammatically content but semantically empty.
VAGUE_WORDS = {
    "thing", "things", "stuff", "someone", "something", "anything",
    "everything", "nothing", "somewhere", "anywhere", "somehow", "way",
    "ways", "bit", "lot", "lots", "kind", "sort", "one", "ones", "okay",
    "fine", "part", "point", "time", "times", "day", "days",
}


def count_content_words(sentence: str) -> int:
    """Concrete content words = tokens that are neither stopwords nor vague."""
    return sum(
        1 for w in tokenize(sentence)
        if len(w) > 2 and w not in STOPWORDS and w not in VAGUE_WORDS
    )


def clarity_prob(feats: dict) -> float:
    """Model probability that the sentence is clearly written."""
    X = pd.DataFrame([[feats[n] for n in _FEATURES]], columns=_FEATURES)
    return float(_MODEL.predict_proba(X)[0][1])


def worth_reading(sentence: str) -> dict:
    """Return the full decision for one sentence (None if too short to score)."""
    feats = extract_features(sentence)
    if feats is None:
        return None
    clear = clarity_prob(feats)
    substantive = count_content_words(sentence) >= MIN_CONTENT_WORDS
    score = clear if substantive else clear * FILLER_PENALTY
    return {
        "score": round(score, 3),
        "clarity": round(clear, 3),
        "substantive": substantive,
    }


if __name__ == "__main__":
    battery = [
        ("Vaccines train your immune system to fight a virus before you catch it.", "clear+important"),
        ("The mitochondrion is the powerhouse of the cell.", "clear+important"),
        ("The global economy grew 3 percent in 2023.", "clear+important w/ number"),
        ("Notwithstanding the aforementioned epistemological constraints, the heuristic remains ontologically indeterminate.", "dense jargon"),
        ("The utilization of said methodology necessitates a comprehensive reevaluation of extant paradigmatic frameworks.", "dense jargon"),
        ("It was a thing that happened at some point, more or less, to some of them.", "vague filler"),
        ("He went there and then he did the stuff and it was fine and okay.", "low-value filler"),
        ("Photosynthesis converts sunlight into the sugars that plants use for energy.", "clear+important"),
    ]
    print(f'{"score":>6} {"clarity":>8} {"subst":>6}  quality              sentence')
    for s, label in battery:
        r = worth_reading(s)
        print(f'{r["score"]:6.2f} {r["clarity"]:8.2f} {str(r["substantive"]):>6}  '
              f'{label:20s} {s[:50]}')
