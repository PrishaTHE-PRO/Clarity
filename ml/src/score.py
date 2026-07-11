"""The full 'worth reading' decision = CLARITY model x IMPORTANCE gate.

The ONNX/scikit model answers only "is this written clearly?" (the fuzzy,
perceptual part ML is good at). Importance is a crisp logical property, so we
handle it with a small transparent rule layer here. The extension mirrors this
exact logic in JavaScript.

Definition:
  worth_reading = clarity_prob  IF the sentence is substantive
                  clarity_prob * FILLER_PENALTY  otherwise

A sentence is "substantive" when it is long enough AND carries at least one
mark of real content: a named entity, a number, or a claim/argument word.
Trivial filler ("he did the stuff and it was fine") has none of these, so it
gets penalized even though it is perfectly clear.
"""

import json
import pickle
from pathlib import Path

import nltk
import pandas as pd

from features import extract_features

MODEL_DIR = Path(__file__).resolve().parents[1] / "model"

with open(MODEL_DIR / "model.pkl", "rb") as f:
    _MODEL = pickle.load(f)
with open(MODEL_DIR / "feature_names.json") as f:
    _FEATURES = json.load(f)

# Importance-gate tunables.
MIN_CONTENT_WORDS = 3  # a substantive sentence names a few concrete things
FILLER_PENALTY = 0.35  # how hard to down-weight clear-but-empty sentences

# POS tags that carry real content (nouns, adjectives, numbers).
CONTENT_TAGS = {"NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS", "CD"}
# Words that are grammatically content but semantically empty -> filler tells.
VAGUE_WORDS = {
    "thing", "things", "stuff", "someone", "something", "anything",
    "everything", "nothing", "somewhere", "anywhere", "somehow", "way",
    "ways", "bit", "lot", "lots", "kind", "sort", "one", "ones", "okay",
    "fine", "some", "part", "point", "time", "times",
}


def count_content_words(sentence: str) -> int:
    """Count concrete content words (nouns/adjectives/numbers, minus vague ones).

    This is the importance signal: 'vaccines/immune/system/virus' has several
    concrete content words; 'he did the stuff and it was fine' has almost none
    (mostly pronouns and vague words), which is exactly what filler looks like.
    """
    tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    return sum(
        1 for w, t in tagged
        if t in CONTENT_TAGS and w.lower() not in VAGUE_WORDS and len(w) > 2
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
