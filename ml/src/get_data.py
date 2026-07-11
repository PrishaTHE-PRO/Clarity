"""Build a labeled sentence dataset for the "worth reading" classifier.

Two complementary sources, both via the MediaWiki `extracts` API (reliable
plain text, no fragile HTML scraping):

Source 1 - Wikipedia (encyclopedic prose)
  - Lead/intro section  -> label 1 (essential, accessible summary)
  - Body-section prose  -> label 0 (supporting detail)

Source 2 - Wikinews (inverted-pyramid journalism)
  - Lede (first sentences) -> label 1 (who/what/when/where, high density)
  - Later body sentences   -> label 0 (background, quotes, detail)

Output: ml/data/labeled_sentences.csv  with columns: sentence, label, source
"""

import csv
import random
import re
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from nltk.tokenize import sent_tokenize

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
SIMPLE_API = "https://simple.wikipedia.org/w/api.php"
WIKINEWS_API = "https://en.wikinews.org/w/api.php"
HEADERS = {"User-Agent": "ClarityDataCollector/1.0 (educational project)"}

# Sections that are lists/metadata, not prose worth learning from.
STOP_SECTIONS = (
    "See also", "References", "External links", "Further reading",
    "Notes", "Bibliography", "Citations", "Sources",
    "Related news", "Related articles",
)

# Wikinews articles start with a dateline like "Wednesday, September 1, 2010".
DATELINE_RE = re.compile(
    r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday), .+\d{4}$"
)

OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "labeled_sentences.csv"


def make_session():
    """A session that retries on rate limits / transient errors with backoff.

    Wikimedia returns HTTP 429 if we hit it too fast; urllib3's Retry honors
    the Retry-After header and backs off exponentially between attempts.
    """
    s = requests.Session()
    retry = Retry(
        total=6,
        backoff_factor=1.5,  # waits 0, 1.5, 3, 6, 12, 24s between tries
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update(HEADERS)
    return s


SESSION = make_session()


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def fetch_extract(api, title, intro_only):
    """Return the plain-text extract for an article (intro only, or full)."""
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
    }
    if intro_only:
        params["exintro"] = 1
    r = SESSION.get(api, params=params, timeout=30)
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    if not pages:
        return ""
    page = next(iter(pages.values()))
    return page.get("extract", "") or ""


def strip_reference_sections(text):
    """Cut everything from the first reference/see-also heading onward."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() in STOP_SECTIONS:
            return "\n".join(lines[:i])
    return text


def is_good_sentence(s):
    """Keep substantive prose; drop headings, fragments, and junk."""
    s = s.strip()
    if len(s) < 40 or len(s) > 400:
        return False
    if s[-1] not in ".!?":  # headings/fragments rarely end in punctuation
        return False
    if len(s.split()) < 6:
        return False
    if s.startswith(("=", "*", "•", "-")):
        return False
    if s.isupper():
        return False
    return True


def split_sentences(text):
    """Collapse paragraph structure, tokenize, and keep good sentences."""
    text = re.sub(r"\s*\n\s*", " ", text).strip()
    return [s.strip() for s in sent_tokenize(text) if is_good_sentence(s)]


# --------------------------------------------------------------------------
# Source 1: Wikipedia (lead vs body)
# --------------------------------------------------------------------------

def collect_wikipedia_article(title):
    """Return (lead_sentences, body_sentences) for one Wikipedia article."""
    intro = fetch_extract(WIKIPEDIA_API, title, intro_only=True)
    full = fetch_extract(WIKIPEDIA_API, title, intro_only=False)
    if not intro or not full:
        return [], []

    body = full[len(intro):] if full.startswith(intro) else full
    body = strip_reference_sections(body)
    return split_sentences(intro), split_sentences(body)


WIKIPEDIA_TOPICS = [
    "Machine_learning", "Climate_change", "Python_(programming_language)",
    "World_War_II", "Photosynthesis", "Democracy", "Black_hole",
    "Artificial_intelligence", "Evolution", "Quantum_mechanics",
    "Ancient_Rome", "French_Revolution", "DNA", "Electricity",
    "Internet", "Gravity", "Vaccine", "Renaissance", "Volcano",
    "Immune_system", "Solar_System", "Great_Depression", "Jazz",
    "Plate_tectonics", "Antibiotic", "Cryptography", "Neuron",
    "Industrial_Revolution", "Photography", "Ecosystem", "Genetics",
    "Cold_War", "Blockchain", "Coral_reef", "Higgs_boson",
    "Human_brain", "Renewable_energy", "Shakespeare", "Pandemic",
    "Nuclear_fission", "Continental_drift", "Impressionism",
    "Supply_and_demand", "Ozone_layer", "Bacteria", "Meteorology",
    "Big_Bang", "Constitution_of_the_United_States", "Beethoven",
    "Rainforest", "Semiconductor", "Ecology", "Magnetism",
    "Roman_Empire", "Vincent_van_Gogh", "Inflation", "Glacier",
    "Virus", "Telescope", "Buddhism", "Enzyme", "Albert_Einstein",
    "Stock_market", "Ocean_current", "Cell_(biology)", "Printing_press",
    "Opera", "Thermodynamics", "Amazon_rainforest", "Federalism",
    "Earthquake", "Isaac_Newton", "Gross_domestic_product",
    "Antarctica", "Metabolism", "Radio", "Gothic_architecture",
    "Capitalism", "Tsunami", "Charles_Darwin", "Electron",
    "Monetary_policy", "Sahara", "Protein", "Vaccination",
    "Baroque_music", "Socialism", "Hurricane", "Marie_Curie",
    "Atom", "Deforestation", "Middle_Ages", "Ballet",
    "Nervous_system", "Solar_energy", "Byzantine_Empire",
    "Galaxy", "Antibody", "Steam_engine", "Surrealism",
    "Globalization", "Wildfire", "Nikola_Tesla", "Photon",
    "Interest_rate", "Mount_Everest", "Chromosome", "Television",
    "Roman_Republic", "Fresco", "Communism", "Drought",
    "Leonardo_da_Vinci", "Molecule", "Reforestation",
    # More common topics for the Simple-vs-Regular dataset (more volume).
    "Water", "Fire", "Sun", "Moon", "Earth", "Ocean", "Mountain", "River",
    "Tree", "Forest", "Desert", "Island", "Lake", "Wind", "Rain", "Snow",
    "Dog", "Cat", "Horse", "Bird", "Fish", "Insect", "Spider", "Whale",
    "Elephant", "Lion", "Tiger", "Bear", "Wolf", "Shark", "Dinosaur",
    "Heart", "Lung", "Kidney", "Liver", "Blood", "Bone", "Muscle", "Skin",
    "Music", "Painting", "Sculpture", "Dance", "Poetry", "Film", "Theatre",
    "Football", "Basketball", "Tennis", "Cricket", "Chess", "Olympic_Games",
    "Computer", "Software", "Smartphone", "Robot", "Electric_car",
    "Airplane", "Ship", "Train", "Bicycle", "Rocket", "Satellite",
    "Language", "Alphabet", "Number", "Mathematics", "Geometry", "Algebra",
    "History", "Geography", "Economy", "Government", "Law", "Religion",
    "Sound", "Light", "Heat", "Color", "Energy", "Force", "Motion",
    "Star", "Planet", "Comet", "Asteroid", "Moon_landing", "Space_station",
]


def collect_wikipedia(rows, seen):
    for i, title in enumerate(WIKIPEDIA_TOPICS, 1):
        try:
            lead, body = collect_wikipedia_article(title)
        except requests.RequestException as e:
            print(f"[wiki {i}/{len(WIKIPEDIA_TOPICS)}] {title}: failed ({e})")
            continue
        if not lead:
            print(f"[wiki {i}/{len(WIKIPEDIA_TOPICS)}] {title}: no lead, skip")
            continue

        # Balance per article: cap body to the lead count, sampled.
        if len(body) > len(lead):
            body = random.sample(body, len(lead))
        added = _add(rows, seen, lead, 1, "wikipedia") + \
            _add(rows, seen, body, 0, "wikipedia")
        print(f"[wiki {i}/{len(WIKIPEDIA_TOPICS)}] {title}: +{added}")
        time.sleep(0.5)


# --------------------------------------------------------------------------
# Source 2: Wikinews (lede vs body)
# --------------------------------------------------------------------------

LEDE_SENTENCES = 2  # first N sentences of a news article count as the lede


def clean_wikinews(text):
    """Drop datelines, image refs, and blank lines from a Wikinews extract."""
    kept = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if DATELINE_RE.match(s):
            continue
        if s.startswith("File:") or s.startswith("Image:"):
            continue
        kept.append(s)
    return strip_reference_sections("\n".join(kept))


def random_wikinews_titles(n):
    """Fetch up to n random Wikinews article titles (main namespace)."""
    titles = []
    while len(titles) < n:
        batch = min(50, n - len(titles))
        r = SESSION.get(WIKINEWS_API, params={
            "action": "query", "format": "json", "list": "random",
            "rnnamespace": 0, "rnlimit": batch,
        }, timeout=30)
        r.raise_for_status()
        titles.extend(p["title"] for p in r.json()["query"]["random"])
        time.sleep(0.3)
    # De-duplicate (random can repeat) while preserving order.
    return list(dict.fromkeys(titles))


def collect_wikinews(rows, seen, n_articles):
    try:
        titles = random_wikinews_titles(n_articles)
    except requests.RequestException as e:
        print(f"[news] could not list articles: {e}")
        return

    for i, title in enumerate(titles, 1):
        try:
            raw = fetch_extract(WIKINEWS_API, title, intro_only=False)
        except requests.RequestException as e:
            print(f"[news {i}/{len(titles)}] {title}: failed ({e})")
            continue

        sents = split_sentences(clean_wikinews(raw))
        if len(sents) < LEDE_SENTENCES + 1:
            continue

        lede = sents[:LEDE_SENTENCES]
        body = sents[LEDE_SENTENCES:]
        if len(body) > len(lede):
            body = random.sample(body, len(lede))
        added = _add(rows, seen, lede, 1, "wikinews") + \
            _add(rows, seen, body, 0, "wikinews")
        print(f"[news {i}/{len(titles)}] {title[:50]}: +{added}")
        time.sleep(0.4)


# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Source 3 (primary): Simple English lead vs Regular English body
#
# Captures BOTH axes of "worth reading":
#   positive = Simple English Wikipedia lead   -> clear AND essential
#   negative = Regular English Wikipedia body  -> complex AND peripheral
# Labeled by wiki variant + section, so it is not a pure position proxy.
# --------------------------------------------------------------------------

def collect_simple_vs_regular(rows, seen):
    """Pure CLARITY signal: Simple English (clear) vs Regular English (complex).

      positive (1): Simple English sentences (lead + body) -> clearly written
      negative (0): Regular English sentences (lead + body) -> complex

    Using both lead and body from each variant removes the position confound:
    the label depends ONLY on writing style, not where the sentence sits. The
    result is a focused "is this written clearly?" model. Importance (is this
    worth reading, not just readable?) is handled separately by a rule layer in
    score.py / the extension.
    """
    topics = WIKIPEDIA_TOPICS
    for i, title in enumerate(topics, 1):
        try:
            simple_full = fetch_extract(SIMPLE_API, title, intro_only=False)
            reg_full = fetch_extract(WIKIPEDIA_API, title, intro_only=False)
        except requests.RequestException as e:
            print(f"[{i}/{len(topics)}] {title}: failed ({e})")
            continue

        simple_sents = split_sentences(strip_reference_sections(simple_full))
        reg_sents = split_sentences(strip_reference_sections(reg_full))
        if not simple_sents or not reg_sents:
            print(f"[{i}/{len(topics)}] {title}: missing on one wiki, skip")
            continue

        # Balance per topic (Regular articles are much longer than Simple).
        k = min(len(simple_sents), len(reg_sents))
        pos = simple_sents[:k]
        neg = random.sample(reg_sents, k) if len(reg_sents) > k else reg_sents

        added = _add(rows, seen, pos, 1, "simple") + \
            _add(rows, seen, neg, 0, "regular")
        print(f"[{i}/{len(topics)}] {title}: +{added} (each={k})")
        time.sleep(0.5)


def _add(rows, seen, sentences, label, source):
    added = 0
    for s in sentences:
        if s not in seen:
            seen.add(s)
            rows.append((s, label, source))
            added += 1
    return added


def main():
    rows = []
    seen = set()

    print("=== Simple English (clear) vs Regular English (complex) ===")
    collect_simple_vs_regular(rows, seen)

    random.shuffle(rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sentence", "label", "source"])
        w.writerows(rows)

    n_pos = sum(1 for r in rows if r[1] == 1)
    by_src = {}
    for _, label, src in rows:
        by_src.setdefault(src, [0, 0])[label] += 1
    print(f"\nWrote {len(rows)} sentences to {OUT_PATH}")
    print(f"  label 1: {n_pos}   label 0: {len(rows) - n_pos}")
    for src, (neg, pos) in by_src.items():
        print(f"  {src}: {pos} pos / {neg} neg")


if __name__ == "__main__":
    main()
