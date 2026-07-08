"""Build a labeled sentence dataset for the "worth reading" classifier.

Strategy (Wikipedia):
  - The lead/intro section of an article is written to be the most essential,
    accessible summary  -> label 1 (important + clear).
  - Body-section prose is supporting detail                -> label 0.

We use the MediaWiki `extracts` API (not the REST `sections` endpoint, which
does not return clean plain-text body content):
  - exintro=1        -> just the lead section
  - (no exintro)     -> the full article as plain text
  - body = full text with the lead prefix removed, minus trailing
    reference/see-also sections.

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

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "ClarityDataCollector/1.0 (educational project)"}


def make_session():
    """A session that retries on rate limits / transient errors with backoff.

    Wikipedia returns HTTP 429 if we hit it too fast; urllib3's Retry honors
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
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.headers.update(HEADERS)
    return s


SESSION = make_session()

# Sections that are lists/metadata, not prose worth learning from.
STOP_SECTIONS = (
    "See also",
    "References",
    "External links",
    "Further reading",
    "Notes",
    "Bibliography",
    "Citations",
    "Sources",
)

OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "labeled_sentences.csv"


def fetch_extract(title, intro_only):
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
    r = SESSION.get(API, params=params, timeout=30)
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    if not pages:
        return ""
    page = next(iter(pages.values()))
    return page.get("extract", "") or ""


def strip_reference_sections(text):
    """Cut everything from the first reference/see-also heading onward.

    With explaintext, section headings appear on their own line as plain text,
    so we look for a stop-section title alone on a line.
    """
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
    # Drop lines that are mostly a heading (Title Case, no verb-ish content).
    if s.isupper():
        return False
    return True


def split_sentences(text):
    # Collapse the newline-delimited paragraph structure before tokenizing.
    text = re.sub(r"\s*\n\s*", " ", text).strip()
    return [s.strip() for s in sent_tokenize(text) if is_good_sentence(s)]


def collect_article(title):
    """Return (lead_sentences, body_sentences) for one article."""
    intro = fetch_extract(title, intro_only=True)
    full = fetch_extract(title, intro_only=False)
    if not intro or not full:
        return [], []

    # Body = full text with the leading intro removed.
    body = full[len(intro):] if full.startswith(intro) else full
    body = strip_reference_sections(body)

    lead_sentences = split_sentences(intro)
    body_sentences = split_sentences(body)
    return lead_sentences, body_sentences


# ~120 diverse topics across science, tech, history, arts, society, geography.
topics = [
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
    "Virus", "Telescope", "Democracy_in_ancient_Greece", "Buddhism",
    "Enzyme", "Tectonic_plate", "Albert_Einstein", "Stock_market",
    "Ocean_current", "Cell_(biology)", "Printing_press", "Opera",
    "Thermodynamics", "Amazon_rainforest", "Federalism", "Earthquake",
    "Isaac_Newton", "Photosynthetic", "Gross_domestic_product",
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
]


def main():
    rows = []  # (sentence, label, source)
    seen = set()

    for i, title in enumerate(topics, 1):
        try:
            lead, body = collect_article(title)
        except requests.RequestException as e:
            print(f"[{i}/{len(topics)}] {title}: request failed ({e})")
            continue

        if not lead:
            print(f"[{i}/{len(topics)}] {title}: no lead content, skipping")
            continue

        # Balance: cap body sentences to the number of lead sentences so the
        # two classes stay roughly even. Sample to avoid front-loading.
        cap = len(lead)
        if len(body) > cap:
            body = random.sample(body, cap)

        added = 0
        for s in lead:
            if s not in seen:
                seen.add(s)
                rows.append((s, 1, "wikipedia"))
                added += 1
        for s in body:
            if s not in seen:
                seen.add(s)
                rows.append((s, 0, "wikipedia"))
                added += 1

        print(f"[{i}/{len(topics)}] {title}: +{added} "
              f"(lead={len(lead)}, body={len(body)})")
        time.sleep(0.5)  # be polite to the API

    random.shuffle(rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sentence", "label", "source"])
        w.writerows(rows)

    n_pos = sum(1 for r in rows if r[1] == 1)
    n_neg = len(rows) - n_pos
    print(f"\nWrote {len(rows)} sentences to {OUT_PATH}")
    print(f"  label 1 (lead):  {n_pos}")
    print(f"  label 0 (body):  {n_neg}")


if __name__ == "__main__":
    main()
