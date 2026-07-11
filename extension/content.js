// content.js — runs on every page. It (1) finds prose text, (2) splits into
// sentences, (3) scores each with the clarity model + importance gate, and
// (4) highlights the ones worth reading.
//
// The feature and gate logic below MUST stay identical to the Python in
// ml/src/features.py and ml/src/score.py, or the ONNX model gets a different
// feature vector than it was trained on and predictions become noise.

const MODEL_URL = chrome.runtime.getURL("model/clarity_model.onnx");
const FEATURE_NAMES_URL = chrome.runtime.getURL("model/feature_names.json");

const MIN_CONTENT_WORDS = 3;
const FILLER_PENALTY = 0.35;
const MAX_SENTENCES = 1200; // safety cap for very long pages

// Highlight the top fraction of sentences by combined importance x clarity.
// Tune to taste: lower SELECT_RATIO = fewer, more selective highlights.
const SELECT_RATIO = 0.3;
const MIN_HIGHLIGHTS = 3;
const MAX_HIGHLIGHTS = 30;
const RELATIVE_FLOOR = 0.3; // drop chosen sentences weaker than 30% of the best

// Definitional sentences ("X is a process...", "Y refers to...") explain what
// things are, so they carry a lot of understanding — boost them.
const DEFINITION_BOOST = 1.6;
const DEFINITION_RE =
  /\b(is|are|was|were)\s+(a|an|the|one of|any|kind of|type of|form of|the process|a process|a type|a form|a kind|a set|a way|a method|a system|a group|a term)\b|\b(refers? to|defined as|is defined|are defined|means that|is known as|are known as|is called|are called|also called|also known as|consists? of|is the study)\b/i;

function isDefinition(sentence) {
  return DEFINITION_RE.test(sentence);
}

// ---------------------------------------------------------------------------
// Feature extraction — mirror of ml/src/features.py
// ---------------------------------------------------------------------------

const VOWELS = new Set(["a", "e", "i", "o", "u", "y"]);

function tokenize(text) {
  return text.toLowerCase().match(/[a-z]+/g) || [];
}

function countSyllables(word) {
  let count = 0;
  let prevVowel = false;
  for (const ch of word) {
    const isVowel = VOWELS.has(ch);
    if (isVowel && !prevVowel) count += 1;
    prevVowel = isVowel;
  }
  if (word.endsWith("e") && count > 1) count -= 1; // silent trailing 'e'
  return Math.max(1, count);
}

function extractFeatures(sentence) {
  const words = tokenize(sentence);
  if (words.length < 3) return null;

  const n = words.length;
  const avgWordLen = words.reduce((s, w) => s + w.length, 0) / n;
  const avgSyllables = words.reduce((s, w) => s + countSyllables(w), 0) / n;
  const rareRatio = words.filter((w) => !CLARITY_COMMON_WORDS.has(w)).length / n;
  const lengthScore =
    n >= 10 && n <= 25 ? 1.0 : Math.max(0, 1 - Math.abs(n - 17) / 20);

  return {
    word_count: n,
    avg_word_len: avgWordLen,
    avg_syllables: avgSyllables,
    rare_word_ratio: rareRatio,
    length_score: lengthScore,
  };
}

// ---------------------------------------------------------------------------
// Importance gate — mirror of ml/src/score.py
// ---------------------------------------------------------------------------

const STOPWORDS = new Set([
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
]);

const VAGUE_WORDS = new Set([
  "thing", "things", "stuff", "someone", "something", "anything",
  "everything", "nothing", "somewhere", "anywhere", "somehow", "way",
  "ways", "bit", "lot", "lots", "kind", "sort", "one", "ones", "okay",
  "fine", "part", "point", "time", "times", "day", "days",
]);

function contentTokens(sentence) {
  return tokenize(sentence).filter(
    (w) => w.length > 2 && !STOPWORDS.has(w) && !VAGUE_WORDS.has(w)
  );
}

function countContentWords(sentence) {
  return contentTokens(sentence).length;
}

// Importance signal: how many of the article's central terms a sentence
// carries. `df` = document frequency of each content word across all
// sentences, so words that recur through the article (its themes) weigh more.
function salienceRaw(sentence, df) {
  const cw = contentTokens(sentence);
  if (cw.length === 0) return 0;
  let sum = 0;
  for (const w of cw) {
    const recurrence = df.get(w) || 1; // central to the article?
    const rarity = CLARITY_COMMON_WORDS.has(w) ? 1 : 1.5; // distinctive term?
    sum += recurrence * rarity;
  }
  return sum / Math.sqrt(cw.length); // length-normalize
}

// ---------------------------------------------------------------------------
// DOM: find prose text nodes and split them into sentence segments
// ---------------------------------------------------------------------------

const SKIP_TAGS = new Set([
  "SCRIPT", "STYLE", "NOSCRIPT", "CODE", "PRE", "TEXTAREA", "INPUT",
  "BUTTON", "SELECT", "OPTION", "KBD", "SAMP",
]);

// Text must live inside one of these paragraph-like blocks to be considered.
const BLOCK_SELECTOR = "p, blockquote, h1, h2, h3, li, dd";

// Prefer the real article container; fall back to <body> if none exists.
const CONTENT_ROOT_SELECTOR = 'article, main, [role="main"]';

// Regions that are never article prose (chrome, promos, boilerplate).
const EXCLUDE_SELECTOR =
  'nav, aside, header, footer, form, figure, figcaption,' +
  '[role="navigation"], [role="complementary"], [role="banner"],' +
  '[role="contentinfo"], [aria-hidden="true"]';

// Class/id fragments that flag non-article widgets (subscribe boxes, ads,
// newsletters, related-links, share bars, captions, etc.).
const EXCLUDE_PATTERN =
  /subscri|promo|paywall|advert|newsletter|banner|footer|header|\bnav\b|menu|sidebar|related|recirc|market|upsell|signup|sign-up|social|share|comment|caption|byline|masthead|dock|ribbon|widget|cookie/i;

// The largest article/main region on the page (by text length).
function getContentRoot() {
  let best = null;
  let bestLen = 0;
  for (const el of document.querySelectorAll(CONTENT_ROOT_SELECTOR)) {
    const len = (el.textContent || "").length;
    if (len > bestLen) {
      best = el;
      bestLen = len;
    }
  }
  return best || document.body;
}

// Walk ancestors up to the root; reject if any is a non-article region.
function isExcluded(el, root) {
  let node = el;
  while (node && node !== root && node !== document.documentElement) {
    if (node.matches && node.matches(EXCLUDE_SELECTOR)) return true;
    const cls = node.getAttribute && node.getAttribute("class");
    if (cls && EXCLUDE_PATTERN.test(cls)) return true;
    if (node.id && EXCLUDE_PATTERN.test(node.id)) return true;
    node = node.parentElement;
  }
  return false;
}

function collectTextNodes(root) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const parent = node.parentElement;
      if (!parent) return NodeFilter.FILTER_REJECT;
      if (SKIP_TAGS.has(parent.tagName)) return NodeFilter.FILTER_REJECT;
      if (parent.closest(".clarity-highlight")) return NodeFilter.FILTER_REJECT;
      if (!parent.closest(BLOCK_SELECTOR)) return NodeFilter.FILTER_REJECT;
      if (isExcluded(parent, root)) return NodeFilter.FILTER_REJECT;
      if (!node.nodeValue || node.nodeValue.trim().length < 30)
        return NodeFilter.FILTER_REJECT;
      return NodeFilter.FILTER_ACCEPT;
    },
  });
  const nodes = [];
  let n;
  while ((n = walker.nextNode())) nodes.push(n);
  return nodes;
}

// Split text into segments, preserving EVERY character (gaps included) so we
// can rebuild the node without losing whitespace or punctuation.
function toSegments(text) {
  const segments = [];
  const re = /[^.!?]+[.!?]+/g;
  let last = 0;
  let m;
  while ((m = re.exec(text))) {
    if (m.index > last)
      segments.push({ text: text.slice(last, m.index), isSentence: false });
    segments.push({ text: m[0], isSentence: true });
    last = re.lastIndex;
  }
  if (last < text.length)
    segments.push({ text: text.slice(last), isSentence: false });
  return segments;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function run() {
  // onnxruntime-web config for a Manifest V3 content script:
  //  - load the .wasm from inside the extension (CSP blocks CDNs)
  //  - single-threaded: content-script pages have no SharedArrayBuffer
  //  - no worker proxy: it cannot spawn cross-origin workers here
  ort.env.wasm.wasmPaths = chrome.runtime.getURL("src/");
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;

  let session;
  try {
    session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ["wasm"],
    });
  } catch (e) {
    console.error("[Clarity] failed to load model:", e);
    return;
  }
  const featureNames = await fetch(FEATURE_NAMES_URL).then((r) => r.json());

  // Pass 1: gather every candidate sentence and its features.
  const root = getContentRoot();
  const nodes = collectTextNodes(root);
  const nodePlans = []; // { node, segments, highlight }
  const items = []; // { planIndex, segIndex, features, sentence }

  for (const node of nodes) {
    if (items.length >= MAX_SENTENCES) break;
    const segments = toSegments(node.nodeValue);
    let hasCandidate = false;
    segments.forEach((seg, segIndex) => {
      if (!seg.isSentence) return;
      const features = extractFeatures(seg.text);
      if (!features) return;
      items.push({
        planIndex: nodePlans.length,
        segIndex,
        features,
        sentence: seg.text,
      });
      hasCandidate = true;
    });
    if (hasCandidate) nodePlans.push({ node, segments, highlight: {} });
  }

  if (items.length === 0) return;

  // Pass 2: one batched inference over all sentences.
  const stride = featureNames.length;
  const data = new Float32Array(items.length * stride);
  items.forEach((it, i) => {
    featureNames.forEach((name, j) => {
      data[i * stride + j] = it.features[name];
    });
  });
  const tensor = new ort.Tensor("float32", data, [items.length, stride]);
  const output = await session.run({ float_input: tensor });
  const probsTensor = output.probabilities || output[Object.keys(output).pop()];
  const probs = probsTensor.data; // flat [n*2]: [p0,p1, p0,p1, ...]

  // Pass 2b: document-level salience — importance needs the whole article,
  // not one sentence in isolation.
  const df = new Map();
  for (const it of items) {
    for (const w of new Set(contentTokens(it.sentence))) {
      df.set(w, (df.get(w) || 0) + 1);
    }
  }
  let maxSalience = 0;
  for (const it of items) {
    it.salience = salienceRaw(it.sentence, df);
    if (it.salience > maxSalience) maxSalience = it.salience;
  }

  // Pass 3: combined = importance (salience) modulated by clarity, with
  // clear-but-empty filler penalized.
  let maxCombined = 0;
  items.forEach((it, i) => {
    const clarity = probs[i * 2 + 1];
    const sal = maxSalience > 0 ? it.salience / maxSalience : 0;
    const substantive = countContentWords(it.sentence) >= MIN_CONTENT_WORDS;
    let combined = sal * (0.5 + 0.5 * clarity);
    if (!substantive) combined *= FILLER_PENALTY;
    if (isDefinition(it.sentence)) combined *= DEFINITION_BOOST;
    it.combined = combined;
    if (combined > maxCombined) maxCombined = combined;
  });

  // Pass 3b: highlight the top fraction of sentences by combined score.
  const ranked = [...items].sort((a, b) => b.combined - a.combined);
  const k = Math.min(
    items.length,
    Math.max(MIN_HIGHLIGHTS, Math.round(SELECT_RATIO * items.length)),
    MAX_HIGHLIGHTS
  );
  const floor = RELATIVE_FLOOR * maxCombined;
  for (const it of ranked.slice(0, k)) {
    if (it.combined < floor) continue;
    nodePlans[it.planIndex].highlight[it.segIndex] = it.combined;
  }

  // Pass 4: rebuild only the nodes that got highlights (DOM-safe, no innerHTML).
  for (const plan of nodePlans) {
    const marks = plan.highlight;
    if (Object.keys(marks).length === 0) continue;
    const frag = document.createDocumentFragment();
    plan.segments.forEach((seg, segIndex) => {
      if (marks[segIndex] !== undefined) {
        const span = document.createElement("span");
        span.className = "clarity-highlight";
        span.dataset.score = marks[segIndex].toFixed(2);
        span.textContent = seg.text;
        frag.appendChild(span);
      } else {
        frag.appendChild(document.createTextNode(seg.text));
      }
    });
    plan.node.replaceWith(frag);
  }

  console.log(`[Clarity] scored ${items.length} sentences.`);
}

run();
