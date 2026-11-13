"""Original Urdu linguistic rules reproduced directly from the supplied Assignment 1 notebook.
Functions are A1-authored, not pretrained. Preserve original behavior for reproducibility.
Known shortcomings: some suffix stemming is too aggressive and percentages may remain literal.
"""
import re

ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u08F0-\u08FF]")

def remove_diacritics(text):
    return ARABIC_DIACRITICS.sub("", text)

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")

EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E0-\U0001F1FF"
    "]+", flags=re.UNICODE
)

ZERO_WIDTH_RE = re.compile(r"[\u200B\u200C\u200D\uFEFF]")

LATIN_RE = re.compile(r"[A-Za-z]+")

URDU_KEEP_RE = re.compile(
    r"[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF"
    r"\u06F0-\u06F90-9\s۔،؟!:٪\-]"
)

def remove_noise(text):
    text = URL_RE.sub(" ", text)
    text = EMOJI_RE.sub(" ", text)
    text = ZERO_WIDTH_RE.sub(" ", text)
    text = LATIN_RE.sub(" ", text)
    return text

def remove_non_urdu(text):
    return URDU_KEEP_RE.sub(" ", text)

SENTENCE_SPLIT_RE = re.compile(r"([^۔؟!]*[۔؟!])")

def segment_sentences(text):
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.findall(text) if s.strip()]
    remainder = SENTENCE_SPLIT_RE.sub("", text).strip()
    if remainder:
        sentences.append(remainder)
    return sentences

WHITESPACE_RE = re.compile(r"\s+")

def normalize_whitespace(text):
    return WHITESPACE_RE.sub(" ", text).strip()

PUNCTUATION = set("۔،؟!:٪-()[]{}\"'")

POSTPOSITIONS = {"کا", "کی", "کے", "میں", "پر", "سے", "تک", "کو", "نے", "ہی", "بھی", "تو", "لیے", "والا", "والی", "والے"}

NUM_RE = re.compile(r"^[0-9\u06F0-\u06F9]+$")

def urdu_tokenize(text):
    tokens = []
    for chunk in text.split():
        if not chunk:
            continue
        # peel off leading punctuation
        while chunk and chunk[0] in PUNCTUATION:
            tokens.append(chunk[0])
            chunk = chunk[1:]
        # peel off trailing punctuation (may be more than one, e.g. word۔)
        trailing = []
        while chunk and chunk[-1] in PUNCTUATION:
            trailing.append(chunk[-1])
            chunk = chunk[:-1]
        if chunk:
            if NUM_RE.match(chunk):
                tokens.append("<NUM>")
            else:
                tokens.append(chunk)
        tokens.extend(reversed(trailing))
    return tokens

URDU_SUFFIXES = sorted(
    ["وں", "یں", "ات", "ائیں", "ائی", "گی", "گا", "گے", "تا", "تی", "تے", "ہا", "یاں", "ے", "ی", "ہ"],
    key=len, reverse=True,
)

def urdu_stem(word):
    if word in {"<NUM>"} or word in PUNCTUATION or word in POSTPOSITIONS:
        return word
    for suf in URDU_SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 2:
            return word[: -len(suf)]
    return word

LEMMA_DICT = {
    "لڑکے": "لڑکا", "لڑکوں": "لڑکا",
    "لوگوں": "لوگ", "بچوں": "بچہ", "بچے": "بچہ",
    "کتابیں": "کتاب", "کتابوں": "کتاب",
    "شہروں": "شہر", "علاقوں": "علاقہ",
    "کھلاڑیوں": "کھلاڑی", "کسانوں": "کسان",
    "مریضوں": "مریض", "شرکاء": "شریک",
}

def urdu_lemmatize(word):
    if word in {"<NUM>"} or word in PUNCTUATION or word in POSTPOSITIONS:
        return word
    if word in LEMMA_DICT:
        return LEMMA_DICT[word]
    # regular oblique-plural pattern: X + وں  ->  X + ا   (e.g. گھروں -> گھرا is imperfect; fall back safely)
    if word.endswith("وں") and len(word) > 3:
        return word[:-2]
    if word.endswith("یاں") and len(word) > 3:
        return word[:-3] + "ی"
    return word

def clean_and_process_article(raw_text):
    text = remove_diacritics(raw_text)
    text = remove_noise(text)
    text = remove_non_urdu(text)
    text = normalize_whitespace(text)
    sentences = segment_sentences(text)

    processed_sentences = []
    for sent in sentences:
        tokens = urdu_tokenize(sent)
        tokens = [urdu_stem(t) for t in tokens]
        tokens = [urdu_lemmatize(t) for t in tokens]
        if tokens:
            processed_sentences.append(" ".join(tokens))
    return processed_sentences