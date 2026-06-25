from pathlib import Path
from collections import Counter, defaultdict
import re

import pandas as pd
import spacy


EXTRACTED_DIR = Path("data/extracted")
OUTPUTS_DIR = Path("outputs")


# These are normal English/function words or slide-extraction artifacts.
CUSTOM_STOPWORDS = {
    "slide",
    "page",
    "presentation",
    "powerpoint",
    "figure",
    "table",
    "example",
    "examples",
    "result",
    "results",
    "problem",
    "solution",
    "reason",
    "conclusion",
    "conclusions",
    "detail",
    "details",
    "use",
    "used",
    "using",
    "show",
    "shows",
    "shown",
    "based",
    "different",
    "important",
    "approach",
    "method",
    "methods",
    "system",
    "systems",
    "input",
    "output",
    "good",
    "better",
    "large",
    "small",
    "new",
    "old",
    "many",
    "much",
    "also",
    "vs",
    "et",
    "al",
}

CUSTOM_STOPWORDS.update(
    {
        "computer",
        "science",
        "department",
        "university",
        "verona",
        "italy",
        "agenda",
        "office",
        "hours",
        "absence",
        "absences",
        "festivities",
        "christmas",
        "pause",
        "appointment",
        "jacob",
        "devlin",
        "google",
        "open",
        "bank",
        "king",
        "queen",
        "crown",
        "man",
        "milk",
        "store",
        "gallon",
        "lalok",
        "voon",
        "izok",
        "wat",
        "sprok",
        "dat",
        "nok",
        "nnat",
        "hihok",
    }
)

CUSTOM_STOPWORDS.update(
    {
        "anok",
        "plok",
        "pippat",
        "rrat",
        "arrat",
        "vat",
        "hilat",
        "wiwok",
        "quat",
        "crrrok",
        "yorok",
        "kantok",
        "yurp",
        "mat",
        "bat",
        "kevin",
        "knight",
        "holger",
        "diessel",
        "med",
        "rio",
        "frasca",
        "katie",
        "xyz",
        "corporation",
        "word1",
        "stemmed1",
        "priorpolarity",
        "len",
        "schedule",
        "issue",
        "today",
    }
)

# These words can be meaningful inside bigrams
GENERIC_UNIGRAMS = {
    "model",
    "models",
    "word",
    "words",
    "task",
    "tasks",
    "data",
    "time",
    "training",
    "train",
    "trained",
    "layer",
    "layers",
    "sequence",
    "sequences",
    "representation",
    "representations",
    "feature",
    "features",
    "text",
    "texts",
    "document",
    "documents",
    "corpus",
}

CUSTOM_STOPWORDS.update(
    {
        "pre",
        "low",
        "case",
        "simple",
        "true",
        "false",
        "set",
        "term",
        "function",
        "number",
        "give",
        "see",
        "let",
        "call",
    }
)

# Short technical terms 
KEEP_SHORT_TERMS = {
    "ai",
    "nlp",
    "lm",
    "llm",
    "qa",
    "ir",
    "pos",
    "ner",
    "ml",
    "dl",
    "t5",
}


# Domain-specific phrases that should be preserved if they appear in text
# This is one shared vocabulary for the whole corpus
DOMAIN_PHRASES = [
    "bag of words",
    "word embeddings",
    "word embedding",
    "language model",
    "language models",
    "large language model",
    "large language models",
    "masked language model",
    "masked lm",
    "next sentence prediction",
    "contextual representation",
    "contextual representations",
    "contextual word representations",
    "self attention",
    "self-attention",
    "multi headed self attention",
    "multi-headed self attention",
    "positional embeddings",
    "wordpiece vocabulary",
    "fine tuning",
    "fine-tuning",
    "pre training",
    "pre-training",
    "knowledge distillation",
    "model compression",
    "text classification",
    "cosine similarity",
    "named entity recognition",
    "dependency grammar",
    "dependency grammars",
    "syntax tree",
    "syntax trees",
    "semantic role",
    "semantic roles",
    "logical representation",
    "information retrieval",
    "statistical natural language processing",
    "natural language processing",
    "machine translation",
    "neural language model",
    "neural language models",
    "transformer encoder",
    "transformer decoder",
    "bidirectional encoder",
    "bidirectional language model",
]


def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm", disable=["ner"])
    except OSError as error:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is not installed. "
            "Run: python -m spacy download en_core_web_sm"
        ) from error


def clean_raw_text(text: str) -> str:
    text = re.sub(r"---\s*(page|slide)\s*\d+\s*---", " ", text, flags=re.IGNORECASE)
    text = text.replace("●", " ")
    text = text.replace("○", " ")
    text = text.replace("■", " ")
    text = text.replace("→", " ")
    text = text.replace("⨉", " ")
    text = text.replace("✕", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_token(token) -> str | None:
    raw = token.text.strip().lower()
    lemma = token.lemma_.strip().lower()

    if not raw:
        return None

    if raw in KEEP_SHORT_TERMS:
        return raw

    if lemma in KEEP_SHORT_TERMS:
        return lemma

    if token.is_stop:
        return None

    if token.is_punct or token.is_space or token.like_num:
        return None

    if not re.search(r"[a-zA-Z]", raw):
        return None

    if len(raw) <= 2:
        return None

    if raw in CUSTOM_STOPWORDS or lemma in CUSTOM_STOPWORDS:
        return None

    # Keep nouns, proper nouns, adjectives and some verbs
    if token.pos_ not in {"NOUN", "PROPN", "ADJ", "VERB"}:
        return None

    return lemma


def extract_phrase_occurrences(text: str) -> list[tuple[str, int]]:
    lowered = text.lower()
    occurrences = []

    for phrase in DOMAIN_PHRASES:
        phrase_pattern = re.escape(phrase.lower())
        phrase_pattern = phrase_pattern.replace(r"\ ", r"\s+")
        for match in re.finditer(phrase_pattern, lowered):
            normalized_phrase = phrase.lower().replace("-", " ")
            normalized_phrase = normalized_phrase.replace("language models", "language model")
            normalized_phrase = normalized_phrase.replace("word embeddings", "word embedding")
            normalized_phrase = normalized_phrase.replace("contextual representations", "contextual representation")
            normalized_phrase = normalized_phrase.replace("semantic roles", "semantic role")
            normalized_phrase = normalized_phrase.replace("neural networks", "neural network")
            normalized_phrase = normalized_phrase.replace("dependency grammars", "dependency grammar")
            normalized_phrase = normalized_phrase.replace("syntax trees", "syntax tree")
            normalized_phrase = re.sub(r"\s+", " ", normalized_phrase).strip()
            occurrences.append((normalized_phrase, match.start()))

    return occurrences


def make_bigrams(clean_terms: list[str]) -> list[tuple[str, int]]:
    bigrams = []

    for index in range(len(clean_terms) - 1):
        first = clean_terms[index]
        second = clean_terms[index + 1]

        if first in CUSTOM_STOPWORDS or second in CUSTOM_STOPWORDS:
            continue

        if first == second:
            continue

        bigram = f"{first} {second}"
        bigrams.append((bigram, index))

    return bigrams


def classify_by_percentiles(scores: pd.Series) -> pd.Series:
    if scores.empty:
        return scores

    low_threshold = scores.quantile(0.10)
    high_threshold = scores.quantile(0.90)

    def classify(score: float) -> str:
        if score >= high_threshold:
            return "top"
        if score <= low_threshold:
            return "bottom"
        return "medium"

    return scores.apply(classify)


def analyze_document(document_id: str, text: str, nlp) -> pd.DataFrame:
    cleaned_text = clean_raw_text(text)
    doc = nlp(cleaned_text)

    clean_terms = []

    for token in doc:
        normalized = normalize_token(token)

        if normalized is not None:
            clean_terms.append(normalized)

    empty_columns = [
        "document_id",
        "term",
        "term_type",
        "frequency",
        "frequency_score",
        "first_position",
        "last_position",
        "distance_from_start",
        "distance_from_end",
        "earliness_score",
        "persistence_score",
        "relevance_score",
        "extended_relevance_score",
        "relevance_class",
    ]

    if not clean_terms:
        return pd.DataFrame(columns=empty_columns)

    term_positions = defaultdict(list)

    # 1. Add unigram terms.
    for index, term in enumerate(clean_terms):
        if term not in GENERIC_UNIGRAMS:
            term_positions[(term, "unigram")].append(index)

    # 2. Add only curated domain phrases
    phrase_occurrences = extract_phrase_occurrences(cleaned_text)
    text_length_chars = max(len(cleaned_text), 1)
    document_length_terms = max(len(clean_terms) - 1, 1)

    for phrase, char_position in phrase_occurrences:
        approximate_token_position = int(
            (char_position / text_length_chars) * document_length_terms
        )

        approximate_token_position = max(
            0,
            min(approximate_token_position, document_length_terms),
        )

        term_positions[(phrase, "domain_phrase")].append(approximate_token_position)

    if not term_positions:
        return pd.DataFrame(columns=empty_columns)

    rows = []
    max_frequency = max(len(positions) for positions in term_positions.values())

    for (term, term_type), positions in term_positions.items():
        frequency = len(positions)

        first_index = min(positions)
        last_index = max(positions)

        first_position = first_index / document_length_terms
        last_position = last_index / document_length_terms

        frequency_score = frequency / max_frequency
        earliness_score = 1 - first_position
        distance_from_start = first_position
        distance_from_end = 1 - last_position
        persistence_score = max(last_position - first_position, 0)

        relevance_score = 0.5 * frequency_score + 0.5 * earliness_score

        extended_relevance_score = (
            0.4 * frequency_score
            + 0.3 * earliness_score
            + 0.3 * persistence_score
        )

        rows.append(
            {
                "document_id": document_id,
                "term": term,
                "term_type": term_type,
                "frequency": frequency,
                "frequency_score": round(frequency_score, 6),
                "first_position": round(first_position, 6),
                "last_position": round(last_position, 6),
                "distance_from_start": round(distance_from_start, 6),
                "distance_from_end": round(distance_from_end, 6),
                "earliness_score": round(earliness_score, 6),
                "persistence_score": round(persistence_score, 6),
                "relevance_score": round(relevance_score, 6),
                "extended_relevance_score": round(extended_relevance_score, 6),
            }
        )

    result = pd.DataFrame(rows)

    result["relevance_class"] = classify_by_percentiles(result["relevance_score"])

    result = result.sort_values(
        by=["relevance_score", "frequency"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return result


def analyze_all_documents(
    extracted_dir: Path = EXTRACTED_DIR,
    outputs_dir: Path = OUTPUTS_DIR,
) -> None:
    outputs_dir.mkdir(parents=True, exist_ok=True)

    text_files = sorted(extracted_dir.glob("*.txt"))

    if not text_files:
        print(f"No .txt files found in {extracted_dir.resolve()}")
        return

    nlp = load_spacy_model()

    all_results = []

    for text_file in text_files:
        document_id = text_file.stem
        text = text_file.read_text(encoding="utf-8", errors="ignore")

        print(f"Analyzing {text_file.name}...")
        document_result = analyze_document(document_id=document_id, text=text, nlp=nlp)
        all_results.append(document_result)

    all_terms = pd.concat(all_results, ignore_index=True)

    all_terms_path = outputs_dir / "term_relevance_all_documents.csv"
    all_terms.to_csv(all_terms_path, index=False, encoding="utf-8")

    top_terms = (
        all_terms
        .sort_values(by=["document_id", "relevance_score"], ascending=[True, False])
        .groupby("document_id")
        .head(20)
        .reset_index(drop=True)
    )

    top_terms_path = outputs_dir / "top_terms_preview.csv"
    top_terms.to_csv(top_terms_path, index=False, encoding="utf-8")

    document_summary = (
        all_terms
        .groupby("document_id")
        .agg(
            unique_terms=("term", "nunique"),
            total_term_occurrences=("frequency", "sum"),
            top_relevance_score=("relevance_score", "max"),
        )
        .reset_index()
        .sort_values(by="document_id")
    )

    summary_path = outputs_dir / "document_summary.csv"
    document_summary.to_csv(summary_path, index=False, encoding="utf-8")

    print()
    print(f"Saved full term table: {all_terms_path}")
    print(f"Saved top terms preview: {top_terms_path}")
    print(f"Saved document summary: {summary_path}")


if __name__ == "__main__":
    analyze_all_documents()