from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


OUTPUTS_DIR = Path("outputs")

TERM_RELEVANCE_PATH = OUTPUTS_DIR / "term_relevance_all_documents.csv"
DOCUMENT_SIMILARITY_PATH = OUTPUTS_DIR / "document_similarity.csv"
REFERENCE_EDGES_PATH = OUTPUTS_DIR / "reference_edges.csv"
REFERENCE_GRAPH_IMAGE_PATH = OUTPUTS_DIR / "reference_graph.png"


TOP_TERMS_PER_DOCUMENT = 80
MIN_SHARED_CONCEPTS = 2
MIN_EDGE_SCORE = 0.11
MAX_TOTAL_EDGES = 10
MAX_SHARED_CONCEPTS_DISPLAY = 6

MAX_EDGES_PER_SOURCE = 2
MAX_TOTAL_EDGES = 16

MAX_DOCUMENT_FREQUENCY_FRACTION = 0.70

EXCLUDED_DOCUMENTS = {"L0 Introduction"}


def natural_document_sort_key(document_id: str) -> tuple[int, str]:
    match = re.search(r"L(\d+)", document_id, flags=re.IGNORECASE)

    if match:
        return int(match.group(1)), document_id.lower()

    return 9999, document_id.lower()


def load_term_relevance(path: Path = TERM_RELEVANCE_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path}. Run preprocessing first:\n"
            f"python .\\src\\preprocessing.py"
        )

    df = pd.read_csv(path)

    required_columns = {
        "document_id",
        "term",
        "term_type",
        "frequency",
        "relevance_score",
        "extended_relevance_score",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns in {path}: {sorted(missing_columns)}")

    df["document_id"] = df["document_id"].astype(str).str.strip()
    df["term"] = df["term"].astype(str).str.strip()

    df = df[~df["document_id"].isin(EXCLUDED_DOCUMENTS)].copy()

    df = df[df["term"].str.len() >= 3].copy()
    df = df[~df["term"].str.fullmatch(r"[a-zA-Z]")].copy()

    number_of_documents = df["document_id"].nunique()

    document_frequency = (
        df.groupby("term")["document_id"]
        .nunique()
        .rename("document_frequency")
        .reset_index()
    )

    df = df.merge(document_frequency, on="term", how="left")

    max_document_frequency = max(
        2,
        int(number_of_documents * MAX_DOCUMENT_FREQUENCY_FRACTION),
    )

    df = df[df["document_frequency"] >= 2].copy()
    df = df[df["document_frequency"] <= max_document_frequency].copy()

    df["idf"] = np.log(
        (1 + number_of_documents) / (1 + df["document_frequency"])
    ) + 1

    df["phrase_boost"] = np.where(df["term_type"] == "domain_phrase", 1.25, 1.0)

    df["concept_weight"] = (
        df["extended_relevance_score"].astype(float)
        * df["idf"].astype(float)
        * df["phrase_boost"].astype(float)
    )

    return df


def select_top_terms(
    df: pd.DataFrame,
    top_n: int = TOP_TERMS_PER_DOCUMENT,
) -> pd.DataFrame:
    top_terms = (
        df.sort_values(
            by=[
                "document_id",
                "concept_weight",
                "extended_relevance_score",
                "frequency",
            ],
            ascending=[True, False, False, False],
        )
        .groupby("document_id")
        .head(top_n)
        .reset_index(drop=True)
    )

    return top_terms


def build_document_term_matrix(top_terms: pd.DataFrame) -> pd.DataFrame:
    matrix = top_terms.pivot_table(
        index="document_id",
        columns="term",
        values="concept_weight",
        aggfunc="max",
        fill_value=0.0,
    )

    ordered_documents = sorted(matrix.index.tolist(), key=natural_document_sort_key)
    matrix = matrix.loc[ordered_documents]

    return matrix


def compute_similarity_matrix(document_term_matrix: pd.DataFrame) -> pd.DataFrame:
    similarity_values = cosine_similarity(document_term_matrix.values)

    similarity_df = pd.DataFrame(
        similarity_values,
        index=document_term_matrix.index,
        columns=document_term_matrix.index,
    )

    return similarity_df


def weighted_jaccard_for_documents(
    source_vector: pd.Series,
    target_vector: pd.Series,
) -> float:
    source_values = source_vector.values.astype(float)
    target_values = target_vector.values.astype(float)

    numerator = np.minimum(source_values, target_values).sum()
    denominator = np.maximum(source_values, target_values).sum()

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def get_shared_concepts(
    source_vector: pd.Series,
    target_vector: pd.Series,
    max_items: int = MAX_SHARED_CONCEPTS_DISPLAY,
) -> list[str]:
    shared_terms = []

    for term in source_vector.index:
        source_weight = float(source_vector[term])
        target_weight = float(target_vector[term])

        if source_weight > 0 and target_weight > 0:
            shared_weight = min(source_weight, target_weight)
            shared_terms.append((term, shared_weight))

    shared_terms = sorted(shared_terms, key=lambda item: item[1], reverse=True)

    return [term for term, _ in shared_terms[:max_items]]


def build_reference_edges(
    document_term_matrix: pd.DataFrame,
    similarity_df: pd.DataFrame,
    min_edge_score: float = MIN_EDGE_SCORE,
) -> pd.DataFrame:
    documents = list(document_term_matrix.index)
    rows: list[dict[str, Any]] = []

    for source_index, source_document in enumerate(documents):
        for target_index, target_document in enumerate(documents):
            if source_index >= target_index:
                continue

            source_vector = document_term_matrix.loc[source_document]
            target_vector = document_term_matrix.loc[target_document]

            cosine_score = float(similarity_df.loc[source_document, target_document])
            overlap_score = weighted_jaccard_for_documents(source_vector, target_vector)

            reference_score = (0.75 * cosine_score) + (0.25 * overlap_score)

            shared_concepts = get_shared_concepts(source_vector, target_vector)

            if (
                reference_score >= min_edge_score
                and len(shared_concepts) >= MIN_SHARED_CONCEPTS
            ):
                rows.append(
                    {
                        "source_document": source_document,
                        "target_document": target_document,
                        "reference_score": round(reference_score, 6),
                        "cosine_similarity": round(cosine_score, 6),
                        "weighted_overlap": round(overlap_score, 6),
                        "shared_concepts": ", ".join(shared_concepts),
                        "num_shared_concepts": len(shared_concepts),
                    }
                )

    edges_df = pd.DataFrame(rows)

    if edges_df.empty:
        return pd.DataFrame(
            columns=[
                "source_document",
                "target_document",
                "reference_score",
                "cosine_similarity",
                "weighted_overlap",
                "shared_concepts",
                "num_shared_concepts",
            ]
        )

    edges_df = edges_df.sort_values(
        by=["reference_score", "cosine_similarity", "num_shared_concepts"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    edges_df = (
        edges_df
        .groupby("source_document", group_keys=False)
        .head(MAX_EDGES_PER_SOURCE)
        .sort_values(
            by=["reference_score", "cosine_similarity", "num_shared_concepts"],
            ascending=[False, False, False],
        )
        .head(MAX_TOTAL_EDGES)
        .reset_index(drop=True)
    )

    return edges_df


def save_similarity_table(similarity_df: pd.DataFrame) -> None:
    similarity_table = similarity_df.reset_index().rename(
        columns={"index": "document_id"}
    )

    similarity_table.to_csv(
        DOCUMENT_SIMILARITY_PATH,
        index=False,
        encoding="utf-8",
    )


def save_reference_edges(edges_df: pd.DataFrame) -> None:
    edges_df.to_csv(
        REFERENCE_EDGES_PATH,
        index=False,
        encoding="utf-8",
    )


def draw_reference_graph(edges_df: pd.DataFrame) -> None:
    if edges_df.empty:
        print("No edges to draw.")
        return

    graph = nx.DiGraph()

    for _, row in edges_df.iterrows():
        source = row["source_document"]
        target = row["target_document"]
        score = float(row["reference_score"])

        graph.add_edge(source, target, weight=score)

    plt.figure(figsize=(14, 8))

    pos = nx.spring_layout(
        graph,
        seed=42,
        k=1.3,
        iterations=150,
    )

    edge_weights = [
        graph[source][target]["weight"]
        for source, target in graph.edges()
    ]

    edge_widths = [
        1.0 + 5.0 * weight
        for weight in edge_weights
    ]

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=1700,
        alpha=0.9,
    )

    nx.draw_networkx_labels(
        graph,
        pos,
        font_size=9,
        font_weight="bold",
    )

    nx.draw_networkx_edges(
        graph,
        pos,
        width=edge_widths,
        arrows=True,
        arrowsize=18,
        alpha=0.6,
        connectionstyle="arc3,rad=0.08",
    )

    edge_labels = {
        (source, target): f"{graph[source][target]['weight']:.2f}"
        for source, target in graph.edges()
    }

    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_size=8,
    )

    plt.title("Conceptual Reference Graph Between Lecture Documents")
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(REFERENCE_GRAPH_IMAGE_PATH, dpi=200)
    plt.close()


def print_top_edges(edges_df: pd.DataFrame, top_n: int = 15) -> None:
    if edges_df.empty:
        print("No reference edges found.")
        return

    print()
    print(f"Top {min(top_n, len(edges_df))} reference edges:")
    print()

    for _, row in edges_df.head(top_n).iterrows():
        print(
            f"{row['source_document']} -> {row['target_document']} "
            f"| score={row['reference_score']} "
            f"| shared: {row['shared_concepts']}"
        )


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading term relevance table...")
    term_relevance_df = load_term_relevance()

    print("Selecting top terms per document...")
    top_terms_df = select_top_terms(term_relevance_df)

    print("Building document-term matrix...")
    document_term_matrix = build_document_term_matrix(top_terms_df)

    print("Computing document similarity...")
    similarity_df = compute_similarity_matrix(document_term_matrix)

    print("Building directed reference edges...")
    edges_df = build_reference_edges(
        document_term_matrix=document_term_matrix,
        similarity_df=similarity_df,
    )

    print("Saving outputs...")
    save_similarity_table(similarity_df)
    save_reference_edges(edges_df)

    print("Drawing graph image...")
    draw_reference_graph(edges_df)

    print()
    print(f"Saved document similarity: {DOCUMENT_SIMILARITY_PATH}")
    print(f"Saved reference edges: {REFERENCE_EDGES_PATH}")

    if not edges_df.empty:
        print(f"Saved reference graph image: {REFERENCE_GRAPH_IMAGE_PATH}")

    print_top_edges(edges_df)


if __name__ == "__main__":
    main()