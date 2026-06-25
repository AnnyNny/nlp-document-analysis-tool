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

DOCUMENT_CONCEPT_WEIGHTS_PATH = OUTPUTS_DIR / "document_concept_weights.csv"
DOCUMENT_SIMILARITY_PATH = OUTPUTS_DIR / "document_similarity.csv"

REFERENCE_EDGES_ALL_PATH = OUTPUTS_DIR / "reference_edges_all.csv"
REFERENCE_EDGES_PATH = OUTPUTS_DIR / "reference_edges.csv"
REFERENCE_GRAPH_IMAGE_PATH = OUTPUTS_DIR / "reference_graph.png"

EXPLORATORY_EDGES_PATH = OUTPUTS_DIR / "reference_edges_exploratory.csv"
EXPLORATORY_GRAPH_IMAGE_PATH = OUTPUTS_DIR / "reference_graph_exploratory.png"

AGENDA_EDGES_PATH = OUTPUTS_DIR / "agenda_edges.csv"
AGENDA_GRAPH_IMAGE_PATH = OUTPUTS_DIR / "agenda_graph.png"


INTRO_DOCUMENT_ID = "L0 Introduction"


# Main graph.
PRESENTATION_INCLUDE_INTRODUCTION = False
PRESENTATION_TOP_TERMS_PER_DOCUMENT = 80
PRESENTATION_MIN_EDGE_SCORE = 0.11
PRESENTATION_MIN_SHARED_CONCEPTS = 2
PRESENTATION_MAX_EDGES_PER_SOURCE = 2
PRESENTATION_MAX_TOTAL_EDGES = 10


# Exploratory graph: denser graph for analysis.
EXPLORATORY_INCLUDE_INTRODUCTION = True
EXPLORATORY_TOP_TERMS_PER_DOCUMENT = 100
EXPLORATORY_MIN_EDGE_SCORE = 0.08
EXPLORATORY_MIN_SHARED_CONCEPTS = 2
EXPLORATORY_MAX_EDGES_PER_SOURCE = 4
EXPLORATORY_MAX_TOTAL_EDGES = 25


# Agenda graph: special use of L0.
AGENDA_TOP_TERMS_PER_DOCUMENT = 120
AGENDA_MIN_EDGE_SCORE = 0.04
AGENDA_MAX_TOTAL_EDGES = 15


MAX_SHARED_CONCEPTS_DISPLAY = 8

MIN_DOCUMENT_FREQUENCY = 2
MAX_DOCUMENT_FREQUENCY_FRACTION = 0.75


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
    df["term_type"] = df["term_type"].astype(str).str.strip()

    df["frequency"] = pd.to_numeric(df["frequency"], errors="coerce").fillna(0)
    df["relevance_score"] = pd.to_numeric(
        df["relevance_score"],
        errors="coerce",
    ).fillna(0.0)
    df["extended_relevance_score"] = pd.to_numeric(
        df["extended_relevance_score"],
        errors="coerce",
    ).fillna(0.0)

    df = df[df["term"].str.len() >= 3].copy()
    df = df[~df["term"].str.fullmatch(r"[a-zA-Z]")].copy()

    return df


def compute_concept_weights(
    df: pd.DataFrame,
    include_introduction: bool,
    max_document_frequency_fraction: float = MAX_DOCUMENT_FREQUENCY_FRACTION,
) -> pd.DataFrame:
    working_df = df.copy()

    if not include_introduction:
        working_df = working_df[
            working_df["document_id"] != INTRO_DOCUMENT_ID
        ].copy()

    number_of_documents = working_df["document_id"].nunique()

    if number_of_documents == 0:
        return working_df

    document_frequency = (
        working_df.groupby("term")["document_id"]
        .nunique()
        .rename("document_frequency")
        .reset_index()
    )

    working_df = working_df.merge(document_frequency, on="term", how="left")

    max_document_frequency = max(
        MIN_DOCUMENT_FREQUENCY,
        int(number_of_documents * max_document_frequency_fraction),
    )

    working_df = working_df[
        working_df["document_frequency"] >= MIN_DOCUMENT_FREQUENCY
    ].copy()

    working_df = working_df[
        working_df["document_frequency"] <= max_document_frequency
    ].copy()

    working_df["idf"] = (
        np.log(
            (1 + number_of_documents)
            / (1 + working_df["document_frequency"].astype(float))
        )
        + 1
    )

    working_df["phrase_boost"] = np.where(
        working_df["term_type"] == "domain_phrase",
        1.25,
        1.0,
    )

    working_df["concept_weight"] = (
        working_df["extended_relevance_score"].astype(float)
        * working_df["idf"].astype(float)
        * working_df["phrase_boost"].astype(float)
    )

    working_df = working_df.sort_values(
        by=[
            "document_id",
            "term",
            "concept_weight",
            "extended_relevance_score",
            "frequency",
        ],
        ascending=[True, True, False, False, False],
    )

    working_df = working_df.drop_duplicates(
        subset=["document_id", "term"],
        keep="first",
    ).reset_index(drop=True)

    return working_df


def select_top_terms(
    concept_df: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    top_terms = (
        concept_df.sort_values(
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


def build_all_reference_edges(
    document_term_matrix: pd.DataFrame,
    similarity_df: pd.DataFrame,
    min_edge_score: float,
    min_shared_concepts: int,
    reverse_direction: bool = False,
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
                and len(shared_concepts) >= min_shared_concepts
            ):
                if reverse_direction:
                    edge_source = target_document
                    edge_target = source_document
                    edge_interpretation = "target_refers_back_to_source"
                else:
                    edge_source = source_document
                    edge_target = target_document
                    edge_interpretation = "source_concepts_propagate_to_target"

                rows.append(
                    {
                        "source_document": edge_source,
                        "target_document": edge_target,
                        "reference_score": round(reference_score, 6),
                        "cosine_similarity": round(cosine_score, 6),
                        "weighted_overlap": round(overlap_score, 6),
                        "shared_concepts": ", ".join(shared_concepts),
                        "num_shared_concepts": len(shared_concepts),
                        "interpretation": edge_interpretation,
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "source_document",
                "target_document",
                "reference_score",
                "cosine_similarity",
                "weighted_overlap",
                "shared_concepts",
                "num_shared_concepts",
                "interpretation",
            ]
        )

    edges_df = pd.DataFrame(rows)

    edges_df = edges_df.sort_values(
        by=["reference_score", "cosine_similarity", "num_shared_concepts"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return edges_df


def select_edges_for_graph(
    edges_df: pd.DataFrame,
    max_edges_per_source: int,
    max_total_edges: int,
) -> pd.DataFrame:
    if edges_df.empty:
        return edges_df

    selected_edges = (
        edges_df
        .sort_values(
            by=["reference_score", "cosine_similarity", "num_shared_concepts"],
            ascending=[False, False, False],
        )
        .groupby("source_document", group_keys=False)
        .head(max_edges_per_source)
        .sort_values(
            by=["reference_score", "cosine_similarity", "num_shared_concepts"],
            ascending=[False, False, False],
        )
        .head(max_total_edges)
        .reset_index(drop=True)
    )

    return selected_edges


def build_agenda_edges(
    document_term_matrix: pd.DataFrame,
    similarity_df: pd.DataFrame,
    min_edge_score: float,
    max_total_edges: int,
) -> pd.DataFrame:
    if INTRO_DOCUMENT_ID not in document_term_matrix.index:
        return pd.DataFrame(
            columns=[
                "source_document",
                "target_document",
                "reference_score",
                "cosine_similarity",
                "weighted_overlap",
                "shared_concepts",
                "num_shared_concepts",
                "interpretation",
            ]
        )

    rows = []
    intro_vector = document_term_matrix.loc[INTRO_DOCUMENT_ID]

    for target_document in document_term_matrix.index:
        if target_document == INTRO_DOCUMENT_ID:
            continue

        target_vector = document_term_matrix.loc[target_document]

        cosine_score = float(similarity_df.loc[INTRO_DOCUMENT_ID, target_document])
        overlap_score = weighted_jaccard_for_documents(intro_vector, target_vector)

        reference_score = (0.75 * cosine_score) + (0.25 * overlap_score)

        shared_concepts = get_shared_concepts(intro_vector, target_vector)

        if reference_score >= min_edge_score and shared_concepts:
            rows.append(
                {
                    "source_document": INTRO_DOCUMENT_ID,
                    "target_document": target_document,
                    "reference_score": round(reference_score, 6),
                    "cosine_similarity": round(cosine_score, 6),
                    "weighted_overlap": round(overlap_score, 6),
                    "shared_concepts": ", ".join(shared_concepts),
                    "num_shared_concepts": len(shared_concepts),
                    "interpretation": "agenda_topic_appears_in_later_lecture",
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "source_document",
                "target_document",
                "reference_score",
                "cosine_similarity",
                "weighted_overlap",
                "shared_concepts",
                "num_shared_concepts",
                "interpretation",
            ]
        )

    agenda_edges = pd.DataFrame(rows)

    agenda_edges = (
        agenda_edges
        .sort_values(
            by=["reference_score", "cosine_similarity", "num_shared_concepts"],
            ascending=[False, False, False],
        )
        .head(max_total_edges)
        .reset_index(drop=True)
    )

    return agenda_edges


def save_similarity_table(similarity_df: pd.DataFrame, path: Path) -> None:
    similarity_table = similarity_df.reset_index().rename(
        columns={"index": "document_id"}
    )

    similarity_table.to_csv(path, index=False, encoding="utf-8")


def draw_reference_graph(
    edges_df: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    if edges_df.empty:
        print(f"No edges to draw for {output_path}.")
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
        k=1.4,
        iterations=180,
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

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(output_path, dpi=200)
    plt.close()


def print_edges(title: str, edges_df: pd.DataFrame, top_n: int = 12) -> None:
    print()
    print(title)

    if edges_df.empty:
        print("No edges found.")
        return

    print(f"Top {min(top_n, len(edges_df))} edges:")
    print()

    for _, row in edges_df.head(top_n).iterrows():
        print(
            f"{row['source_document']} -> {row['target_document']} "
            f"| score={row['reference_score']} "
            f"| shared: {row['shared_concepts']}"
        )


def run_content_graph(
    term_relevance_df: pd.DataFrame,
    include_introduction: bool,
    top_terms_per_document: int,
    min_edge_score: float,
    min_shared_concepts: int,
    max_edges_per_source: int,
    max_total_edges: int,
    all_edges_path: Path,
    selected_edges_path: Path,
    graph_path: Path,
    similarity_path: Path | None,
    graph_title: str,
) -> pd.DataFrame:
    concept_df = compute_concept_weights(
        term_relevance_df,
        include_introduction=include_introduction,
    )

    top_terms = select_top_terms(
        concept_df,
        top_n=top_terms_per_document,
    )

    document_term_matrix = build_document_term_matrix(top_terms)

    similarity_df = compute_similarity_matrix(document_term_matrix)

    all_edges = build_all_reference_edges(
        document_term_matrix=document_term_matrix,
        similarity_df=similarity_df,
        min_edge_score=min_edge_score,
        min_shared_concepts=min_shared_concepts,
        reverse_direction=False,
    )

    selected_edges = select_edges_for_graph(
        edges_df=all_edges,
        max_edges_per_source=max_edges_per_source,
        max_total_edges=max_total_edges,
    )

    all_edges.to_csv(all_edges_path, index=False, encoding="utf-8")
    selected_edges.to_csv(selected_edges_path, index=False, encoding="utf-8")

    if similarity_path is not None:
        save_similarity_table(similarity_df, similarity_path)

    draw_reference_graph(
        edges_df=selected_edges,
        output_path=graph_path,
        title=graph_title,
    )

    return selected_edges


def run_agenda_graph(term_relevance_df: pd.DataFrame) -> pd.DataFrame:
    concept_df = compute_concept_weights(
        term_relevance_df,
        include_introduction=True,
        max_document_frequency_fraction=0.95,
    )

    top_terms = select_top_terms(
        concept_df,
        top_n=AGENDA_TOP_TERMS_PER_DOCUMENT,
    )

    document_term_matrix = build_document_term_matrix(top_terms)
    similarity_df = compute_similarity_matrix(document_term_matrix)

    agenda_edges = build_agenda_edges(
        document_term_matrix=document_term_matrix,
        similarity_df=similarity_df,
        min_edge_score=AGENDA_MIN_EDGE_SCORE,
        max_total_edges=AGENDA_MAX_TOTAL_EDGES,
    )

    agenda_edges.to_csv(AGENDA_EDGES_PATH, index=False, encoding="utf-8")

    draw_reference_graph(
        edges_df=agenda_edges,
        output_path=AGENDA_GRAPH_IMAGE_PATH,
        title="introduction topics in later lectures",
    )

    return agenda_edges


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading term relevance table...")
    term_relevance_df = load_term_relevance()

    print("Computing concept weights for inspection...")
    concept_weights = compute_concept_weights(
        term_relevance_df,
        include_introduction=True,
        max_document_frequency_fraction=0.95,
    )

    concept_weights.to_csv(
        DOCUMENT_CONCEPT_WEIGHTS_PATH,
        index=False,
        encoding="utf-8",
    )

    print("Building presentation graph...")
    presentation_edges = run_content_graph(
        term_relevance_df=term_relevance_df,
        include_introduction=PRESENTATION_INCLUDE_INTRODUCTION,
        top_terms_per_document=PRESENTATION_TOP_TERMS_PER_DOCUMENT,
        min_edge_score=PRESENTATION_MIN_EDGE_SCORE,
        min_shared_concepts=PRESENTATION_MIN_SHARED_CONCEPTS,
        max_edges_per_source=PRESENTATION_MAX_EDGES_PER_SOURCE,
        max_total_edges=PRESENTATION_MAX_TOTAL_EDGES,
        all_edges_path=REFERENCE_EDGES_ALL_PATH,
        selected_edges_path=REFERENCE_EDGES_PATH,
        graph_path=REFERENCE_GRAPH_IMAGE_PATH,
        similarity_path=DOCUMENT_SIMILARITY_PATH,
        graph_title="conceptual reference graph between lecture documents",
    )

    print("Building exploratory graph...")
    exploratory_edges = run_content_graph(
        term_relevance_df=term_relevance_df,
        include_introduction=EXPLORATORY_INCLUDE_INTRODUCTION,
        top_terms_per_document=EXPLORATORY_TOP_TERMS_PER_DOCUMENT,
        min_edge_score=EXPLORATORY_MIN_EDGE_SCORE,
        min_shared_concepts=EXPLORATORY_MIN_SHARED_CONCEPTS,
        max_edges_per_source=EXPLORATORY_MAX_EDGES_PER_SOURCE,
        max_total_edges=EXPLORATORY_MAX_TOTAL_EDGES,
        all_edges_path=OUTPUTS_DIR / "reference_edges_exploratory_all.csv",
        selected_edges_path=EXPLORATORY_EDGES_PATH,
        graph_path=EXPLORATORY_GRAPH_IMAGE_PATH,
        similarity_path=None,
        graph_title="exploratory conceptual reference graph",
    )

    print("Building agenda graph...")
    agenda_edges = run_agenda_graph(term_relevance_df)

    print()
    print(f"Saved concept weights: {DOCUMENT_CONCEPT_WEIGHTS_PATH}")
    print(f"Saved document similarity: {DOCUMENT_SIMILARITY_PATH}")
    print(f"Saved all candidate presentation edges: {REFERENCE_EDGES_ALL_PATH}")
    print(f"Saved selected presentation edges: {REFERENCE_EDGES_PATH}")
    print(f"Saved presentation graph image: {REFERENCE_GRAPH_IMAGE_PATH}")
    print(f"Saved exploratory edges: {EXPLORATORY_EDGES_PATH}")
    print(f"Saved exploratory graph image: {EXPLORATORY_GRAPH_IMAGE_PATH}")
    print(f"Saved agenda edges: {AGENDA_EDGES_PATH}")
    print(f"Saved agenda graph image: {AGENDA_GRAPH_IMAGE_PATH}")

    print_edges("Presentation graph", presentation_edges)
    print_edges("Exploratory graph", exploratory_edges)
    print_edges("Agenda graph", agenda_edges)


if __name__ == "__main__":
    main()