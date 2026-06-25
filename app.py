from pathlib import Path
import re

import pandas as pd
import streamlit as st


OUTPUTS_DIR = Path("outputs")

DOCUMENT_SUMMARY_PATH = OUTPUTS_DIR / "document_summary.csv"
TERM_RELEVANCE_PATH = OUTPUTS_DIR / "term_relevance_all_documents.csv"
TOP_TERMS_PATH = OUTPUTS_DIR / "top_terms_preview.csv"

DOCUMENT_CONCEPT_WEIGHTS_PATH = OUTPUTS_DIR / "document_concept_weights.csv"
DOCUMENT_SIMILARITY_PATH = OUTPUTS_DIR / "document_similarity.csv"

REFERENCE_EDGES_ALL_PATH = OUTPUTS_DIR / "reference_edges_all.csv"
REFERENCE_EDGES_PATH = OUTPUTS_DIR / "reference_edges.csv"
REFERENCE_GRAPH_IMAGE_PATH = OUTPUTS_DIR / "reference_graph.png"

EXPLORATORY_EDGES_PATH = OUTPUTS_DIR / "reference_edges_exploratory.csv"
EXPLORATORY_GRAPH_IMAGE_PATH = OUTPUTS_DIR / "reference_graph_exploratory.png"

AGENDA_EDGES_PATH = OUTPUTS_DIR / "agenda_edges.csv"
AGENDA_GRAPH_IMAGE_PATH = OUTPUTS_DIR / "agenda_graph.png"


st.set_page_config(
    page_title="NLP Document Analysis Tool",
    page_icon="📄",
    layout="wide",
)


def natural_document_sort_key(document_id: str) -> tuple[int, str]:
    match = re.search(r"L(\d+)", str(document_id), flags=re.IGNORECASE)

    if match:
        return int(match.group(1)), str(document_id).lower()

    return 9999, str(document_id).lower()


def load_csv(path: Path, required: bool = True) -> pd.DataFrame | None:
    if not path.exists():
        if required:
            st.error(f"Missing file: `{path}`")
            st.stop()
        return None

    return pd.read_csv(path)


def file_status(path: Path) -> str:
    if path.exists():
        return "available"
    return "missing"


def readable_term_type(series: pd.Series) -> pd.Series:
    return series.replace(
        {
            "unigram": "single term",
            "domain_phrase": "multi-word keyphrase",
        }
    )


def show_download_button(df: pd.DataFrame, filename: str, label: str) -> None:
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
    )


def get_existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


st.title("NLP Document Analysis Tool")

st.write(
    """
    This graphical tool analyses a corpus of NLP lecture documents.
    It extracts terms and keyphrases, computes relevance scores, measures positional
    information, and builds document-to-document reference graphs.
    """
)

with st.sidebar:
    st.header("Pipeline status")

    status_rows = [
        {"file": "document_summary.csv", "status": file_status(DOCUMENT_SUMMARY_PATH)},
        {"file": "term_relevance_all_documents.csv", "status": file_status(TERM_RELEVANCE_PATH)},
        {"file": "top_terms_preview.csv", "status": file_status(TOP_TERMS_PATH)},
        {"file": "document_concept_weights.csv", "status": file_status(DOCUMENT_CONCEPT_WEIGHTS_PATH)},
        {"file": "document_similarity.csv", "status": file_status(DOCUMENT_SIMILARITY_PATH)},
        {"file": "reference_edges.csv", "status": file_status(REFERENCE_EDGES_PATH)},
        {"file": "reference_graph.png", "status": file_status(REFERENCE_GRAPH_IMAGE_PATH)},
        {"file": "reference_edges_exploratory.csv", "status": file_status(EXPLORATORY_EDGES_PATH)},
        {"file": "reference_graph_exploratory.png", "status": file_status(EXPLORATORY_GRAPH_IMAGE_PATH)},
        {"file": "agenda_edges.csv", "status": file_status(AGENDA_EDGES_PATH)},
        {"file": "agenda_graph.png", "status": file_status(AGENDA_GRAPH_IMAGE_PATH)},
    ]

    status_df = pd.DataFrame(status_rows)
    st.dataframe(status_df, use_container_width=True, hide_index=True)

    st.markdown(
        """
        To regenerate outputs:

        ```powershell
        python .\\src\\text_extraction.py
        python .\\src\\preprocessing.py
        python .\\src\\reference_graph.py
        ```
        """
    )


document_summary = load_csv(DOCUMENT_SUMMARY_PATH)
term_relevance = load_csv(TERM_RELEVANCE_PATH)
top_terms = load_csv(TOP_TERMS_PATH)

concept_weights = load_csv(DOCUMENT_CONCEPT_WEIGHTS_PATH, required=False)
document_similarity = load_csv(DOCUMENT_SIMILARITY_PATH, required=False)

reference_edges = load_csv(REFERENCE_EDGES_PATH, required=False)
reference_edges_all = load_csv(REFERENCE_EDGES_ALL_PATH, required=False)
exploratory_edges = load_csv(EXPLORATORY_EDGES_PATH, required=False)
agenda_edges = load_csv(AGENDA_EDGES_PATH, required=False)


document_summary["document_id"] = document_summary["document_id"].astype(str)
term_relevance["document_id"] = term_relevance["document_id"].astype(str)
term_relevance["term"] = term_relevance["term"].astype(str)
top_terms["document_id"] = top_terms["document_id"].astype(str)
top_terms["term"] = top_terms["term"].astype(str)

if "term_type" in term_relevance.columns:
    term_relevance["term_type_readable"] = readable_term_type(term_relevance["term_type"])

if "term_type" in top_terms.columns:
    top_terms["term_type_readable"] = readable_term_type(top_terms["term_type"])


tab_overview, tab_relevance, tab_concepts, tab_similarity, tab_graphs, tab_edges, tab_method = st.tabs(
    [
        "Corpus overview",
        "Relevance analysis",
        "Concept weights",
        "Similarity matrix",
        "Reference graphs",
        "Edge explorer",
        "Method",
    ]
)


with tab_overview:
    st.subheader("Corpus overview")

    ordered_summary = document_summary.copy()
    ordered_summary = ordered_summary.sort_values(
        by="document_id",
        key=lambda column: column.map(natural_document_sort_key),
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Documents", ordered_summary["document_id"].nunique())

    if "unique_terms" in ordered_summary.columns:
        col2.metric("Total unique term rows", int(ordered_summary["unique_terms"].sum()))
    else:
        col2.metric("Total term rows", len(term_relevance))

    if "total_term_occurrences" in ordered_summary.columns:
        col3.metric("Total term occurrences", int(ordered_summary["total_term_occurrences"].sum()))
    else:
        col3.metric("Total term occurrences", int(term_relevance["frequency"].sum()))

    if reference_edges is not None:
        col4.metric("Presentation graph edges", len(reference_edges))
    else:
        col4.metric("Presentation graph edges", 0)

    st.dataframe(ordered_summary, use_container_width=True, hide_index=True)

    st.subheader("Document size by extracted term occurrences")

    if "total_term_occurrences" in ordered_summary.columns:
        chart_df = ordered_summary[["document_id", "total_term_occurrences"]].copy()
        chart_df = chart_df.set_index("document_id")
        st.bar_chart(chart_df)

    show_download_button(
        ordered_summary,
        "document_summary.csv",
        "Download document summary",
    )


with tab_relevance:
    st.subheader("Term relevance analysis")

    document_ids = sorted(
        term_relevance["document_id"].unique(),
        key=natural_document_sort_key,
    )

    selected_document = st.selectbox(
        "Choose a document",
        document_ids,
        key="relevance_document_selector",
    )

    selected_terms = term_relevance[
        term_relevance["document_id"] == selected_document
    ].copy()

    if "term_type_readable" not in selected_terms.columns and "term_type" in selected_terms.columns:
        selected_terms["term_type_readable"] = readable_term_type(selected_terms["term_type"])

    available_term_types = sorted(selected_terms["term_type_readable"].unique())

    selected_term_types = st.multiselect(
        "Term types",
        available_term_types,
        default=available_term_types,
    )

    selected_terms = selected_terms[
        selected_terms["term_type_readable"].isin(selected_term_types)
    ].copy()

    score_column = st.radio(
        "Score to visualize",
        ["relevance_score", "extended_relevance_score"],
        horizontal=True,
    )

    top_k = st.slider(
        "Number of terms to show",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
    )

    selected_terms = selected_terms.sort_values(
        by=score_column,
        ascending=False,
    )

    st.write(f"Top terms for **{selected_document}** by `{score_column}`")

    chart_data = selected_terms.head(top_k)[["term", score_column]].copy()
    chart_data = chart_data.set_index("term")
    st.bar_chart(chart_data)

    columns_to_show = [
        "term",
        "term_type_readable",
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

    st.dataframe(
        selected_terms[get_existing_columns(selected_terms, columns_to_show)].head(100),
        use_container_width=True,
        hide_index=True,
    )

    show_download_button(
        selected_terms,
        f"{selected_document}_term_relevance.csv",
        "Download selected document terms",
    )


with tab_concepts:
    st.subheader("Concept weights used for graph construction")

    if concept_weights is None:
        st.warning("Concept weights file is missing. Run `python .\\src\\reference_graph.py`.")
    else:
        concept_weights["document_id"] = concept_weights["document_id"].astype(str)
        concept_weights["term"] = concept_weights["term"].astype(str)

        document_ids_for_concepts = sorted(
            concept_weights["document_id"].unique(),
            key=natural_document_sort_key,
        )

        selected_concept_document = st.selectbox(
            "Choose a document",
            document_ids_for_concepts,
            key="concept_document_selector",
        )

        selected_concept_weights = concept_weights[
            concept_weights["document_id"] == selected_concept_document
        ].copy()

        selected_concept_weights = selected_concept_weights.sort_values(
            by="concept_weight",
            ascending=False,
        )

        st.write(
            """
            Concept weight is used for graph construction. It combines extended relevance,
            inverse document frequency, and a small boost for multi-word keyphrases.
            """
        )

        concept_chart = selected_concept_weights.head(25)[["term", "concept_weight"]].copy()
        concept_chart = concept_chart.set_index("term")
        st.bar_chart(concept_chart)

        columns_to_show = [
            "term",
            "term_type",
            "frequency",
            "extended_relevance_score",
            "document_frequency",
            "idf",
            "phrase_boost",
            "concept_weight",
        ]

        st.dataframe(
            selected_concept_weights[get_existing_columns(selected_concept_weights, columns_to_show)].head(100),
            use_container_width=True,
            hide_index=True,
        )

        show_download_button(
            concept_weights,
            "document_concept_weights.csv",
            "Download all concept weights",
        )


with tab_similarity:
    st.subheader("Document similarity matrix")

    if document_similarity is None:
        st.warning("Document similarity file is missing. Run `python .\\src\\reference_graph.py`.")
    else:
        similarity_df = document_similarity.copy()

        st.dataframe(similarity_df, use_container_width=True, hide_index=True)

        document_column = "document_id"

        if document_column in similarity_df.columns:
            similarity_documents = sorted(
                similarity_df[document_column].astype(str).tolist(),
                key=natural_document_sort_key,
            )

            selected_similarity_document = st.selectbox(
                "Show most similar documents to",
                similarity_documents,
                key="similarity_document_selector",
            )

            row = similarity_df[
                similarity_df[document_column].astype(str) == selected_similarity_document
            ]

            if not row.empty:
                similarity_scores = row.drop(columns=[document_column]).T.reset_index()
                similarity_scores.columns = ["document_id", "similarity"]
                similarity_scores["similarity"] = pd.to_numeric(
                    similarity_scores["similarity"],
                    errors="coerce",
                )
                similarity_scores = similarity_scores[
                    similarity_scores["document_id"] != selected_similarity_document
                ]
                similarity_scores = similarity_scores.sort_values(
                    by="similarity",
                    ascending=False,
                )

                st.write(f"Most similar documents to **{selected_similarity_document}**")

                similarity_chart = similarity_scores.head(10).set_index("document_id")
                st.bar_chart(similarity_chart)

                st.dataframe(
                    similarity_scores.head(20),
                    use_container_width=True,
                    hide_index=True,
                )

        show_download_button(
            similarity_df,
            "document_similarity.csv",
            "Download similarity matrix",
        )


with tab_graphs:
    st.subheader("Reference graph visualizations")

    graph_mode = st.radio(
        "Choose graph mode",
        [
            "Presentation graph",
            "Exploratory graph",
            "Agenda graph",
        ],
        horizontal=True,
    )

    if graph_mode == "Presentation graph":
        st.markdown(
            """
            **Presentation graph** excludes `L0 Introduction` and keeps only the strongest
            conceptual links between content lectures. This is the cleanest graph for the report.
            """
        )

        if reference_edges is not None:
            st.dataframe(reference_edges, use_container_width=True, hide_index=True)
            show_download_button(reference_edges, "reference_edges.csv", "Download presentation edges")
        else:
            st.warning("Presentation edges are missing.")

        if REFERENCE_GRAPH_IMAGE_PATH.exists():
            st.image(str(REFERENCE_GRAPH_IMAGE_PATH), caption="Presentation reference graph")
        else:
            st.warning("Presentation graph image is missing.")

    elif graph_mode == "Exploratory graph":
        st.markdown(
            """
            **Exploratory graph** includes more candidate links and can include `L0 Introduction`.
            It is useful for analysis, but it may be visually denser.
            """
        )

        if exploratory_edges is not None:
            st.dataframe(exploratory_edges, use_container_width=True, hide_index=True)
            show_download_button(exploratory_edges, "reference_edges_exploratory.csv", "Download exploratory edges")
        else:
            st.warning("Exploratory edges are missing.")

        if EXPLORATORY_GRAPH_IMAGE_PATH.exists():
            st.image(str(EXPLORATORY_GRAPH_IMAGE_PATH), caption="Exploratory reference graph")
        else:
            st.warning("Exploratory graph image is missing.")

    else:
        st.markdown(
            """
            **Agenda graph** treats `L0 Introduction` as a special curriculum node.
            It shows which later lectures reuse concepts announced in the introduction.
            """
        )

        if agenda_edges is not None:
            st.dataframe(agenda_edges, use_container_width=True, hide_index=True)
            show_download_button(agenda_edges, "agenda_edges.csv", "Download agenda edges")
        else:
            st.warning("Agenda edges are missing.")

        if AGENDA_GRAPH_IMAGE_PATH.exists():
            st.image(str(AGENDA_GRAPH_IMAGE_PATH), caption="Agenda coverage graph")
        else:
            st.warning("Agenda graph image is missing.")


with tab_edges:
    st.subheader("Edge explorer")

    edge_source = st.radio(
        "Choose edge table",
        [
            "Selected presentation edges",
            "All candidate presentation edges",
            "Exploratory edges",
            "Agenda edges",
        ],
        horizontal=True,
    )

    if edge_source == "Selected presentation edges":
        edges_df = reference_edges
    elif edge_source == "All candidate presentation edges":
        edges_df = reference_edges_all
    elif edge_source == "Exploratory edges":
        edges_df = exploratory_edges
    else:
        edges_df = agenda_edges

    if edges_df is None:
        st.warning("Selected edge table is missing.")
    elif edges_df.empty:
        st.info("Selected edge table is empty.")
    else:
        edges_df = edges_df.copy()
        edges_df["source_document"] = edges_df["source_document"].astype(str)
        edges_df["target_document"] = edges_df["target_document"].astype(str)

        source_options = ["All"] + sorted(
            edges_df["source_document"].unique(),
            key=natural_document_sort_key,
        )

        selected_source = st.selectbox("Source document", source_options)

        filtered_edges = edges_df.copy()

        if selected_source != "All":
            filtered_edges = filtered_edges[
                filtered_edges["source_document"] == selected_source
            ].copy()

        min_score = float(filtered_edges["reference_score"].min())
        max_score = float(filtered_edges["reference_score"].max())

        if min_score < max_score:
            score_threshold = st.slider(
                "Minimum reference score",
                min_value=min_score,
                max_value=max_score,
                value=min_score,
            )

            filtered_edges = filtered_edges[
                filtered_edges["reference_score"] >= score_threshold
            ].copy()

        filtered_edges = filtered_edges.sort_values(
            by="reference_score",
            ascending=False,
        )

        st.dataframe(filtered_edges, use_container_width=True, hide_index=True)

        if not filtered_edges.empty:
            st.subheader("Strongest edge explanation")

            strongest_edge = filtered_edges.iloc[0]

            st.markdown(
                f"""
                **{strongest_edge["source_document"]} → {strongest_edge["target_document"]}**

                Reference score: `{strongest_edge["reference_score"]}`

                Shared concepts:

                `{strongest_edge["shared_concepts"]}`
                """
            )

        show_download_button(
            filtered_edges,
            "filtered_reference_edges.csv",
            "Download filtered edges",
        )


with tab_method:
    st.subheader("Method")

    st.markdown(
        """
        The project implements a graphical NLP pipeline for document processing.

        The pipeline contains the following stages:

        1. **Text extraction** from `.pptx`, `.pdf`, and `.txt` files.
        2. **Stopword removal**.
        3. **Lemmatization**.
        4. **Frequency computation**.
        5. **Distance measurement** from the beginning and end of each document.
        6. **Compound relevance scoring**.
        7. **Concept weighting** for document-level analysis.
        8. **Document similarity computation**.
        9. **Reference graph construction**.

        The assignment relevance score is:

        ```
        relevance_score =
            0.5 * frequency_score
            + 0.5 * earliness_score
        ```

        The extended relevance score adds persistence:

        ```
        extended_relevance_score =
            0.4 * frequency_score
            + 0.3 * earliness_score
            + 0.3 * persistence_score
        ```

        The graph uses a data-driven concept weight:

        ```
        concept_weight =
            extended_relevance_score
            * idf
            * phrase_boost
        ```

        A directed edge from document A to document B means that B is later in the
        ordered corpus and reuses relevant concepts that appeared in A.

        The graph does not claim to detect explicit citations. It approximates
        conceptual reference through shared relevant concepts and document similarity.
        """
    )