import streamlit as st

st.set_page_config(
    page_title="NLP Document Analysis Tool",
    page_icon="📄",
    layout="wide"
)

st.title("NLP Document Analysis Tool")

st.write(
    """
    This app will analyse documents using an NLP pipeline:
    stopword removal, lemmatization, frequency analysis,
    positional analysis, relevance scoring, and document reference graphs.
    """
)

uploaded_files = st.file_uploader(
    "Upload documents",
    type=["txt", "pdf", "pptx"],
    accept_multiple_files=True
)

if uploaded_files:
    st.subheader("Uploaded files")
    for file in uploaded_files:
        st.write(f"- {file.name}")

st.info("Setup is working. Next step: implement text extraction.")
