# Assignment report

## 1. Project choice

The project follows the research-oriented assignment 2: implementing a graphical tool for document processing. It also includes an extension taken from assignment 3 and relates documents to each other by using a reference graph. 
This project covers all the following assignment 2 subgoals:
- Eliminate stopwords; 
- Lemmatize terms; 
- Compute frequencies; 
- Measure distances from strategic points (start and end); 
- Compute compound relevance indices (50% frequency and 50% earliness).

## 2. Corpus choice

As a corpus, I chose all the lecture slides of the course 'Natural Language Processing', because has a good property for my research: the concepts are introduced in earlier lectures and then reappear in the later chapters of the course. The ordering makes the corpus suitable for studying term relevance and conceptual references between the documents.


## 3. Text extraction

The project analyzes `.pptx`, `.pdf`, and `.txt` files.

Initially, the first extractor was too restrictive since it produced almost empty text files for the PowerPoint lectures. So it was simplified to directly read text from shapes, tables and slide objects. After this change the extracted lecture files have realistic word counts.

This step was important because the quality of the whole analysis depends on the quality of the extracted text.

## 4. Term representation

The system uses two term types: a 'single term' and a 'domain phrase'. Single trm is a lemmatized unigram and domain phrase is a list of manually selected phrases, based on the course content (for example, 'language model' or 'word embedding'). I also tried the extraction of bigrams but did not keep it in the final version, because it produced noisy examples of bigrams and the random adjacency would not be meaningful for representing the concepts.         | 

For this reason, the final pipeline uses single lemmatized terms plus selected multi-word keyphrases.


## 5. Graph modes

The final app contains three graphs: presentation graph, exploratory graph and agenda graph. The presentation graph is showing a clean reference graph for report. Exploratory graph is a denser version for analysis and the agenda graph is showing how topics from the `L0 Introduction` document appear in later documents. `L0 Introduction` is treated separately because it is an agenda document, not a normal content lecture. If included in the same graph as all other lectures, it tends to connect to too many nodes.

## 6. Project pipeline

The project extracts the texts from the slides and saves them into .txt files. Then it applies basic NLP preprocessing and then builds the graphical tool which helps to understand how words in each document are important for it and then to see how different documents are rellated to each other in a graph. 

Stopwords elimination

Articles, prepositions, generic words are removed. This is to filter out the words that are frequent but do not convey additional meaning to understand the lecture. I use spaCy stopwords and also add a custom list based on the words found in previous iterations of the program.

Lemmatization

Different forms of the lowcase single terms are tokenized by spaCy, which reads each token and gives a basic form of the word. Lemmatization is done after filtering stopwords, punctuation, number, short tokens and custom stopwords. 

Frequency computation

After cleaning the program calculates how many times each word appears in each document.
Then frquency is normalized:
frequency_score = frequency / max_frequency_in_document

Distance measurement

For every term, the code stores all positions where this term appears. Then it computes the first occurrence and the last occurrence of the term. These positions are normalized with respect to the document length, so they are represented on a scale from 0 to 1. A value close to 0 means that the term appears near the beginning of the document, while a value close to 1 means that it appears near the end. From these values, the system computes `distance_from_start` and `distance_from_end`. 

earliness_score = 1 - first_position
persistence_score = last_position - first_position

Final scoring

The frequency component is represented by frequency_score, which is a normalized version of the raw term frequency. The positional component is represented by earliness_score, which is higher when the term first appears closer to the beginning of the document.

relevance_score = 0.5 * frequency_score + 0.5 * earliness_score

Reference graph

Each document is represented by its most relevant concepts. For each concept, the code computes a weight:

concept_weight = extended_relevance_score * idf * phrase_boost

extended_relevance_score shows how important the term is inside one document. idf reduces the weight of terms that appear in too many documents. phrase_boost gives a small additional weight to domain phrases, such as language model or word embedding.

Then the code compares documents using their weighted concept vectors:

similarity_matrix = cosine_similarity(document_term_matrix)

If a later document is sufficiently similar to an earlier document and they share important concepts, the system creates a directed edge:

source_document -> target_document

## Results

The final project produces both tabular results and graphs.

Example output columns are:

```python
[
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
    "extended_relevance_score"
]
```

The results identify meaningful terms for different lectures. For example, lectures about language modelling contain terms such as `language model`, `bigram`; lectures about vector semantics contain terms such as `embedding`, `similarity`,`word embedding`; and lectures about sentiment analysis have terms related to `sentiment`, `emotion` and `classification`.

The reference graph produced connections between lectures. Some examples of conceptual links are:

L5 -> L7   logistic regression / neural network concepts
L7 -> L10  neural language models / BERT and contextual representations
L8 -> L11  POS tagging / dependency parsing
L12 -> L14 predicate and argument structure / semantic roles
L4 -> L15  classification and sentiment

The Streamlit interface allows the user to select a document, view the most relevant terms, compare relevance scores, inspect the similarity matrix and explore the generated reference graphs. The app includes three graph modes: a clean presentation graph, a denser exploratory graph and a separate agenda graph for `L0 Introduction`.

The resulting project can process a corpus of lecture documents, compute interpretable relevance scores and build an exploratory conceptual graph of the course.


## 9. Limitations

1. Frequency does not guarantee conceptual importance.
2. Early occurrence does not always mean centrality.
3. Lecture slides are semi-structured and contain diagrams, fragments, examplesand repeated labels.
5. Some noisy terms may remain because the corpus is extracted from slides.

