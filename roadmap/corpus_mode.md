# Corpus Mode

## Overview

Corpus Mode is a feature that allows the agent to process a large collection of documents (a corpus) efficiently. When the corpus size exceeds a predefined token limit (e.g., 100,000 tokens), the agent employs embedding-based techniques to prioritize and filter the content, ensuring that only the most relevant and non-redundant information is used.

## How it Works

1.  **Corpus Ingestion:** The agent receives a bundle of the whole corpus.
2.  **Size Check:** The agent checks if the total size of the corpus exceeds the token limit.
3.  **Embedding Generation:** If the limit is exceeded, the agent generates embeddings for all documents in the corpus and for the specific task description or query.
4.  **Relevance Sorting:** Documents are sorted based on the similarity of their embeddings to the task embedding. This helps prioritize documents that are most relevant to the current task.
5.  **Redundancy Omission:** The agent identifies and omits parts of documents that are repetitive. This can be done by comparing embeddings of document chunks or sections.
6.  **Threshold Filtering:** Documents or parts of documents that fall below a certain relevance threshold are omitted.
7.  **Token Limit Adherence:** The filtering process continues until the corpus fits within the specified token limit.
8.  **Processing:** The agent then processes the filtered and prioritized corpus.

## Use Cases

-   Analyzing large research paper collections.
-   Processing extensive legal document sets.
-   Extracting insights from large codebases.
-   Summarizing vast amounts of news articles or reports.

## Benefits

-   **Efficiency:** Reduces the amount of data that needs to be processed by the LLM, saving time and computational resources.
-   **Relevance:** Focuses the agent's attention on the most pertinent information for a given task.
-   **Scalability:** Enables the agent to handle very large corpora that would otherwise be too unwieldy.

## Configuration

-   **Token Limit:** The maximum number of tokens the corpus should be reduced to (e.g., `CORPUS_MAX_TOKENS=100000`).
-   **Relevance Threshold:** The minimum similarity score for a document or chunk to be considered relevant.
