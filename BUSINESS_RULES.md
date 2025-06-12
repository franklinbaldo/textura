# Textura - Agent Logic & Business Rules

This document serves as the single source of truth for the core decision-making logic and operational heuristics of the Textura agent. Every rule listed here must have a corresponding test case in the test suite.

## 1. Core Ingestion & Data Integrity Rules

| Rule ID | Rule Name | Rule Description (Plain English) | Implementation Notes & Location | Test Case(s) |
| :--- | :--- | :--- | :--- | :--- |
| **BR-001** | **Incremental Ingestion Trigger** | A source file will be re-ingested only if it is new or if its SHA256 content hash has changed since the last successful run. | `watcher.py` checks against `manifest.json`. | `test_watcher_detects_new_file()`, `test_watcher_detects_modified_file()`, `test_watcher_ignores_unchanged_file()` |
| **BR-002** | **Human Override Preservation** | An auto-generated note in the vault must NOT be overwritten if the content of the note (above the metadata comment) has been modified by a human. | `writer.py` performs a content hash check against the hash stored in the `<!-- TEXTURA:_v1 -->` comment before writing. | `test_writer_overwrites_unchanged_note()`, `test_writer_skips_modified_note()` |
| **BR-003** | **Stale Data Purge** | When a source file is re-ingested (due to modification), all of its previous `Chunks` and associated data must be purged from the vector store before new data is inserted. | `ingest.py` must call a `vector_store.delete_by_doc_id()` method before inserting new nodes. | `test_ingest_purges_stale_chunks()` |

## 2. Nugget Extraction & Identification Rules

| Rule ID | Rule Name | Rule Description (Plain English) | Implementation Notes & Location | Test Case(s) |
| :--- | :--- | :--- | :--- | :--- |
| **BR-004** | **Event Validity** | To be classified as an `Event`, a nugget of information must contain a specific or clearly implied date (e.g., "last week," "in 2023"). Events without dates are ignored. | The `Extractor Agent` prompt will instruct the LLM to only extract events with temporal context. The Pydantic `EventV1` model requires a `date` field. | `test_extractor_identifies_dated_event()`, `test_extractor_ignores_dateless_claim()` |
| **BR-005** | **Nugget Dating Fallback** | If a Nugget (like an `Event`) has a date that can be resolved to `YYYY-MM-DD`, that date is used for its `CID`. If a Nugget has no date, its `CID` will be prefixed with the date of the current `run`, and its provenance will be flagged with `date_inferred: true`. | The `ID Generator` utility will handle this logic. | `test_cid_generation_with_date()`, `test_cid_generation_fallback()` |
| **BR-006** | **Mystery Identification Trigger** | A statement is flagged as a potential `Mystery` if it contains explicit language of uncertainty. The initial keyword list is: "unknown," "unclear," "unresolved," "uncertain," "no reason was given," "further investigation is needed." | The `Extractor Agent` prompt will specifically look for these patterns. | `test_extractor_flags_mystery_on_keyword()`, `test_extractor_ignores_certain_statement()` |

## 3. Weaving & Synthesis Rules

| Rule ID | Rule Name | Rule Description (Plain English) | Implementation Notes & Location | Test Case(s) |
| :--- | :--- | :--- | :--- | :--- |
| **BR-007** | **Mystery Investigation Scope** | When investigating a `Mystery`, the agent will perform a vector similarity search and retrieve the top 5 most relevant `Chunks` from the entire corpus as potential context. | The `Mystery Agent` will query the vector store with `k=5`. | `test_mystery_agent_retrieves_k_chunks()` |
| **BR-008** | **Timeline Note Creation** | A `_Timelines/YYYY.md` note is created for any year in which at least one `Event` occurs. A `_Timelines/YYYY-MM.md` note is created for any month in which at least one `Event` occurs. | The `Timeline Builder` will iterate through all extracted events and create notes on demand. | `test_timeline_builder_creates_year_and_month_notes()` |

## 4. Additional Operational Guidelines

| Rule ID | Rule Name | Rule Description (Plain English) | Implementation Notes & Location | Test Case(s) |
| :--- | :--- | :--- | :--- | :--- |
| **BR-009** | **Markdown for Agent Status** | Agent status, todos, and memory should be tracked in plain `.md` files whenever possible. Avoid other document formats. | Backlog and status notes live in Markdown files like `TODO.md`. | `test_markdown_status_files()` |
| **BR-010** | **Minimal PDF Operations** | Heavy PDF processing is discouraged; rely on Gemini's built-in PDF handling rather than custom OCR or parsing logic. | Ingestion uses simple readers (e.g., `pymupdf`) without advanced PDF manipulation. | `test_ingest_handles_pdf_simple()` |
| **BR-011** | **No NLTK Dependency** | Natural language tasks should not use `nltk`; Gemini is preferred. | `requirements.txt` excludes `nltk` and code relies on Gemini's capabilities. | `test_no_nltk_imports()` |
