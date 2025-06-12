Of course. This is a critical document for turning our well-defined plan into immediate, actionable work. A great `TODO.md` serves as a public-facing project board, an onboarding tool for new contributors, and a source of truth for our current priorities.

This version is structured like a Kanban board, directly reflects the PoC/MVP sprint plan, and provides clear, detailed tasks that can be broken out into GitHub Issues.

---

# Textura - Project TODO & Roadmap

This document tracks the development progress for Textura, from the initial Proof-of-Concept to the v1.0 release. It is a living document that will be updated as tasks are completed and new priorities emerge.

## 🎯 Sprint 0 Goal: The Proof-of-Concept (PoC)

The goal of this initial sprint is to achieve a **"zero-to-one" proof of concept**: turning a small folder of documents into a minimally viable, linked Obsidian vault. This sprint focuses on validating the core end-to-end pipeline.

### Definition of "Done" for the PoC

A successful PoC means we can run a single script on a sample folder and produce a vault where:
- `Narrative.md` exists and contains ≥ 500 words of synthesized text.
- At least one `_Timelines/YYYY/MM.md` note is created with links to events.
- At least three `_Mysteries/` notes are created with backlinks to their source context.
- `_Meta/log.jsonl` is created and contains ≥ 10 structured log entries.
- The resulting vault opens in Obsidian without broken links.
- The CI pipeline is green on the `main` branch.

---

##  Kanban Board: Current Sprint (PoC/MVP)

### 📋 Backlog (To Be Prioritized)

*   **[EPIC] Implement Full Mystery Investigation Pass:**
    -   `[TASK]` Create Mystery Agent that takes a `Mystery` object as input.
    -   `[TASK]` Perform a vector similarity search against the FAISS index using the mystery's question.
    -   `[TASK]` Feed top-k results to an LLM to synthesize a "lead."
    -   `[TASK]` Append the generated lead to the corresponding `Mystery.md` note.
    -   `[TASK]` Update the mystery's status (e.g., to `INVESTIGATING`).
*   **[EPIC] Implement Robust Human-Override Reconciliation:**
    -   `[TASK]` Refine the `Vault Writer` to perform the hash comparison logic defined in the TDD.
    *   `[TASK]` When a conflict is detected, log the event to `Metacog` with the note ID and a `reconciliation_conflict` message.
    *   `[TASK]` Ensure the writer correctly skips overwriting the user-edited file.
*   **[FEATURE] Implement `--stats` Flag:**
    -   `[TASK]` Create a stats collector that aggregates data from the `Metacog` log at the end of a run.
    -   `[TASK]` Format the stats into a clean, human-readable summary table.
    -   `[TASK]` Print the summary to the console when the `--stats` flag is used.
*   **[FEATURE] Implement Narrative Agent:**
    -   `[TASK]` Design a prompt that queries the `Timeline Builder`'s output.
    -   `[TASK]` Synthesize a high-level `Narrative.md` that summarizes key periods and links to the `_Timelines/` notes.
*   **[FEATURE] Implement ID Generator & Date Fallback:**
    -   `[TASK]` Create a utility that assigns a consistent CID to each nugget using its `YYYY-MM-DD` date when available.
    -   `[TASK]` When a nugget has no date, prefix the CID with the current run date and mark `date_inferred: true` (**BR-005**).

### ⏳ To Do (This Week's Priority)

*   **[CHORE] Initialize Repository & Scaffolding (#1):**
    -   `[SUBTASK]` Create `requirements.txt` with pinned versions (`uv pip freeze > requirements.txt`).
    -   `[SUBTASK]` Add `.editorconfig` and `.gitignore`.
    -   `[SUBTASK]` Set up `pyproject.toml` with `ruff` and `mypy` configurations.
    -   `[SUBTASK]` Create `LICENSE` (MIT) and `CONTRIBUTING.md` files.
*   **[FEATURE] Implement CLI Skeleton (`init`, `ingest`, `weave`) (#2):**
    -   `[SUBTASK]` Use `click` or `argparse` to create the main `textura` entry point.
    -   `[SUBTASK]` Define the `init`, `ingest`, and `weave` commands with their core arguments (`--workspace`, `--source`).
    -   `[SUBTASK]` Implement the `init` command to create the `textura_workspace/` directory structure.
*   **[FEATURE] Implement Core Ingestion Pipeline (`ingest` command) (#3):**
    -   `[SUBTASK]` Implement `Source Watcher` to read files and check against `manifest.json`.
    -   `[SUBTASK]` Implement `Chunker` using LlamaIndex readers (`pymupdf`).
    -   `[SUBTASK]` Implement `Embedder` to generate BGE embeddings for chunks.
    -   `[SUBTASK]` Wire up `FAISSVectorStore` to save and persist the index to `workspace/faiss.index`.
    -   `[SUBTASK]` Purge stale chunks from the vector store before re-indexing a modified document (**BR-003**).
*   **[FEATURE] Implement Extractor Agent (First Pass) (#4):**
    -   `[SUBTASK]` Draft the "Schema-First Extraction" prompt and JSON schema as a Pydantic model (`EventV1`, `MysteryV1`, etc.).
    -   `[SUBTASK]` Create an agent that iterates through chunks from the vector store.
    -   `[SUBTASK]` For each chunk, call the LLM (Gemini) and validate the JSON output against the Pydantic models.
    -   `[SUBTASK]` Log the raw JSON output and any validation errors to `Metacog`.
*   **[FEATURE] Implement `Vault Writer` & `Timeline Builder` (#5):**
    -   `[SUBTASK]` Create a `writer` module that takes Pydantic objects (e.g., `EventV1`) and renders them into Markdown strings.
    -   `[SUBTASK]` The writer must include the `<!-- TEXTURA:_v1 {...} -->` metadata comment in every generated file.
    -   `[SUBTASK]` Create a `Timeline Builder` that groups extracted events by date and generates `_Timelines/YYYY.md` and `_Timelines/YYYY-MM.md` files with wikilinks.
*   **[CHORE] Set up Initial CI Pipeline (#6):**
    -   `[SUBTASK]` Create a GitHub Actions workflow that runs on push/PR to `main`.
    -   `[SUBTASK]` The workflow must run `ruff check`, `mypy .`, and `pytest`.
    -   `[SUBTASK]` Set up a mock LLM service in `tests/` to prevent real API calls during CI.

### 🚧 In Progress

*(Move tasks from "To Do" here when you start working on them.)*

### ✅ Done

*(Move tasks here upon completion and merge to `main`.)*

---

## 📚 Future Sprints (Post-MVP Backlog)

This is a holding area for important features and epics that are out of scope for the initial MVP but are on the long-term roadmap.

*   **[EPIC] Local Model Support:**
    -   Integrate with Ollama or LM Studio as an alternative to remote APIs.
    -   Add configuration for local model names and API endpoints.
*   **[EPIC] Advanced PDF Processing:**
    -   Integrate a proper OCR library (e.g., Tesseract via `pytesseract`) for image-based PDFs.
    -   Add support for extracting tables and figures as distinct notes.
*   **[EPIC] Advanced Caching & Performance:**
    -   Implement a more sophisticated caching layer for LLM calls (e.g., using Redis or a local SQLite DB).
    -   Parallelize the chunking and embedding processes.
*   **[FEATURE] Pluggable Architecture:**
    -   Refactor core components (`Reader`, `Embedder`, `VectorStore`) to use formal plugin base classes.
*   **[FEATURE] GUI / Obsidian Plugin:**
    -   A simple status panel inside Obsidian to show Textura's last run, stats, and a "Re-weave" button.
*   **[FEATURE] Web Content Ingestion:**
    -   Add a `textura ingest --url <URL>` command to process web pages.

---

Feel free to pick up a task from the `To Do` list, create a new GitHub Issue referencing it, and start a discussion
