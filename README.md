# Textura

![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Works with Obsidian](https://img.shields.io/badge/Obsidian-Ready-purple.svg?logo=obsidian)

**Textura** is your open-source, AI-powered weaver for turning messy document collections into a structured, narrative-driven Obsidian knowledge garden.

---

## 🌟 Vision

> Every researcher can cultivate a living knowledge garden that learns and reflects alongside them.

---

## 💡 The Problem Textura Solves

*(Here, we'll distill the "Problem of Knowledge" section from our TDD into a concise, impactful narrative. Focus on the pain points: "digital landfill," "manual note-taking bottleneck," "opaque AI tools.")*

Today's researchers face a paradox: abundant information, but a severe bottleneck in turning it into usable knowledge. Traditional PKM tools demand immense manual effort, while AI solutions are often opaque "black boxes." Textura bridges this gap.

---

## ✨ How Textura Transforms Your Documents

*(This is where we highlight the core differentiators: Narrative, Mystery, Metacognition. Explain them simply but powerfully.)*

Textura uniquely combines the power of AI with the transparency of a personal knowledge management system. It's not just a note-taker; it's a **metacognitive agent** that:

*   **Weaves a Narrative:** Automatically structures your corpus into a chronological story, organized by year and month notes, with an overarching `Narrative.md` entry point.
*   **Hunts for Mysteries:** Identifies unanswered questions and ambiguities within your documents, flagging them as `Mystery` notes. On subsequent runs, Textura actively searches for clues, helping you pinpoint knowledge gaps.
*   **Explains Itself (Metacognition):** Generates a `_Meta/` folder, documenting its own architecture, prompt history, and decision-making process. This provides unprecedented transparency, allowing you to audit and trust the AI's work.
*   **Obsidian-Native Output:** Produces clean, interlinked Markdown notes, ready to be opened directly in Obsidian with zero special plugins.

---

## 🚀 Key Features

*   **Narrative Generation:** Chronological stories directly from your documents.
*   **Mystery Detection & Investigation:** Turns unknowns into active research prompts.
*   **Self-Documenting Agent (`_Meta/`):** Understand Textura's reasoning and evolution.
*   **Atomic Note Creation:** Extracts key facts and ideas into individual, linked Markdown notes.
*   **Temporal Organization:** Auto-generates Year and Month notes, acting as living timelines.
*   **Automated Linking:** Intelligent bidirectional linking (`[[wikilinks]]`) between notes.
*   **Human-Override Friendly:** Preserves your manual edits in auto-generated notes.
*   **Incremental Processing:** Efficiently processes only new or changed files on subsequent runs.
*   **Local-First & Private:** Your data stays on your machine unless you explicitly configure remote LLM APIs.

---

## 🛠️ Getting Started (Proof of Concept)

This section outlines how to get the very first version of Textura up and running.

### Prerequisites

*   Python 3.12+
*   A Google Cloud Project with the Gemini API enabled (for LLM access). You will need `GOOGLE_APPLICATION_CREDENTIALS` or a direct API key.
*   Obsidian (optional, but recommended to view the output).

### Installation

```bash
# Clone the repository
git clone https://github.com/franklinbaldo/textura.git
cd textura

# Create a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: `venv\Scripts\activate`
pip install -r requirements.txt
```

### Basic Usage

1.  **Initialize your Textura Workspace:**
    This creates the `textura_workspace/` directory which will store all internal data and your Obsidian vault.

    ```bash
    textura init --workspace ./my_textura_project
    ```

2.  **Ingest your Documents:**
    Point Textura to your source folder. It will read, chunk, embed, and index your documents.

    ```bash
    textura ingest --workspace ./my_textura_project --source /path/to/your/documents
    ```

3.  **Weave Your Knowledge Garden:**
    This is the core step. Textura will process the indexed documents, extract entities, events, mysteries, and generate the narrative. The output will be in `./my_textura_project/vault/`.

    ```bash
    textura weave --workspace ./my_textura_project
    ```

4.  **Explore in Obsidian:**
    Open the `./my_textura_project/vault/` folder as a new vault in Obsidian!

---

## ⚙️ Configuration

Textura uses environment variables for configuration.

| Environment Variable | Default | Description |
| :------------------- | :------ | :---------- |
| `TEXTURA_LLM_MODEL`  | `gemini-1.5-pro` | The LLM model to use. |
| `TEXTURA_EMBED_MODEL` | `BAAI/bge-base-en-v1.5` | The embedding model to use. |
| `TEXTURA_VECTOR_BACKEND` | `faiss` | The vector store backend (`faiss` or `milvus-lite`). |
| `TEXTURA_CHUNK_SIZE` | `1024` | Max token count for text chunks. |
| `TEXTURA_CHUNK_OVERLAP` | `128` | Overlap in tokens between chunks. |

---

## 🛣️ Roadmap & Contributing

Textura is in its early stages. We have an ambitious roadmap towards a v1.0 release.

*   **Phase 1 (PoC):** Core ingestion, basic extraction, and narrative generation.
*   **Phase 2 (MVP):** Full Mystery workflow, improved metacognition, and human-override detection.
*   **Phase 3 (Alpha):** Enhanced PDF handling, more robust error recovery, and community feedback integration.

We welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines, and check our [Issues](https://github.com/franklinbaldo/textura/issues) for open tasks.

---

## 🛡️ Security & Privacy

Textura is designed with privacy in mind. Your documents are processed locally by default. LLM API keys are handled securely via environment variables and are never stored persistently.

---

## 📜 License

Textura is released under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

*   Built with [LlamaIndex](https://www.llamaindex.ai/)
*   Leverages [Obsidian](https://obsidian.md/) for knowledge visualization
*   Inspired by the incredible work in LLMs and PKM communities

