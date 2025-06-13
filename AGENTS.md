# Agent Instructions

- Always run the full suite of local checks before creating a pull request:
  - `ruff check .`
  - `pytest -q`
- Include the output of these commands in the PR description under a **Testing** section.
- If any command fails due to missing dependencies or other environment issues, note the failure in the **Testing** section rather than omitting it.

These instructions apply to the entire repository.
