import click
import os
import json
from pathlib import Path
from typing import List, Union # Added Union

from textura.ingestion.source_watcher import SourceWatcher
from textura.ingestion.chunker import Chunker
from textura.ingestion.embedder import Embedder
from textura.ingestion.vector_store import FAISSVectorStore
from textura.extraction.models import ExtractionItem, EventV1, MysteryV1 # Added EventV1, MysteryV1
from textura.extraction.extractor_agent import ExtractorAgent
from textura.logging.metacog import Metacog
from textura.logging.stats_collector import StatsCollector
from textura.vault.vault_writer import VaultWriter # Added
from textura.vault.timeline_builder import TimelineBuilder # Added


@click.group()
def textura_cli():
    """A CLI tool for managing textual data workflows."""
    pass

@textura_cli.command()
@click.option('--workspace', default='textura_workspace', help='Path to the Textura workspace.', show_default=True)
def init(workspace: str):
    """Initializes a new Textura workspace."""
    workspace_path = Path(workspace).resolve()
    click.echo(f"Initializing Textura workspace at {workspace_path}")
    try:
        # Core directories
        dirs_to_create = [
            workspace_path,
            workspace_path / 'config',
            workspace_path / 'data' / 'raw',
            workspace_path / 'data' / 'processed',
            workspace_path / 'models',
            workspace_path / 'pipelines',
            workspace_path / 'notebooks',
            workspace_path / 'logs',
            workspace_path / 'index',
            workspace_path / 'vault' / 'notes' / 'events', # For VaultWriter
            workspace_path / 'vault' / 'notes' / 'mysteries', # For VaultWriter
            workspace_path / 'vault' / 'timelines', # For TimelineBuilder
        ]
        for d in dirs_to_create:
            d.mkdir(parents=True, exist_ok=True)

        _ = SourceWatcher(source_dir=str(workspace_path), workspace_path=str(workspace_path))

        click.echo(f"Workspace structure and manifest created successfully in {workspace_path}")
    except OSError as e:
        click.echo(f"Error creating workspace directory structure: {e}", err=True)

@textura_cli.command()
@click.option('--workspace', default='textura_workspace', help='Path to the Textura workspace.', show_default=True, required=True, type=click.Path())
@click.option('--source', help='Path to the source data directory or a single file to ingest.', required=True, type=click.Path(exists=True, file_okay=True, dir_okay=True, readable=True))
def ingest(workspace: str, source: str):
    """Ingests data from a source into the Textura workspace."""
    workspace_path = Path(workspace).resolve()
    source_path = Path(source).resolve()

    click.echo(f"Starting ingestion process for source: {source_path} into workspace: {workspace_path}")

    if not workspace_path.exists() or not (workspace_path / 'manifest.json').exists():
        click.echo(f"Error: Workspace '{workspace_path}' is not initialized or manifest.json is missing. Run 'textura init' first.", err=True)
        return

    try:
        source_dir_for_watcher = source_path.parent if source_path.is_file() else source_path
        watcher = SourceWatcher(source_dir=str(source_dir_for_watcher), workspace_path=str(workspace_path))

        if source_path.is_file():
            relative_file_path_str = str(source_path.relative_to(source_dir_for_watcher))
            if relative_file_path_str in watcher.ingested_files:
                click.echo(f"File '{source_path.name}' has already been ingested. Skipping.")
                files_to_process = []
            else:
                files_to_process = [source_path]
        else:
            files_to_process = watcher.get_new_files()

        if not files_to_process:
            click.echo("No new files to ingest.")
            return

        click.echo(f"Found {len(files_to_process)} new file(s) to ingest:")
        for f_idx, f_path in enumerate(files_to_process):
             click.echo(f"  [{f_idx+1}/{len(files_to_process)}] {f_path.name}")


        chunker = Chunker()
        embedder = Embedder()
        vector_store = FAISSVectorStore(workspace_path=str(workspace_path))

        processed_files_count = 0
        for file_path in files_to_process:
            click.echo(f"\nProcessing file: {file_path}...")

            text_chunks = chunker.chunk_file(file_path)
            if not text_chunks:
                click.echo(f"No text chunks generated for {file_path.name}. Skipping.")
                continue
            click.echo(f"  - Extracted {len(text_chunks)} chunk(s).")

            embeddings = embedder.generate_embeddings(text_chunks)
            if not embeddings or len(embeddings) != len(text_chunks):
                click.echo(f"Error generating embeddings for {file_path.name}. Skipping.")
                continue
            click.echo(f"  - Generated {len(embeddings)} embedding(s).")

            relative_path_for_doc_store = str(file_path.relative_to(source_dir_for_watcher))
            vector_store.add_embeddings(text_chunks, embeddings, source_file=relative_path_for_doc_store)
            click.echo(f"  - Added embeddings to vector store.")

            watcher.mark_as_ingested(file_path)
            click.echo(f"  - Marked '{file_path.name}' as ingested in manifest.")
            processed_files_count += 1

        if processed_files_count > 0:
            vector_store.save()
            click.echo(f"\nSuccessfully processed and ingested {processed_files_count} file(s).")
            click.echo(f"Vector store index and document metadata saved in {vector_store.index_dir}")
        else:
            click.echo("\nNo files were processed in this run.")

    except Exception as e:
        click.echo(f"An unexpected error occurred during the ingestion process: {e}", err=True)
        import traceback
        traceback.print_exc()


@textura_cli.command()
@click.option('--workspace', default='textura_workspace', help='Path to the Textura workspace.', show_default=True, required=True, type=click.Path(exists=True))
def extract(workspace: str):
    """Extracts structured data (Events, Mysteries) from ingested documents."""
    workspace_path = Path(workspace).resolve()
    click.echo(f"Starting extraction process for workspace: {workspace_path}")

    docs_jsonl_path = workspace_path / "index" / "docs.jsonl"
    if not docs_jsonl_path.exists():
        click.echo(f"Error: Document store 'docs.jsonl' not found in '{docs_jsonl_path.parent}'. "
                   "Run 'textura ingest' first.", err=True)
        return

    metacog_logger = Metacog(workspace_path=str(workspace_path))
    extractor_agent = ExtractorAgent(metacog_logger=metacog_logger)

    all_extractions: List[ExtractionItem] = []
    processed_doc_count = 0

    try:
        with open(docs_jsonl_path, 'r') as f:
            for line_num, line in enumerate(f):
                if not line.strip():
                    continue

                try:
                    doc_data = json.loads(line)
                    processed_doc_count +=1
                except json.JSONDecodeError as e:
                    click.echo(f"Warning: Skipping malformed line {line_num+1} in docs.jsonl: {e}", err=True)
                    continue

                chunk_text = doc_data.get("text")
                chunk_id = str(doc_data.get("id", f"unknown_id_line_{line_num+1}"))
                source_file = doc_data.get("source_file", "unknown_source")

                if not chunk_text:
                    click.echo(f"Warning: Skipping document/chunk with ID '{chunk_id}' (line {line_num+1}) due to missing 'text' field.", err=True)
                    continue

                click.echo(f"  Processing chunk ID: {chunk_id} (Source: {source_file})...")

                validated_items = extractor_agent.extract_from_chunk(
                    chunk_text=chunk_text,
                    chunk_id=chunk_id,
                    source_file=source_file
                )
                all_extractions.extend(validated_items)

        click.echo(f"\nProcessed {processed_doc_count} documents/chunks from docs.jsonl.")

        if not all_extractions:
            click.echo("No structured data items were extracted successfully.")
            return

        output_dir = workspace_path / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        extractions_file_path = output_dir / "extractions.jsonl"

        with open(extractions_file_path, 'w') as f_out:
            for item in all_extractions:
                f_out.write(item.model_dump_json() + '\n') # model_dump_json includes file_id if set

        click.echo(f"Successfully extracted {len(all_extractions)} items.")
        click.echo(f"Validated extractions saved to: {extractions_file_path}")
        click.echo(f"Metacognitive logs for extraction saved in: {metacog_logger.log_file_path}")

    except Exception as e:
        click.echo(f"An unexpected error occurred during the extraction process: {e}", err=True)
        import traceback
        traceback.print_exc()


@textura_cli.command() # Changed from placeholder
@click.option('--workspace', default='textura_workspace', help='Path to the Textura workspace.', show_default=True, required=True, type=click.Path(exists=True))
@click.option('--stats', is_flag=True, help='Print a run statistics summary based on the metacog log.')
# Removed --pipeline option for now, as weave is specific to extractions -> vault
def weave(workspace: str, stats: bool): # Renamed pipeline arg
    """Processes extractions, writes them to the vault, and builds timelines."""
    workspace_path = Path(workspace).resolve()
    click.echo(f"Starting weave process for workspace: {workspace_path}")

    extractions_jsonl_path = workspace_path / "data" / "processed" / "extractions.jsonl"
    if not extractions_jsonl_path.exists():
        click.echo(f"Error: Extractions file '{extractions_jsonl_path}' not found. "
                   "Run 'textura extract' first.", err=True)
        return

    vault_writer = VaultWriter(workspace_path=str(workspace_path))

    all_events: List[EventV1] = []
    all_mysteries: List[MysteryV1] = [] # Though not used by timeline builder yet

    processed_extractions_count = 0
    written_to_vault_count = 0

    try:
        with open(extractions_jsonl_path, 'r') as f:
            for line_num, line in enumerate(f):
                if not line.strip():
                    continue

                try:
                    extraction_data = json.loads(line)
                    processed_extractions_count += 1
                except json.JSONDecodeError as e:
                    click.echo(f"Warning: Skipping malformed line {line_num+1} in {extractions_jsonl_path}: {e}", err=True)
                    continue

                # Determine if it's an Event or Mystery based on fields (Pydantic doesn't store the original type name in JSON by default)
                # A more robust way would be to have ExtractorAgent add a 'model_type': 'EventV1' field.
                # For now, we check for unique fields like 'description' for Event, 'question' for Mystery.

                item_processed = False
                if "description" in extraction_data and "timestamp" in extraction_data: # Likely EventV1
                    try:
                        event = EventV1(**extraction_data)
                        # Capture the file_id returned by vault_writer
                        _, event.file_id = vault_writer.write_event(event)
                        all_events.append(event)
                        written_to_vault_count +=1
                        item_processed = True
                    except Exception as e: # Catch Pydantic validation or other errors
                        click.echo(f"Error processing potential EventV1 object (line {line_num+1}): {e}\nData: {extraction_data}", err=True)

                elif "question" in extraction_data and "context" in extraction_data: # Likely MysteryV1
                    try:
                        mystery = MysteryV1(**extraction_data)
                        _, mystery.file_id = vault_writer.write_mystery(mystery)
                        all_mysteries.append(mystery)
                        written_to_vault_count +=1
                        item_processed = True
                    except Exception as e:
                        click.echo(f"Error processing potential MysteryV1 object (line {line_num+1}): {e}\nData: {extraction_data}", err=True)

                if not item_processed:
                    click.echo(f"Warning: Could not determine type for extraction on line {line_num+1}. Data: {extraction_data}", err=True)


        click.echo(f"\nProcessed {processed_extractions_count} extractions from '{extractions_jsonl_path}'.")
        click.echo(f"Wrote {written_to_vault_count} items to the vault at '{vault_writer.vault_path}'.")

        if not all_events:
            click.echo("No events found to build timelines.")
        else:
            click.echo(f"Building timelines from {len(all_events)} events...")
            timeline_builder = TimelineBuilder(workspace_path=str(workspace_path), events=all_events)
            timeline_builder.build_timelines()
            click.echo("Timeline building complete.")

        if stats:
            collector = StatsCollector(workspace_path=str(workspace_path))
            summary = collector.format_summary(collector.collect())
            click.echo(summary)

        click.echo("\nWeave process finished.")

    except Exception as e:
        click.echo(f"An unexpected error occurred during the weave process: {e}", err=True)
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    textura_cli()
