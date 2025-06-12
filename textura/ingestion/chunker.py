from pathlib import Path
from typing import List, Dict, Any
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document as LlamaDocument

# It's good practice to centralize reader mapping if more types are added.
READER_MAPPING: Dict[str, BaseReader] = {}

try:
    from llama_index.readers.file import PyMuPDFReader
    READER_MAPPING[".pdf"] = PyMuPDFReader()
except ImportError:
    print("PyMuPDFReader not available. Please install pymupdf and llama-index-readers-file.")
    # You could fall back to a default simple text reader or raise an error.

# Add more readers as needed:
# from llama_index.readers.docx import DocxReader
# READER_MAPPING[".docx"] = DocxReader()
# from llama_index.readers.html import HTMLParseReader
# READER_MAPPING[".html"] = HTMLParseReader()


class Chunker:
    """
    Chunks documents using LlamaIndex readers based on file type.
    """
    def __init__(self, default_chunk_size: int = 1024, default_chunk_overlap: int = 20):
        """
        Initializes the Chunker.

        Args:
            default_chunk_size: The default size for text chunks. (Currently not directly used by all LlamaIndex readers)
            default_chunk_overlap: The default overlap for text chunks. (Currently not directly used by all LlamaIndex readers)
        """
        # These defaults might be used if we implement manual chunking or specific LlamaIndex text splitters later.
        self.default_chunk_size = default_chunk_size
        self.default_chunk_overlap = default_chunk_overlap

    def chunk_file(self, file_path: Path) -> List[str]:
        """
        Chunks a file into text segments based on its extension.

        Args:
            file_path: Path to the file to be chunked.

        Returns:
            A list of text chunks. Returns an empty list if the file type
            is not supported or an error occurs.
        """
        file_ext = file_path.suffix.lower()
        reader = READER_MAPPING.get(file_ext)

        if not reader:
            # Fallback for unsupported types: try to read as plain text
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                # Simple split by paragraph, could be more sophisticated
                return [chunk for chunk in text_content.split('\n\n') if chunk.strip()]
            except Exception as e:
                print(f"Unsupported file type '{file_ext}' and failed to read as plain text: {file_path}. Error: {e}")
                return []

        try:
            # LlamaIndex readers return a list of LlamaDocument objects
            # PyMuPDFReader might expect file_path as a direct keyword argument.
            llama_documents: List[LlamaDocument] = reader.load_data(file_path=file_path)

            # Extract text content from each LlamaDocument
            text_chunks: List[str] = []
            for doc in llama_documents:
                text_chunks.append(doc.get_content()) # doc.text is an alias for doc.get_content()
            return text_chunks
        except Exception as e:
            print(f"Error chunking file {file_path} with {reader.__class__.__name__}: {e}")
            return []

if __name__ == '__main__':
    # Example Usage (requires a PDF file for PyMuPDFReader)
    # Create a dummy PDF for testing if pymupdf is installed
    # You'll need to have pymupdf installed: pip install pymupdf

    # Create a dummy source directory
    dummy_source_dir = Path("_test_source_chunker")
    dummy_source_dir.mkdir(exist_ok=True)

    # Dummy text file
    dummy_text_file = dummy_source_dir / "sample.txt"
    dummy_text_file.write_text("This is a sample text file.\n\nIt has multiple paragraphs.\n\nThis is the third paragraph.")

    # Dummy PDF file (requires fitz, i.e. PyMuPDF)
    dummy_pdf_file = None
    try:
        import fitz # PyMuPDF
        dummy_pdf_file = dummy_source_dir / "sample.pdf"
        doc = fitz.open() # New PDF
        page = doc.new_page()
        page.insert_text((50, 72), "This is a sample PDF document.")
        page.insert_text((50, 144), "It contains some text for chunking.")
        doc.save(str(dummy_pdf_file))
        doc.close()
        print(f"Created dummy PDF: {dummy_pdf_file}")
    except ImportError:
        print("PyMuPDF (fitz) not installed. PDF chunking example will be skipped.")
    except Exception as e:
        print(f"Error creating dummy PDF: {e}")

    chunker = Chunker()

    print("\n--- Chunking TXT file ---")
    text_chunks = chunker.chunk_file(dummy_text_file)
    if text_chunks:
        for i, chunk in enumerate(text_chunks):
            print(f"Chunk {i+1}:\n{chunk}\n---")
    else:
        print("No chunks generated for text file.")

    if dummy_pdf_file and dummy_pdf_file.exists():
        print("\n--- Chunking PDF file ---")
        pdf_chunks = chunker.chunk_file(dummy_pdf_file)
        if pdf_chunks:
            for i, chunk in enumerate(pdf_chunks):
                print(f"Chunk {i+1}:\n{chunk}\n---")
        else:
            print("No chunks generated for PDF file.")
    else:
        print("\nSkipping PDF chunking example as dummy PDF could not be created.")

    # Clean up
    # import shutil
    # shutil.rmtree(dummy_source_dir)
