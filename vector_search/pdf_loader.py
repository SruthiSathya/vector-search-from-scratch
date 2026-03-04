import fitz  # PyMuPDF
from pathlib import Path

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    full_text = ""

    for page in doc:
        full_text += page.get_text()

    return full_text


def load_pdfs_from_directory(directory: str):
    texts = []
    paths = Path(directory).glob("*.pdf")

    for path in paths:
        print(f"Loading {path.name}")
        text = extract_text_from_pdf(str(path))
        texts.append((path.name, text))

    return texts
