import re

def chunk_text_from_scratch(text: str, chunk_size: int = 1000, chunk_overlap: int = 150):
    """
    Splits the input text into chunks of roughly chunk_size characters,
    with an overlap of chunk_overlap characters between consecutive chunks.
    Tries to split at sentence boundaries if possible.
    """
    # Split text into sentences (basic approximation)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If adding the next sentence exceeds chunk_size, start a new chunk
        if len(current_chunk) + len(sentence) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Start new chunk with overlap
            if chunk_overlap > 0:
                # Take last `chunk_overlap` characters of previous chunk as overlap
                current_chunk = current_chunk[-chunk_overlap:] if len(current_chunk) >= chunk_overlap else current_chunk
            else:
                current_chunk = ""
        # Add sentence to current chunk
        current_chunk += " " + sentence

    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
