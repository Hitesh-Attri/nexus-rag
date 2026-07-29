import sys

sys.path.insert(0, "src")

from core.chunking import chunk_text

sample = "word " * 600  # ~3000 characters
chunks = chunk_text(sample)
print("chunks:", len(chunks))
print("sizes:", [len(c) for c in chunks])