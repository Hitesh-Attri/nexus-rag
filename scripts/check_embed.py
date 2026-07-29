import sys

sys.path.insert(0, "src")

from core.db import get_pool
from core.embeddings import embed_texts

vecs = embed_texts(["hello world", "the cat sat on the mat"])
print("vectors:", len(vecs), "dim:", len(vecs[0]))  # expect: vectors: 2 dim: 384

with get_pool().connection() as conn:
    conn.execute(
        "INSERT INTO documents (source, content, embedding) VALUES (%s, %s, %s)",
        ("smoke-test", "hello world", vecs[0]),
    )
    n = conn.execute(
        "SELECT count(*) FROM documents WHERE source = 'smoke-test'"
    ).fetchone()[0]
    print("smoke rows inserted:", n)          # expect: 1
    conn.execute("DELETE FROM documents WHERE source = 'smoke-test'")
    print("cleaned up")

get_pool().close()