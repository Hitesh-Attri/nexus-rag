"""Apply every .sql file in migrations/ in order. Run: python scripts/migrate.py

Uses a plain connection (NOT the pgvector-registered pool) on purpose: the pool
registers the vector type on connect, which requires the extension to already
exist — a chicken-and-egg the migration itself resolves.
"""

import sys
from pathlib import Path

sys.path.insert(0, "src")

import psycopg  # noqa: E402

from core.config import get_settings  # noqa: E402


def main() -> None:
    settings = get_settings()
    files = sorted(Path("migrations").glob("*.sql"))
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        for f in files:
            print(f"applying {f.name}")
            conn.execute(f.read_text())
    print("done")


if __name__ == "__main__":
    main()