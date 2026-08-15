"""Interaktywna wyszukiwarka w konsoli."""

import os
import sqlite3
import time

from src.setup import get_manager, setup
from src.search import cosine_search, lsi_search

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "simplified_wiki_index.db",
)


def _fetch_title_url(conn: sqlite3.Connection, doc_id: int) -> tuple[str, str]:
    cur = conn.cursor()
    cur.execute("SELECT title, url FROM pages WHERE id = ?", (doc_id,))
    row = cur.fetchone()
    return (row[0], row[1]) if row else ("?", "?")


def run_search_loop(
    use_svd: bool,
    doc_matrix,
    svd_components,
    index: dict,
    doc_ids: list,
    idf_weights,
    k_results: int,
) -> None:
    """Pętla interaktywna przyjmująca zapytania i wyświetlająca wyniki."""
    conn = sqlite3.connect(DB_PATH)

    while True:
        try:
            query = input("\nZapytanie (wpisz 'koniec' żeby wyjść): ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if query.lower() in ("koniec", "exit", "quit", "q"):
            break

        if not query:
            print("Podaj treść zapytania.")
            continue

        t0 = time.perf_counter()
        if use_svd:
            results = lsi_search(query, svd_components, index, doc_ids, idf_weights, k_results)
        else:
            results = cosine_search(query, doc_matrix, index, doc_ids, idf_weights, k_results)
        elapsed = time.perf_counter() - t0

        filtered = [(did, sim) for did, sim in results if sim > 0.001]
        if filtered:
            print(f"\nWyniki ({elapsed:.3f}s):")
            for rank, (doc_id, sim) in enumerate(filtered, 1):
                title, url = _fetch_title_url(conn, doc_id)
                print(f"  {rank:2}. [{sim:.4f}] {title}")
                print(f"       {url}")
        else:
            print("\nBrak wyników dla tego zapytania.")

    conn.close()
    print("\nDo widzenia!")


def main():
    """Punkt wejścia dla trybu konsolowego."""
    params = setup()
    run_search_loop(*params)


if __name__ == "__main__":
    main()