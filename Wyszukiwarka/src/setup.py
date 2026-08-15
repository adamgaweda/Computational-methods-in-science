"""
Menedżer indeksu wyszukiwarki.

Odpowiada za jednorazowe zbudowanie (lub wczytanie z cache) wszystkich
komponentów potrzebnych do działania wyszukiwarki: słownika, macierzy
term-dokument, wag IDF oraz komponentów SVD.
"""

import os
import pickle
import time

from scipy.sparse import load_npz, save_npz
from sklearn.preprocessing import normalize as sk_normalize

from src.preprocessing import apply_idf, build_index, compute_tf_matrix
from src.svd import SVDComponents, truncated_svd

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
)

DB_PATH = os.path.join(_DATA_DIR, "simplified_wiki_index.db")


def _data_path(*parts: str) -> str:
    """Zwraca ścieżkę w katalogu data/, tworząc katalog jeśli nie istnieje."""
    path = os.path.join(_DATA_DIR, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


class IndexManager:
    """Zarządza komponentami indeksu wyszukiwarki z obsługą cache (pickle/npz).

    Każdy komponent jest buforowany na dysk; przy kolejnym uruchomieniu
    jest wczytywany bez ponownego obliczania.

    Attributes:
        index: Słownik terminów ``{termin: int}``.
        doc_matrix: Znormalizowana macierz TF-IDF ``(vocab × docs)``.
        doc_ids: Lista ID dokumentów.
        idf_weights: Wagi IDF dla każdego terminu.
        svd_cache: Słownik ``{k: SVDComponents}`` z załadowanymi SVD.
    """

    def __init__(self):
        self.index: dict = {}
        self.doc_matrix = None
        self.doc_ids: list = []
        self.idf_weights = None
        self.svd_cache: dict[int, SVDComponents] = {}

    # ------------------------------------------------------------------
    # Słownik (indeks terminów)
    # ------------------------------------------------------------------

    def load_index(self) -> dict:
        """Wczytuje (lub buduje) słownik terminów."""
        cache_file = _data_path("term_index.pkl")

        if os.path.isfile(cache_file):
            print("Wczytywanie słownika z pliku cache...")
            try:
                with open(cache_file, "rb") as f:
                    self.index = pickle.load(f)
                print(f"  Słownik: {len(self.index):,} terminów")
                return self.index
            except Exception as exc:
                print(f"  Błąd wczytywania: {exc}")

        print("Budowanie słownika...")
        t0 = time.perf_counter()
        self.index = build_index(DB_PATH)
        elapsed = time.perf_counter() - t0
        print(f"  Gotowe [{elapsed:.2f}s] – {len(self.index):,} terminów")

        with open(cache_file, "wb") as f:
            pickle.dump(self.index, f)

        return self.index

    # ------------------------------------------------------------------
    # Macierz term-dokument
    # ------------------------------------------------------------------

    def load_matrix(self) -> tuple:
        """Wczytuje (lub oblicza) macierz TF-IDF i wektor IDF."""
        matrix_file = _data_path("tf_idf_matrix.npz")
        ids_file = _data_path("document_ids.pkl")

        if os.path.isfile(matrix_file) and os.path.isfile(ids_file):
            print("Wczytywanie macierzy term-dokument z pliku cache...")
            try:
                tf_matrix = load_npz(matrix_file)
                with open(ids_file, "rb") as f:
                    self.doc_ids = pickle.load(f)

                print("Stosowanie wag IDF...")
                t0 = time.perf_counter()
                tfidf, self.idf_weights = apply_idf(tf_matrix)
                elapsed = time.perf_counter() - t0
                print(f"  IDF [{elapsed:.2f}s]")

                print("Normalizacja macierzy...")
                t0 = time.perf_counter()
                self.doc_matrix = sk_normalize(tfidf, axis=0, norm="l2")
                elapsed = time.perf_counter() - t0
                print(f"  Normalizacja [{elapsed:.2f}s]")
                return self.doc_matrix, self.doc_ids
            except Exception as exc:
                print(f"  Błąd wczytywania: {exc}")

        print("Obliczanie macierzy TF...")
        t0 = time.perf_counter()
        tf_matrix, self.doc_ids = compute_tf_matrix(DB_PATH, self.index)
        elapsed = time.perf_counter() - t0
        shape = tf_matrix.shape
        print(f"  TF [{elapsed:.2f}s] – {shape[0]}×{shape[1]}")

        save_npz(matrix_file, tf_matrix)
        with open(ids_file, "wb") as f:
            pickle.dump(self.doc_ids, f)

        print("Stosowanie wag IDF...")
        t0 = time.perf_counter()
        tfidf, self.idf_weights = apply_idf(tf_matrix)
        elapsed = time.perf_counter() - t0
        print(f"  IDF [{elapsed:.2f}s]")

        print("Normalizacja macierzy...")
        t0 = time.perf_counter()
        self.doc_matrix = sk_normalize(tfidf, axis=0, norm="l2")
        elapsed = time.perf_counter() - t0
        print(f"  Normalizacja [{elapsed:.2f}s]")

        return self.doc_matrix, self.doc_ids

    # ------------------------------------------------------------------
    # SVD
    # ------------------------------------------------------------------

    def load_svd(self, k: int) -> SVDComponents:
        """Wczytuje (lub oblicza) komponenty SVD dla danego k.

        Args:
            k: Liczba zachowanych wymiarów.

        Returns:
            :class:`SVDComponents`.
        """
        if k in self.svd_cache:
            return self.svd_cache[k]

        cache_file = _data_path(f"svd_k{k}.pkl")

        if os.path.isfile(cache_file):
            print(f"Wczytywanie SVD (k={k}) z pliku cache...")
            try:
                with open(cache_file, "rb") as f:
                    svd = pickle.load(f)
                self.svd_cache[k] = svd
                return svd
            except Exception as exc:
                print(f"  Błąd wczytywania SVD: {exc}")

        print(f"Obliczanie SVD (k={k})...")
        t0 = time.perf_counter()
        svd = truncated_svd(self.doc_matrix, k)
        elapsed = time.perf_counter() - t0
        print(f"  SVD [{elapsed:.2f}s]")

        with open(cache_file, "wb") as f:
            pickle.dump(svd, f)

        self.svd_cache[k] = svd
        return svd

    def initialize(self) -> None:
        """Inicjalizuje wszystkie komponenty (słownik + macierz)."""
        self.load_index()
        self.load_matrix()

    @property
    def ready(self) -> bool:
        """Czy menedżer jest zainicjalizowany."""
        return self.doc_matrix is not None and len(self.index) > 0


# Singleton – używany przez app.py i main.py
_manager: IndexManager | None = None


def get_manager() -> IndexManager:
    """Zwraca globalny singleton IndexManager."""
    global _manager
    if _manager is None:
        _manager = IndexManager()
    return _manager


# ------------------------------------------------------------------
# Funkcje kompatybilności wstecznej (używane przez report.ipynb)
# ------------------------------------------------------------------

def load_or_compute_vocabulary():
    """Alias dla kompatybilności ze starym interfejsem."""
    mgr = get_manager()
    return mgr.load_index()


def load_or_compute_term_doc_matrix(vocabulary):
    """Alias dla kompatybilności ze starym interfejsem."""
    mgr = get_manager()
    if not mgr.index:
        mgr.index = vocabulary
    return mgr.load_matrix()


def load_or_compute_svd(use_idf, normalized_matrix, k_svd):
    """Alias dla kompatybilności ze starym interfejsem."""
    mgr = get_manager()
    # Dla kompatybilności – używamy aktualnej macierzy z managera
    if mgr.doc_matrix is not None:
        return mgr.load_svd(k_svd)
    # fallback: oblicz bezpośrednio
    svd = truncated_svd(normalized_matrix, k_svd)
    return svd


def setup():
    """Interaktywna konfiguracja wyszukiwarki (dla main.py)."""
    use_idf_inp = input("Użyć IDF? (t/n, domyślnie t): ").lower()
    use_idf = not use_idf_inp.startswith("n")

    use_svd_inp = input("Użyć SVD? (t/n, domyślnie t): ").lower()
    use_svd = not use_svd_inp.startswith("n")

    k_svd = 100
    if use_svd:
        k_raw = input("Liczba wymiarów SVD (domyślnie 100): ")
        k_svd = int(k_raw) if k_raw.isdigit() else 100

    k_results_raw = input("Liczba wyników (domyślnie 10): ")
    k_results = int(k_results_raw) if k_results_raw.isdigit() else 10

    mgr = get_manager()
    t0 = time.perf_counter()
    mgr.initialize()
    elapsed = time.perf_counter() - t0
    print(f"\nCałkowity czas inicjalizacji: {elapsed:.2f}s")

    svd_components = mgr.load_svd(k_svd) if use_svd else None

    # Zwraca krotka kompatybilna ze starym kodem main.py
    return (
        use_svd,
        mgr.doc_matrix,
        svd_components,
        mgr.index,
        mgr.doc_ids,
        mgr.idf_weights if use_idf else None,
        k_results,
    )