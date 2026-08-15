"""
Moduł przetwarzania wstępnego dokumentów.

Zawiera funkcje do budowy indeksu terminów (słownika), obliczania
macierzy term-dokument z wagami TF oraz stosowania wag IDF i BM25.
"""

import re
import sqlite3
from collections import Counter, defaultdict

import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.sparse import csr_matrix, diags


_STEMMER = PorterStemmer()
_STOP_WORDS = None


def _get_stop_words():
    """Zwraca zbiór stop-words (lazy init)."""
    global _STOP_WORDS
    if _STOP_WORDS is None:
        import nltk
        try:
            _STOP_WORDS = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            _STOP_WORDS = set(stopwords.words("english"))
    return _STOP_WORDS


def _tokenize(text: str) -> list[str]:
    """Tokenizuje tekst: małe litery, tylko słowa >= 3 litery, stemming."""
    stop = _get_stop_words()
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return [_STEMMER.stem(w) for w in words if w not in stop]


def build_index(db_path: str, min_df: int = 3, max_df: float = 0.6) -> dict:
    """Buduje słownik terminów na podstawie dokumentów w bazie.

    Args:
        db_path: Ścieżka do bazy SQLite z tabelą `pages`.
        min_df: Minimalna liczba dokumentów, w których musi wystąpić termin.
        max_df: Maksymalny ułamek dokumentów (względem całości zbioru).

    Returns:
        Słownik ``{termin: indeks}`` posortowany po indeksie.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM pages")
    n_docs = cur.fetchone()[0]

    doc_freq: Counter = Counter()

    cur.execute("SELECT content FROM pages")
    for (content,) in cur.fetchall():
        unique_terms = set(_tokenize(content))
        doc_freq.update(unique_terms)

    conn.close()

    upper = max_df * n_docs
    selected = [
        term
        for term, df in doc_freq.items()
        if min_df <= df <= upper
    ]

    return {term: idx for idx, term in enumerate(selected)}


def compute_tf_matrix(db_path: str, index: dict):
    """Tworzy macierz TF (term frequency) jako macierz rzadką CSR.

    Wagi TF są normalizowane przez długość dokumentu (liczba wszystkich
    pasujących tokenów w dokumencie).

    Args:
        db_path: Ścieżka do bazy SQLite.
        index: Słownik ``{termin: indeks}`` zwrócony przez :func:`build_index`.

    Returns:
        Krotka ``(tf_matrix, doc_ids)``:
            - ``tf_matrix`` – macierz rzadka ``(vocab_size × n_docs)``
            - ``doc_ids`` – lista identyfikatorów wierszy bazy danych
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM pages")
    n_docs = cur.fetchone()[0]

    vocab_size = len(index)
    rows, cols, data = [], [], []
    doc_ids = []

    cur.execute("SELECT id, content FROM pages")
    for col_idx, (doc_id, content) in enumerate(cur.fetchall()):
        doc_ids.append(doc_id)
        tokens = _tokenize(content)

        freq: dict[int, int] = defaultdict(int)
        total = 0
        for t in tokens:
            if t in index:
                freq[index[t]] += 1
                total += 1

        if total == 0:
            continue

        for term_idx, count in freq.items():
            rows.append(term_idx)
            cols.append(col_idx)
            data.append(count / total)

    conn.close()

    tf_matrix = csr_matrix(
        (data, (rows, cols)), shape=(vocab_size, n_docs)
    )
    return tf_matrix, doc_ids


def apply_idf(tf_matrix: csr_matrix):
    """Stosuje wagi IDF do macierzy TF (przekształcenie TF-IDF).

    Wzór: IDF(t) = log(N / df(t)), gdzie N – liczba dokumentów, df(t) –
    liczba dokumentów zawierających termin t.

    Args:
        tf_matrix: Rzadka macierz TF o wymiarze ``(vocab_size × n_docs)``.

    Returns:
        Krotka ``(tfidf_matrix, idf_weights)``:
            - ``tfidf_matrix`` – macierz TF-IDF
            - ``idf_weights`` – wektor wag IDF dla każdego terminu
    """
    _, n_docs = tf_matrix.shape
    doc_counts = np.array((tf_matrix > 0).sum(axis=1)).flatten()
    idf_weights = np.log(n_docs / np.maximum(doc_counts, 1))
    tfidf = diags(idf_weights).dot(tf_matrix)
    return tfidf, idf_weights


def apply_bm25(tf_matrix: csr_matrix, k1: float = 1.5, b: float = 0.75):
    """Stosuje schemat wagowania BM25 do macierzy TF.

    BM25 jest rozszerzeniem TF-IDF uwzględniającym długość dokumentu.
    Wagi BM25(t, d) = IDF(t) · (tf · (k1 + 1)) / (tf + k1·(1 - b + b·dl/avgdl))

    Args:
        tf_matrix: Rzadka macierz TF o wymiarze ``(vocab_size × n_docs)``.
        k1: Parametr nasycenia częstości. Typowo 1.2–2.0.
        b: Parametr normalizacji długości. 0 = brak normalizacji, 1 = pełna.

    Returns:
        Krotka ``(bm25_matrix, idf_weights)``.
    """
    n_terms, n_docs = tf_matrix.shape

    # Długości dokumentów (suma wszystkich wag TF kolumny)
    doc_lengths = np.array(tf_matrix.sum(axis=0)).flatten()
    avg_dl = doc_lengths.mean()

    # IDF
    doc_counts = np.array((tf_matrix > 0).sum(axis=1)).flatten()
    idf_weights = np.log((n_docs - doc_counts + 0.5) / (doc_counts + 0.5) + 1)

    bm25_matrix = tf_matrix.copy().astype(float)
    bm25_matrix = bm25_matrix.tocsc()

    for j in range(n_docs):
        dl_norm = 1 - b + b * doc_lengths[j] / avg_dl
        col = bm25_matrix.getcol(j)
        # przekształcamy: tf → tf*(k1+1) / (tf + k1*dl_norm)
        col_data = col.data
        col_data[:] = col_data * (k1 + 1) / (col_data + k1 * dl_norm)
        bm25_matrix[:, j] = col

    bm25_matrix = bm25_matrix.tocsr()
    idf_diag = diags(idf_weights)
    bm25_matrix = idf_diag.dot(bm25_matrix)

    return bm25_matrix, idf_weights