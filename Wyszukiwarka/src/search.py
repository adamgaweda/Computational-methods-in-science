"""
Moduł wyszukiwania dokumentów.

Obsługuje trzy strategie wyszukiwania:
- cosinusowe podobieństwo na macierzy TF-IDF (klasyczne)
- LSI z projekcją zapytania do przestrzeni pojęć (SVD)
- BM25 z podobieństwem cosinusowym
"""

import re

import numpy as np
from nltk.stem import PorterStemmer
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize as sk_normalize

from src.svd import SVDComponents

_STEMMER = PorterStemmer()


def vectorize_query(
    query: str,
    index: dict,
    idf_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Zamienia tekst zapytania na znormalizowany wektor TF(-IDF).

    Pipeline: tokenizacja → stemming → zliczenie częstości → normalizacja
    (opcjonalnie ważona IDF).

    Args:
        query: Tekst zapytania w języku naturalnym.
        index: Słownik terminów ``{termin: indeks}`` z indeksu wyszukiwarki.
        idf_weights: Opcjonalny wektor wag IDF.  Jeśli podany, wagi TF
            są przemnażane przez odpowiednie wartości IDF.

    Returns:
        Znormalizowany (L2) wektor ``np.ndarray`` o długości ``len(index)``.
        Jeśli żaden termin z zapytania nie należy do słownika, zwracany
        jest wektor zerowy.
    """
    words = re.findall(r"\b[a-z]{3,}\b", query.lower())
    stems = [_STEMMER.stem(w) for w in words]

    vec = np.zeros(len(index), dtype=float)
    total = 0
    for stem in stems:
        if stem in index:
            vec[index[stem]] += 1
            total += 1

    if total > 0:
        vec /= total

    if idf_weights is not None:
        vec *= idf_weights

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm

    return vec


def cosine_search(
    query: str,
    doc_matrix: csr_matrix,
    index: dict,
    doc_ids: list,
    idf_weights: np.ndarray | None = None,
    top_k: int = 10,
) -> list[tuple]:
    """Wyszukuje dokumenty przez podobieństwo cosinusowe.

    Zapytanie jest rzutowane na przestrzeń terminów i porównywane
    z każdym dokumentem w macierzy.

    Args:
        query: Tekst zapytania.
        doc_matrix: Znormalizowana macierz term-dokument ``(vocab × docs)``.
        index: Słownik terminów.
        doc_ids: Lista identyfikatorów dokumentów (wierszy bazy).
        idf_weights: Wagi IDF (opcjonalne).
        top_k: Liczba wyników do zwrócenia.

    Returns:
        Lista par ``(doc_id, similarity)`` posortowana malejąco wg podobieństwa.
    """
    q_vec = vectorize_query(query, index, idf_weights)

    scores = np.abs(q_vec @ doc_matrix)

    top_idx = np.argsort(-scores)[:top_k]
    return [(doc_ids[i], float(scores[i])) for i in top_idx]


def lsi_search(
    query: str,
    svd: SVDComponents,
    index: dict,
    doc_ids: list,
    idf_weights: np.ndarray | None = None,
    top_k: int = 10,
) -> list[tuple]:
    """Wyszukuje dokumenty w zredukowanej przestrzeni pojęć (LSI).

    Zapytanie jest rzutowane do przestrzeni konceptów SVD zgodnie z:
        q_concept = q_vec @ U @ diag(1/sigma)

    Dokumenty (kolumny Vt) są porównywane z wektorem zapytania przez
    podobieństwo cosinusowe w tej przestrzeni.

    Args:
        query: Tekst zapytania.
        svd: Komponenty SVD z :func:`src.svd.truncated_svd`.
        index: Słownik terminów.
        doc_ids: Lista identyfikatorów dokumentów.
        idf_weights: Wagi IDF (opcjonalne).
        top_k: Liczba wyników.

    Returns:
        Lista par ``(doc_id, similarity)`` posortowana malejąco.
    """
    q_vec = vectorize_query(query, index, idf_weights)

    # Projekcja do przestrzeni pojęć: q → q @ U @ Σ⁻¹
    q_concept = q_vec @ svd.U @ np.diag(1.0 / svd.sigma)

    # Normalizacja zapytania i dokumentów w tej przestrzeni
    q_norm = np.linalg.norm(q_concept)
    if q_norm > 0:
        q_concept = q_concept / q_norm

    # Kolumny Vt to współrzędne dokumentów w przestrzeni pojęć
    doc_concepts = sk_normalize(svd.Vt, axis=0, norm="l2")
    scores = np.abs(q_concept @ doc_concepts)

    top_idx = np.argsort(-scores)[:top_k]
    return [(doc_ids[i], float(scores[i])) for i in top_idx]