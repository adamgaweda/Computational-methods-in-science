"""
Moduł SVD dla wyszukiwarki semantycznej (LSI – Latent Semantic Indexing).

Implementuje obcięty rozkład SVD macierzy term-dokument, co pozwala
na projekcję dokumentów i zapytań do niskowymiarowej przestrzeni
pojęć (latent concept space).
"""

from typing import NamedTuple

import numpy as np
from scipy.sparse.linalg import svds


class SVDComponents(NamedTuple):
    """Komponenty rozkładu SVD.

    Attributes:
        U: Macierz lewych wektorów osobliwych, kształt ``(vocab_size, k)``.
           Wiersze odpowiadają terminom, kolumny – pojęciom.
        sigma: Wektor wartości osobliwych, kształt ``(k,)``, malejąco.
        Vt: Macierz prawych wektorów osobliwych (transponowana),
            kształt ``(k, n_docs)``.  Kolumny odpowiadają dokumentom.
        k: Liczba zachowanych wymiarów.
    """

    U: np.ndarray
    sigma: np.ndarray
    Vt: np.ndarray
    k: int


from sklearn.decomposition import TruncatedSVD

def truncated_svd(matrix, k: int) -> SVDComponents:
    """Oblicza obcięty rozkład SVD macierzy term-dokument.

    Używa algorytmu Randomized SVD (`sklearn.decomposition.TruncatedSVD`), 
    który jest wysoce zoptymalizowany i wydajny dla dużych macierzy rzadkich.

    Args:
        matrix: Macierz rzadka lub gęsta o kształcie ``(vocab_size × n_docs)``.
        k: Liczba dominujących wymiarów do zachowania.  Musi spełniać
           ``1 ≤ k < min(vocab_size, n_docs)``.

    Returns:
        :class:`SVDComponents` z polami ``U``, ``sigma``, ``Vt``, ``k``.

    Raises:
        ValueError: Jeśli ``k`` jest poza dopuszczalnym zakresem.
    """
    min_dim = min(matrix.shape)
    if not (1 <= k < min_dim):
        raise ValueError(f"k musi być w zakresie [1, {min_dim - 1}], podano k={k}")

    # TruncatedSVD działa bardzo szybko
    svd = TruncatedSVD(n_components=k, algorithm='randomized', random_state=42)
    
    # fit_transform zwraca U * Sigma
    U_sigma = svd.fit_transform(matrix)
    sigma = svd.singular_values_
    Vt = svd.components_
    
    # Odzyskanie lewych wektorów osobliwych U (zapobiegamy dzieleniu przez 0)
    sigma_safe = np.where(sigma == 0, 1e-10, sigma)
    U = U_sigma / sigma_safe

    return SVDComponents(
        U=U,
        sigma=sigma,
        Vt=Vt,
        k=k,
    )