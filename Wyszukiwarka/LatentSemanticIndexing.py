#notatki z zajęć
#TF-IDF
#BM25
#porter Stemmer
#porter Stemmer 2
#stop words removal
#HNSW graph
#Inverse document frequency - IDF
#dictionary >30k ~100k
#records >50k ~300k


"""
LatentSemanticIndexing.py
=========================

Demonstracja Latent Semantic Indexing (LSI) na minimalnym przykładzie
bez zewnętrznych zależności poza NumPy.

LSI (znane też jako LSA – Latent Semantic Analysis) to technika
przetwarzania języka naturalnego opierająca się na rozkładzie SVD macierzy
term-dokument.  Redukuje wymiarowość przestrzeni terminów do k-wymiarowej
przestrzeni *pojęć* (latent concepts), co pozwala na:
 - wyszukiwanie semantyczne (nie tylko dokładne dopasowanie terminów)
 - wykrywanie synonimii
 - usuwanie szumu z reprezentacji dokumentów

Uruchom: python LatentSemanticIndexing.py
"""

import numpy as np


# ---------------------------------------------------------------------------
# Krok 1: Zbiór przykładowych dokumentów
# ---------------------------------------------------------------------------

DOCUMENTS = [
    "human computer interaction",
    "survey of user opinion computer system response time",
    "EPS user interface management system",
    "system and human system engineering testing of EPS",
    "relation of user perceived response time to error measurement",
    "random graph generation and graph minors",
    "intersection graph partition and graph coloring",
    "graph minors widths and heights of trees",
]

QUERIES = [
    "human computer interface",   # powinien trafić w dok. 0-4
    "graph theory and algorithms", # powinien trafić w dok. 5-7
]


# ---------------------------------------------------------------------------
# Krok 2: Budowa macierzy term-dokument (TF)
# ---------------------------------------------------------------------------

def build_term_doc_matrix(docs: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    """Buduje binarną macierz term-dokument.

    Returns:
        (matrix, terms, documents) – macierz (|terms|×|docs|), lista terminów,
        lista dokumentów.
    """
    stop = {"and", "of", "to", "the", "a", "in", "with"}
    term_set = set()
    tokenized = []
    for doc in docs:
        tokens = {w for w in doc.lower().split() if w not in stop}
        tokenized.append(tokens)
        term_set |= tokens

    terms = sorted(term_set)
    term_idx = {t: i for i, t in enumerate(terms)}

    mat = np.zeros((len(terms), len(docs)), dtype=float)
    for j, tokens in enumerate(tokenized):
        for t in tokens:
            mat[term_idx[t], j] += 1.0

    return mat, terms, docs


# ---------------------------------------------------------------------------
# Krok 3: IDF – ważenie terminów
# ---------------------------------------------------------------------------

def compute_idf(mat: np.ndarray) -> np.ndarray:
    """Oblicza wagi IDF dla każdego terminu."""
    n_docs = mat.shape[1]
    df = (mat > 0).sum(axis=1)
    return np.log(n_docs / np.maximum(df, 1))


# ---------------------------------------------------------------------------
# Krok 4: SVD i redukcja do k wymiarów
# ---------------------------------------------------------------------------

def lsi_decompose(tfidf_mat: np.ndarray, k: int):
    """Rozkłada macierz TF-IDF na k dominujących składowych SVD.

    Returns:
        (U_k, sigma_k, Vt_k) – komponenty SVD o obciętym rankingu k.
    """
    U, sigma, Vt = np.linalg.svd(tfidf_mat, full_matrices=False)
    return U[:, :k], sigma[:k], Vt[:k, :]


# ---------------------------------------------------------------------------
# Krok 5: Projekcja zapytania i wyszukiwanie
# ---------------------------------------------------------------------------

def query_to_concept_vector(
    query: str,
    terms: list[str],
    idf: np.ndarray,
    U_k: np.ndarray,
    sigma_k: np.ndarray,
) -> np.ndarray:
    """Rzutuje zapytanie do k-wymiarowej przestrzeni pojęć."""
    term_idx = {t: i for i, t in enumerate(terms)}
    stop = {"and", "of", "to", "the", "a", "in", "with"}

    q_vec = np.zeros(len(terms))
    for w in query.lower().split():
        if w not in stop and w in term_idx:
            q_vec[term_idx[w]] += 1.0

    # Zastosowanie IDF
    q_vec *= idf

    # Normalizacja
    norm = np.linalg.norm(q_vec)
    if norm > 0:
        q_vec /= norm

    # Projekcja: q → q @ U_k @ Σ_k^{-1}
    return q_vec @ U_k @ np.diag(1.0 / sigma_k)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Podobieństwo cosinusowe między wektorami a i b."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Latent Semantic Indexing – demonstracja")
    print("=" * 60)

    mat, terms, docs = build_term_doc_matrix(DOCUMENTS)
    idf = compute_idf(mat)

    # TF-IDF
    tfidf = mat * idf[:, np.newaxis]

    k = 3  # wymiar przestrzeni pojęć
    U_k, sigma_k, Vt_k = lsi_decompose(tfidf, k)

    print(f"\nMacierz term-dokument: {mat.shape[0]} terminy × {mat.shape[1]} dokumentów")
    print(f"Redukcja SVD do k={k} wymiarów")
    print(f"Zachowane wartości osobliwe: {sigma_k.round(3)}")

    # Kolumny Vt_k to reprezentacje dokumentów w przestrzeni pojęć
    doc_concepts = Vt_k.T  # shape: (n_docs, k)
    # Normalizacja kolumn
    norms = np.linalg.norm(doc_concepts, axis=1, keepdims=True)
    norms[norms == 0] = 1
    doc_concepts_norm = doc_concepts / norms

    print("\n" + "-" * 60)
    for query in QUERIES:
        q_concept = query_to_concept_vector(query, terms, idf, U_k, sigma_k)
        q_norm = np.linalg.norm(q_concept)
        if q_norm > 0:
            q_concept /= q_norm

        sims = [cosine_sim(q_concept, doc_concepts_norm[j]) for j in range(len(docs))]
        ranking = sorted(enumerate(sims), key=lambda x: -x[1])

        print(f"\nZapytanie: \"{query}\"")
        print("Ranking wyników:")
        for rank, (idx, sim) in enumerate(ranking, 1):
            bar = "#" * int(sim * 20)
            print(f"  {rank}. [{sim:+.4f}] {bar:<20} {docs[idx]}")


if __name__ == "__main__":
    main()
