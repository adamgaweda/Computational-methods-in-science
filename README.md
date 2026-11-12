# MOwNiT — Metody Obliczeniowe w Nauce i Technice

Repozytorium zawiera implementacje algorytmów numerycznych, sprawozdania oraz projekty realizowane w ramach przedmiotu **Metody Obliczeniowe w Nauce i Technice** (MOwNiT).

---

## 📂 Zawartość repozytorium

| Moduł / Katalog | Tematyka i opis | Kluczowe pliki |
|---|---|---|
| **[Gauss-Jordan](./Gauss-Jordan/)** | Układy równań liniowych: eliminacja Gaussa-Jordana z częściowym wyborem elementu głównego, rozkład LU oraz zastosowanie do analizy obwodów elektrycznych (prawa Kirchhoffa). | `Gauss-Jordan.py`, `lu_factoralization.py`, `Kirchhoff_law.py`, raporty Jupyter |
| **[Annaeling](./Annaeling/)** | Heurystyczna optymalizacja metodą symulowanego wyżarzania (Simulated Annealing) zastosowana do problemu komiwojażera (TSP), rozwiązywania plansz Sudoku oraz odszumiania obrazów binarnych. | `TSP.py`, `sudoku.py`, `binary_image.py`, raporty Jupyter |
| **[Wyszukiwarka](./Wyszukiwarka/)** | Wyszukiwarka dokumentów oparta na analizie semantycznej LSI (*Latent Semantic Indexing*): macierz częstości TF-IDF, redukcja wymiarowości z dekompozycją SVD oraz interfejs webowy. | `app.py`, `LatentSemanticIndexing.py`, moduły `src/`, frontend `client/` |
| **[PageRank](./PageRank/)** | Algorytm rankingu węzłów sieci Google PageRank bazujący na łańcuchach Markowa i wyznaczaniu wektora stacjonarnego macierzy stochastycznej. | `PageRank.py`, `PageRank_report.ipynb` |
| **[SpectralDecomposition](./SpectralDecomposition/)** | Wyznaczanie wartości i wektorów własnych macierzy za pomocą metody potęgowej oraz odwrotnej metody potęgowej ze skalowaniem. | `power_iterations.py`, `reversed_power.py` |
| **[QR](./QR/)** | Ortogonalizacja i faktoryzacja macierzy: klasyczny algorytm Grama-Schmidta (CGS) oraz rozkład QR. | `CGS.py` |
| **[SVD](./SVD/)** | Dekompozycja według wartości osobliwych (*Singular Value Decomposition*) i jej numeryczne własności. | `zad1.py` |
| **[Discrete Fourier Transform](./Discrete%20Fourier%20Transform/)** | Transformacje częstotliwościowe sygnałów — Dyskretna Transformata Kosinusowa (DCT). | `DCT.py` |
| **[Numerical Integration](./Numerical%20Integration/)** | Numeryczne obliczanie całek oznaczonych z wykorzystaniem kwadratur Newtona-Cotesa (metody prostokątów, trapezów, Simpsona). | `Integral_type.py` |

---

## 🛠️ Technologie i narzędzia

- **Język:** Python 3.x
- **Biblioteki obliczeniowe i numeryczne:** NumPy, SciPy
- **Analiza danych i wizualizacja:** Matplotlib, Jupyter Notebook
- **Aplikacja webowa:** Python / HTML / CSS / JS (dla modułu wyszukiwarki LSI)

---

## 🚀 Uruchomienie

### 1. Wymagania wstępne
Upewnij się, że masz zainstalowanego Pythona (wersja 3.10 lub nowsza).

### 2. Utworzenie i aktywacja wirtualnego środowiska

```bash
# Utworzenie środowiska
python -m venv .venv

# Aktywacja (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Aktywacja (Linux / macOS)
source .venv/bin/activate
```

### 3. Instalacja podstawowych zależności

```bash
pip install numpy scipy matplotlib jupyter
```

### 4. Uruchamianie notebooków lub skryptów

- Aby przeglądać raporty i analizy:
  ```bash
  jupyter notebook
  ```
- Aby uruchomić pojedynczy skrypt numeryczny (np. eliminacja Gaussa-Jordana):
  ```bash
  python "Gauss-Jordan/Gauss-Jordan.py"
  ```
