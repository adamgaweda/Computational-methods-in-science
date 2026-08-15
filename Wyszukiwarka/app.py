"""
Serwer Flask dla wyszukiwarki semantycznej.

Wystawia REST API do wyszukiwania dokumentów oraz serwuje
statyczny interfejs webowy z katalogu `client/`.
"""

import os
import sqlite3
import threading
import time

import nltk
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from nltk.corpus import stopwords

from src.preprocessing import apply_idf
from src.search import cosine_search, lsi_search
from src.setup import get_manager

# ---------------------------------------------------------------------------
# Inicjalizacja NLTK
# ---------------------------------------------------------------------------

def _ensure_nltk_data():
    try:
        stopwords.words("english")
    except LookupError:
        print("Pobieranie danych NLTK...")
        nltk.download("stopwords")
        print("NLTK – pobrano pomyślnie.")


# ---------------------------------------------------------------------------
# Aplikacja Flask
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder="client")
CORS(app)

# Stan silnika wyszukiwarki (protected by _lock)
_lock = threading.Lock()
_engine_state = {
    "ready": False,
    "initializing": False,
    "use_svd": True,
    "k_svd": 500,
    "k_results": 10,
    "error": None,
}

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "simplified_wiki_index.db",
)


# ---------------------------------------------------------------------------
# Inicjalizacja silnika (w tle)
# ---------------------------------------------------------------------------

def _init_engine():
    """Inicjalizuje komponenty silnika wyszukiwarki w osobnym wątku."""
    with _lock:
        if _engine_state["ready"] or _engine_state["initializing"]:
            return
        _engine_state["initializing"] = True
        _engine_state["error"] = None

    try:
        print("[engine] Inicjalizacja...")
        mgr = get_manager()
        mgr.initialize()

        if _engine_state["use_svd"]:
            k = _engine_state["k_svd"]
            print(f"[engine] Ładowanie SVD k={k}...")
            mgr.load_svd(k)

        with _lock:
            _engine_state["ready"] = True
            _engine_state["initializing"] = False
        print("[engine] Gotowy.")

    except Exception as exc:
        with _lock:
            _engine_state["initializing"] = False
            _engine_state["error"] = str(exc)
        print(f"[engine] BŁĄD: {exc}")


# ---------------------------------------------------------------------------
# Endpointy API
# ---------------------------------------------------------------------------

@app.route("/api/status")
def api_status():
    """Zwraca bieżący stan silnika wyszukiwarki."""
    with _lock:
        state = dict(_engine_state)

    mgr = get_manager()

    response = {
        "status": "ready" if state["ready"] else ("initializing" if state["initializing"] else "idle"),
        "use_svd": state["use_svd"],
        "k_svd": state["k_svd"],
        "k_results": state["k_results"],
        "error": state["error"],
    }

    if state["ready"]:
        response["message"] = "Silnik gotowy"
        response["vocab_size"] = len(mgr.index)
        response["n_docs"] = len(mgr.doc_ids)
    elif state["initializing"]:
        response["message"] = "Inicjalizacja w toku..."
    else:
        response["message"] = "Silnik nie zainicjalizowany"

    return jsonify(response)


@app.route("/api/config", methods=["POST"])
def api_config():
    """Aktualizuje konfigurację silnika i restartuje inicjalizację."""
    with _lock:
        if _engine_state["initializing"]:
            return jsonify({"error": "Silnik jest w trakcie inicjalizacji"}), 409

    payload = request.get_json() or {}
    changed = False

    with _lock:
        if "use_svd" in payload:
            _engine_state["use_svd"] = bool(payload["use_svd"])
            changed = True
        if "k_svd" in payload and isinstance(payload["k_svd"], int) and payload["k_svd"] > 0:
            _engine_state["k_svd"] = payload["k_svd"]
            changed = True
        if "k_results" in payload and isinstance(payload["k_results"], int) and payload["k_results"] > 0:
            _engine_state["k_results"] = payload["k_results"]

        if changed:
            _engine_state["ready"] = False

    if changed:
        threading.Thread(target=_init_engine, daemon=True).start()

    with _lock:
        state = dict(_engine_state)

    return jsonify({
        "status": "config_updated",
        "use_svd": state["use_svd"],
        "k_svd": state["k_svd"],
        "k_results": state["k_results"],
    })


@app.route("/api/search")
def api_search():
    """Endpoint wyszukiwania.

    Query params:
        q (str): Zapytanie użytkownika.

    Returns:
        JSON z listą wyników lub błędem.
    """
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Parametr 'q' jest wymagany"}), 400

    with _lock:
        state = dict(_engine_state)

    if not state["ready"]:
        if not state["initializing"]:
            threading.Thread(target=_init_engine, daemon=True).start()
        return jsonify({"error": "Silnik jest w trakcie inicjalizacji, spróbuj ponownie za chwilę"}), 503

    try:
        mgr = get_manager()
        k = state["k_results"]

        t0 = time.perf_counter()
        if state["use_svd"]:
            svd = mgr.load_svd(state["k_svd"])
            results = lsi_search(query, svd, mgr.index, mgr.doc_ids, mgr.idf_weights, k)
        else:
            results = cosine_search(query, mgr.doc_matrix, mgr.index, mgr.doc_ids, mgr.idf_weights, k)
        elapsed = time.perf_counter() - t0

        formatted = []
        if results and results[0][1] > 0.001:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            for doc_id, sim in results:
                if sim > 0.001:
                    cur.execute("SELECT title, url FROM pages WHERE id = ?", (doc_id,))
                    row = cur.fetchone()
                    if row:
                        formatted.append({
                            "doc_id": doc_id,
                            "similarity": sim,
                            "title": row[0],
                            "url": row[1],
                        })
            conn.close()

        return jsonify({
            "query": query,
            "time": elapsed,
            "results_count": len(formatted),
            "results": formatted,
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/stats")
def api_stats():
    """Zwraca statystyki indeksu (nowy endpoint)."""
    mgr = get_manager()
    if not mgr.ready:
        return jsonify({"error": "Silnik nie jest gotowy"}), 503

    return jsonify({
        "vocab_size": len(mgr.index),
        "n_docs": len(mgr.doc_ids),
        "svd_cached_k": list(mgr.svd_cache.keys()),
    })


@app.route("/api/document/<int:doc_id>")
def api_document(doc_id: int):
    """Pobiera pełną treść dokumentu po ID."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT id, title, url, content FROM pages WHERE id = ?", (doc_id,))
        row = cur.fetchone()
        conn.close()

        if row:
            return jsonify({"id": row[0], "title": row[1], "url": row[2], "content": row[3]})
        return jsonify({"error": f"Dokument {doc_id} nie istnieje"}), 404

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Serwowanie interfejsu webowego
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("client", "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("client", path)


# ---------------------------------------------------------------------------
# Uruchomienie
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _ensure_nltk_data()
    print("Uruchamianie API na http://localhost:5000")
    threading.Thread(target=_init_engine, daemon=True).start()
    app.run(debug=False)