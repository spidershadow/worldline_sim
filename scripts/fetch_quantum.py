from __future__ import annotations
"""fetch_quantum.py – pull two quantum‑progress metrics and save CSVs.

1. max_qubits.csv         – largest announced qubit device each year
2. quantum_volume.csv     – best Quantum Volume record per year

If scraping fails (offline, layout change) the script uses hard‑coded
fallback dictionaries so the simulator still has data.

Run:  python fetch_quantum.py
Requires: pandas, requests, beautifulsoup4
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dump_csv(d: dict[int, int | float], fname: str, col: str) -> None:
    (pd.Series(d, name=col).sort_index().to_csv(DATA_DIR / fname, header=True))


# ---------------------------------------------------------------------------
# 1) Largest announced qubit device per year
# ---------------------------------------------------------------------------

def scrape_max_qubits() -> dict[int, int]:
    url = "https://en.wikipedia.org/wiki/Timeline_of_quantum_computing"
    html = requests.get(url, timeout=30).text
    tables = pd.read_html(html)
    records: dict[int, int] = {}
    for t in tables:
        # Look for any cell containing the substring 'qubit'.
        if not any(t.astype(str).apply(lambda s: s.str.contains("qubit", case=False, na=False)).any()):
            continue
        # Ensure column labels are strings before lower‑casing.
        t.columns = [str(c).lower() for c in t.columns]
        if "qubits" not in t.columns and "qubit" not in t.columns:
            continue
        qcol = "qubits" if "qubits" in t.columns else "qubit"
        if "year" not in t.columns:
            t["year"] = t.get("date", "").astype(str).str.extract(r"(\d{4})").astype(float)
        t = t[["year", qcol]].dropna()
        t[qcol] = t[qcol].astype(str).str.extract(r"(\d+)").astype(float)
        for _, row in t.iterrows():
            y, q = int(row["year"]), int(row[qcol])
            if q > 0:
                records[y] = max(records.get(y, 0), q)
    if not records:
        raise RuntimeError("no qubit rows parsed")
    return records


# ---------------------------------------------------------------------------
# 2) Quantum Volume record per year
# ---------------------------------------------------------------------------

def scrape_quantum_volume() -> dict[int, int]:
    wiki_url = "https://en.wikipedia.org/wiki/Quantum_volume"
    try:
        soup = BeautifulSoup(requests.get(wiki_url, timeout=30).text, "html.parser")
        tables = soup.select("table.wikitable")
        for tab in tables:
            if "Quantum Volume" in tab.get_text():
                df = pd.read_html(str(tab))[0]
                break
        else:
            raise ValueError
        df.columns = [c.lower() for c in df.columns]
        qcol = "quantum volume" if "quantum volume" in df.columns else "qv"
        df[qcol] = df[qcol].astype(str).str.replace(",", "").str.extract(r"(\d+)").astype(float)
        df["year"] = df["year"].astype(int)
        out = df.groupby("year")[qcol].max().dropna().astype(int).to_dict()
        if out:
            return out
    except Exception:
        pass  # fall through to IBM blog fallback

    # IBM blog JSON blob fallback
    txt = requests.get("https://research.ibm.com/blog/quantum-volume", timeout=30).text
    blob = re.search(r"QV_RECORDS\s*=\s*(\[[^\]]+\])", txt, re.S)
    if not blob:
        blob = re.search(r"qv_records\s*=\s*(\[[^\]]+\])", txt, re.S)
    if not blob:
        raise RuntimeError("QV records not found in IBM blog")
    records = json.loads(blob.group(1))
    return {int(rec["year"]): int(rec["v"]) for rec in records}


# ---------------------------------------------------------------------------
# Fallback mini‑datasets (good through early‑2025)
# ---------------------------------------------------------------------------

FALLBACK_QUBITS = {
    1998: 2, 2000: 5, 2001: 7, 2017: 50, 2019: 53, 2020: 65,
    2021: 127, 2022: 133, 2023: 433, 2024: 1121,
}

FALLBACK_QV = {
    2017: 4, 2018: 8, 2019: 16, 2020: 128, 2021: 1024,
    2022: 4096, 2023: 16384,
}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        qdict = scrape_max_qubits()
        print(f"[max_qubits] scraped {len(qdict)} rows")
    except Exception as e:
        print(f"scrape_max_qubits failed ({e}); using fallback")
        qdict = FALLBACK_QUBITS
    _dump_csv(qdict, "max_qubits.csv", "max_qubits")

    try:
        qvdict = scrape_quantum_volume()
        print(f"[quantum_volume] scraped {len(qvdict)} rows")
    except Exception as e:
        print(f"scrape_quantum_volume failed ({e}); using fallback")
        qvdict = FALLBACK_QV
    _dump_csv(qvdict, "quantum_volume.csv", "quantum_volume")

    print("✅ quantum CSVs written to ./data/") 