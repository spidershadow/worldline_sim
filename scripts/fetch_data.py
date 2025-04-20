from __future__ import annotations

"""Utility script for downloading/refreshing observation CSVs.

Each *pattern* in the simulator can be constrained by real‑world data.  This
script automates the retrieval of such data and writes it into the
``data/`` directory in the exact format expected by ``Pattern`` subclasses:

    year,value

Usage (from repository root) – examples::

    python fetch_data.py --all            # refresh all patterns
    python fetch_data.py uap tech         # subset

The module can also be invoked via ``python -m worldline_sim.fetch_data``.
"""

import argparse
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import re
import requests

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent / "data"
_DATA_DIR.mkdir(exist_ok=True)

IntFetcher = Callable[[Path], bool]


def _write_df(df: pd.DataFrame, csv_name: str) -> None:
    """Write *df* (must contain year/value) to *_DATA_DIR/csv_name*."""
    out = _DATA_DIR / csv_name
    df.to_csv(out, index=False)
    print(f"[fetch] wrote {len(df)} rows → {out.relative_to(Path.cwd())}")

# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

def _fetch_uap(dest_dir: Path) -> bool:
    url = "https://nuforc.org/ndx/?id=event"
    try:
        tables = pd.read_html(url)
    except Exception as exc:
        print(f"[uap] failed: {exc}")
        return False
    if not tables:
        print("[uap] page layout changed?")
        return False
    df = tables[0].iloc[:, :2].copy()
    df.columns = ["ym", "monthly"]
    df["year"] = df["ym"].astype(str).str.split("/").str[0]
    df = df[df["year"].str.fullmatch(r"\d{4}")]
    df["year"] = df["year"].astype(int)
    df["monthly"] = pd.to_numeric(df["monthly"], errors="coerce")
    df = df.dropna(subset=["monthly"])
    yearly = df.groupby("year")["monthly"].sum().reset_index()
    _write_df(yearly.rename(columns={"monthly": "value"}), "uap_observations.csv")
    return True


def _fetch_tech(dest_dir: Path) -> bool:
    url = "https://ourworldindata.org/grapher/transistors-per-microprocessor.csv"
    try:
        df = pd.read_csv(url)
    except Exception as exc:
        print(f"[tech] failed: {exc}")
        return False
    df = df[df["Entity"] == "World"][["Year", "Transistors per microprocessor"]]
    df.columns = ["year", "value"]
    df["value"] = df["value"].astype(float).clip(lower=1).apply(np.log10)
    _write_df(df.sort_values("year"), "tech_observations.csv")
    return True


def _fetch_rng(dest_dir: Path) -> bool:
    import csv, io, requests
    from collections import defaultdict
    from datetime import date
    start = date(1998, 8, 4)
    today = date.today()
    yearly: dict[int, tuple[float, int]] = {}
    for yr in range(start.year, today.year + 1):
        url = (
            "https://global-mind.org/cgi-bin/basketran.cgi?"
            f"year={yr}&stime=00:00:00&etime=23:59:59&dtype=CSV&idate=No&gzip=No"
        )
        try:
            r = requests.get(url, stream=True, timeout=60)
            r.raise_for_status()
        except Exception as exc:
            print(f"[rng] HTTP {yr}: {exc}")
            continue
        reader = csv.reader(io.TextIOWrapper(r.raw, newline=""))
        per_day = defaultdict(lambda: (0.0, 0))
        for row in reader:
            if not row or row[0] != "13":
                continue
            try:
                samples = [float(x) for x in row[3:] if x]
                if not samples:
                    continue
                mean = sum(samples) / len(samples)
                d = date.fromisoformat(row[1]).timetuple().tm_yday
                s, n = per_day[d]
                per_day[d] = (s + mean, n + 1)
            except Exception:
                continue
        if not per_day:
            continue
        tot = sum(s / n for s, n in per_day.values() if n)
        yearly[yr] = (tot, len(per_day))
    if not yearly:
        return False
    df = pd.DataFrame({"year": list(yearly.keys()), "value": [t / d for t, d in yearly.values()]})
    _write_df(df.sort_values("year"), "rng_observations.csv")
    return True


def _fetch_events(dest_dir: Path) -> bool:
    url = "https://noosphere.princeton.edu/pred_formal.html"
    try:
        tables = pd.read_html(url, header=0)
    except Exception as exc:
        print(f"[events] failed: {exc}")
        return False
    if not tables:
        return False
    df = tables[0]
    df = df[df["Include"].astype(str).str.startswith("Yes")]
    df["begin_dt"] = pd.to_datetime(df["Begin Date/Time"], utc=True, errors="coerce")
    df = df.dropna(subset=["begin_dt"])
    yearly = df.groupby(df["begin_dt"].dt.year).size().reset_index(name="value")
    yearly.columns = ["year", "value"]
    _write_df(yearly.sort_values("year"), "events_observations.csv")
    return True


# New quantum pattern fetcher -------------------------------------------------

def _fetch_maxqubits(dest_dir: Path) -> bool:
    """Largest announced qubit count per year scraped from Wikipedia.

    We load the HTML 'Timeline of quantum computing' page, scan the
    Events column for phrases like 'xxx‑qubit' or 'xxx qubit', extract
    the largest integer per year.
    """
    url = "https://en.wikipedia.org/w/index.php?title=Timeline_of_quantum_computing&action=raw"
    import requests
    try:
        raw = requests.get(url, timeout=60).text.splitlines()
    except Exception as exc:
        print(f"[max_qubits] failed to fetch raw page: {exc}")
        return False

    records: dict[int, int] = {}
    for line in raw:
        if "qubit" not in line.lower():
            continue
        # Table rows often start with '|-' or '|', skip header lines
        if not line.lstrip().startswith("|"):
            continue
        # Split by '|' to get cells
        parts = [p.strip() for p in line.split('|') if p.strip()]
        if not parts:
            continue
        # First cell is year (might include notes footnote)
        try:
            year = int(re.match(r"(\d{4})", parts[0]).group(1))
        except Exception:
            continue
        text = " ".join(parts)
        m = re.search(r"(\d{2,5})\s*-?\s*qubit", text, re.I)
        if m:
            q = int(m.group(1))
            records[year] = max(q, records.get(year, 0))

    if not records:
        print("[max_qubits] no qubit counts found – abort")
        return False

    yearly = pd.DataFrame({"year": list(records.keys()), "value": list(records.values())})
    yearly.columns = ["year", "value"]
    _write_df(yearly.sort_values("year"), "max_qubits_observations.csv")
    return True

# Registry --------------------------------------------------------------------
_FETCHERS: Dict[str, IntFetcher] = {
    "uap": _fetch_uap,
    "tech": _fetch_tech,
    "rng": _fetch_rng,
    "events": _fetch_events,
    "max_qubits": _fetch_maxqubits,
}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Refresh observation CSVs for patterns")
    p.add_argument("patterns", nargs="*", choices=sorted(_FETCHERS.keys()), help="patterns to refresh")
    p.add_argument("--all", action="store_true", help="refresh ALL patterns")
    return p


def main(argv: List[str] | None = None):  # noqa: D401
    args = _build_argparser().parse_args(argv)
    targets = list(_FETCHERS) if (args.all or not args.patterns) else args.patterns
    for name in targets:
        print(f"[fetch] {name} …")
        ok = _FETCHERS[name](_DATA_DIR)
        print(f"[fetch] {name} … {'OK' if ok else 'FAILED'}\n")


if __name__ == "__main__":  # pragma: no cover
    main() 