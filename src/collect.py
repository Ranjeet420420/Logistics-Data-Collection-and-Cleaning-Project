"""
collect.py - Data collection layer.

Downloads public transportation datasets (vega-datasets: US flights + airports),
caches them under data/raw/, verifies they are non-empty and records a small
manifest (URL, size, SHA-256, timestamp) so the collection step is reproducible
and auditable.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

BASE = "https://raw.githubusercontent.com/vega/vega-datasets/main/data"
SOURCES = {
    "flights-20k.json": f"{BASE}/flights-20k.json",
    "airports.csv": f"{BASE}/airports.csv",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, retries: int = 3, timeout: int = 60) -> Path:
    """Download with simple retry + exponential backoff. Skips if cached."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[cache] {dest.name} already present")
        return dest

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            if not resp.content:
                raise ValueError("empty response body")
            dest.write_bytes(resp.content)
            print(f"[ok] downloaded {dest.name} ({len(resp.content):,} bytes)")
            return dest
        except (requests.RequestException, ValueError) as err:
            last_err = err
            wait = 2 ** attempt
            print(f"[retry {attempt}/{retries}] {dest.name}: {err} - waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Could not download {url}: {last_err}")


def collect_all() -> dict:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"collected_at_utc": datetime.now(timezone.utc).isoformat(), "files": []}
    for name, url in SOURCES.items():
        path = download(url, RAW_DIR / name)
        manifest["files"].append(
            {"file": name, "url": url, "bytes": path.stat().st_size, "sha256": sha256(path)}
        )
    (RAW_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    collect_all()
