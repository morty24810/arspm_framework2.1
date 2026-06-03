# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path("/Users/Morty/Desktop/毕业设计/Code/RUL_GRU/arspm_framework2.1")
REFS = ROOT / "tmp/docs/thesis_2_0_refs_only.txt"
OUT = ROOT / "tmp/docs/reference_metadata_crossref.json"


def parse_refs():
    refs = []
    for line in REFS.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\[(\d+)\]\s+(.*)$", line.strip())
        if m:
            refs.append((int(m.group(1)), m.group(2)))
    return refs


def get_json(url: str):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "thesis-reference-check/1.0 (mailto:example@example.com)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    results = {}
    for idx, ref in parse_refs():
        query = re.sub(r"\[[A-Z/]+\]|\[.*?\]|DOI:.*$", "", ref).strip()
        url = "https://api.crossref.org/works?rows=3&query.bibliographic=" + urllib.parse.quote(query)
        try:
            data = get_json(url)
            items = data.get("message", {}).get("items", [])
            simplified = []
            for it in items:
                simplified.append({
                    "title": it.get("title", []),
                    "container-title": it.get("container-title", []),
                    "published": it.get("published-print") or it.get("published-online") or it.get("issued"),
                    "DOI": it.get("DOI"),
                    "URL": it.get("URL"),
                    "type": it.get("type"),
                    "score": it.get("score"),
                    "abstract": it.get("abstract"),
                    "author": [
                        {"given": a.get("given"), "family": a.get("family")}
                        for a in it.get("author", [])[:5]
                    ],
                })
            results[str(idx)] = {"ref": ref, "query": query, "items": simplified}
            print(idx, simplified[0].get("DOI") if simplified else "NOHIT")
        except Exception as e:
            results[str(idx)] = {"ref": ref, "query": query, "error": repr(e)}
            print(idx, "ERR", repr(e))
        time.sleep(0.25)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
