# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path("/Users/Morty/Desktop/毕业设计/Code/RUL_GRU/arspm_framework2.1")
OUT = ROOT / "tmp/docs/reference_metadata_openalex.json"

DOI = {
    2: "10.1016/j.asoc.2020.106208",
    3: "10.1016/j.cie.2021.107489",
    4: "10.1038/s41598-024-79593-8",
    5: "10.1016/j.rcim.2022.102412",
    6: "10.1016/j.ress.2025.111394",
    7: "10.1016/j.cirpj.2024.02.003",
    9: "10.1038/nature14236",
    14: "10.1007/s12599-014-0334-4",
    15: "10.1007/s10796-022-10252-x",
    16: "10.1080/00207543.2022.2122620",
    17: "10.1016/j.jmsy.2024.05.021",
    18: "10.1016/j.iswa.2025.200501",
    19: "10.1016/j.cie.2020.106889",
    20: "10.1016/j.cie.2019.106024",
    21: "10.1016/j.jmsy.2022.05.010",
    22: "10.1016/j.ymssp.2024.111120",
    23: "10.1016/j.ymssp.2017.11.016",
    24: "10.1162/neco.1997.9.8.1735",
    25: "10.3115/v1/d14-1179",
    27: "10.1016/j.asoc.2023.110419",
    28: "10.1007/s12206-024-0402-8",
    29: "10.1038/s41598-022-10191-2",
    30: "10.1016/j.ymssp.2023.110221",
    32: "10.1287/opre.25.1.45",
    33: "10.1287/mnsc.33.8.1035",
    34: "10.1007/bf02238804",
    35: "10.1007/bf02023073",
    36: "10.1002/(sici)1099-1425(200001/02)3:1<3::aid-jos32>3.0.co;2-y",
    37: "10.1109/tsmcc.2002.1009117",
    38: "10.1007/s40747-024-01374-7",
    39: "10.1016/j.ejor.2023.05.017",
    40: "10.1016/j.cie.2022.108786",
    41: "10.1007/s10845-025-02585-6",
    42: "10.1109/tcyb.2022.3151855",
    43: "10.1016/j.jmsy.2021.09.018",
    44: "10.1016/j.cor.2023.106365",
    45: "10.1016/j.ress.2024.110129",
    46: "10.1016/j.jclepro.2025.145856",
    47: "10.1016/s0004-3702(99)00052-1",
    48: "10.1023/a:1022676722315",
    49: "10.1609/aaai.v30i1.10295",
}

TITLE_QUERY = {
    1: "Recommendations for implementing the strategic initiative INDUSTRIE 4.0 final report",
    8: "Reinforcement Learning An Introduction Sutton Barto second edition",
    10: "Proximal Policy Optimization Algorithms Schulman Wolski Dhariwal Radford Klimov",
    11: "Monte-Carlo planning in large POMDPs Silver Veness",
    12: "Creation of Publicly Available Data Sets for Prognostics and Diagnostics Addressing Data Scenarios Relevant to Industrial Applications",
    13: "Preventive to predictive maintenance PrognosticsHSE Kaggle",
    26: "Empirical evaluation of gated recurrent neural networks on sequence modeling Chung Gulcehre Cho Bengio",
    31: "Scheduling Theory Algorithms and Systems Pinedo 5th edition",
}


def get_json(url: str):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "thesis-reference-check/1.0 (mailto:example@example.com)"},
    )
    with urllib.request.urlopen(req, timeout=25) as resp:
        return json.loads(resp.read().decode("utf-8"))


def reconstruct_abstract(inv):
    if not inv:
        return None
    positions = []
    for word, locs in inv.items():
        for loc in locs:
            positions.append((int(loc), word))
    return " ".join(word for _, word in sorted(positions))


def simplify(work):
    if not work:
        return None
    return {
        "id": work.get("id"),
        "doi": work.get("doi"),
        "title": work.get("title"),
        "publication_year": work.get("publication_year"),
        "source": ((work.get("primary_location") or {}).get("source") or {}).get("display_name"),
        "type": work.get("type"),
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        "concepts": [
            c.get("display_name")
            for c in (work.get("concepts") or [])[:8]
            if c.get("display_name")
        ],
        "open_access": work.get("open_access"),
        "url": work.get("doi") or work.get("id"),
    }


def search_title(query):
    url = "https://api.openalex.org/works?search=" + urllib.parse.quote(query) + "&per-page=3"
    data = get_json(url)
    results = data.get("results") or []
    return results[0] if results else None


def main():
    out = {}
    for i in range(1, 50):
        try:
            if i in DOI:
                url = "https://api.openalex.org/works/https://doi.org/" + urllib.parse.quote(DOI[i], safe="")
                work = get_json(url)
            else:
                work = search_title(TITLE_QUERY.get(i, ""))
            out[str(i)] = simplify(work)
            print(i, (out[str(i)] or {}).get("title"))
        except Exception as e:
            out[str(i)] = {"error": repr(e)}
            print(i, "ERR", repr(e))
        time.sleep(0.2)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
