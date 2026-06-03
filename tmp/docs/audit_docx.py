from __future__ import annotations

import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET


NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}


def qn(ns: str, tag: str) -> str:
    return f"{{{NS[ns]}}}{tag}"


def elem_text(elem: ET.Element) -> str:
    parts: list[str] = []
    for node in elem.iter():
        if node.tag in {qn("w", "t"), qn("m", "t"), qn("w", "instrText")}:
            parts.append(node.text or "")
        elif node.tag == qn("w", "tab"):
            parts.append("\t")
        elif node.tag == qn("w", "br"):
            parts.append("\n")
    text = "".join(parts)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def para_style(p: ET.Element) -> str:
    ppr = p.find("w:pPr", NS)
    if ppr is None:
        return ""
    pstyle = ppr.find("w:pStyle", NS)
    if pstyle is None:
        return ""
    return pstyle.attrib.get(qn("w", "val"), "")


def has_superscript(p: ET.Element) -> bool:
    for vert in p.findall(".//w:vertAlign", NS):
        if vert.attrib.get(qn("w", "val"), "") == "superscript":
            return True
    return False


def collect_document(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        doc = ET.fromstring(zf.read("word/document.xml"))
        app = ET.fromstring(zf.read("docProps/app.xml"))
        core = ET.fromstring(zf.read("docProps/core.xml"))
        namelist = zf.namelist()

        paras: list[dict] = []
        tables: list[dict] = []
        body = doc.find("w:body", NS)
        if body is None:
            raise RuntimeError("missing document body")

        block_idx = 0
        for child in list(body):
            if child.tag == qn("w", "p"):
                text = elem_text(child)
                if text:
                    paras.append(
                        {
                            "block": block_idx,
                            "type": "p",
                            "style": para_style(child),
                            "text": text,
                            "superscript": has_superscript(child),
                        }
                    )
            elif child.tag == qn("w", "tbl"):
                rows = []
                for tr in child.findall("w:tr", NS):
                    row = []
                    for tc in tr.findall("w:tc", NS):
                        row.append(elem_text(tc))
                    if any(cell.strip() for cell in row):
                        rows.append(row)
                tables.append({"block": block_idx, "rows": rows})
                flat = " | ".join(" / ".join(row) for row in rows[:3])
                if flat.strip():
                    paras.append(
                        {
                            "block": block_idx,
                            "type": "table",
                            "style": "",
                            "text": flat[:2000],
                            "superscript": False,
                        }
                    )
            block_idx += 1

        all_text = "\n".join(p["text"] for p in paras)
        images = [n for n in namelist if n.startswith("word/media/")]

        def text_of_xml(name: str) -> str:
            if name not in namelist:
                return ""
            return elem_text(ET.fromstring(zf.read(name)))

        headers = {n: text_of_xml(n) for n in namelist if n.startswith("word/header")}
        footers = {n: text_of_xml(n) for n in namelist if n.startswith("word/footer")}
        footnotes = text_of_xml("word/footnotes.xml")
        endnotes = text_of_xml("word/endnotes.xml")

    refs_start = None
    for i, p in enumerate(paras):
        if re.fullmatch(r"\s*参考文献\s*", p["text"]):
            refs_start = i
            break
    refs = []
    if refs_start is not None:
        for p in paras[refs_start + 1 :]:
            t = p["text"].strip()
            if re.match(r"^\[\d+\]", t):
                refs.append(t)

    citations = []
    for p in paras:
        for m in re.finditer(r"\[(\d+(?:[-,，]\d+)*)\]", p["text"]):
            citations.append({"block": p["block"], "text": m.group(0), "para": p["text"][:220]})

    headings = []
    for idx, p in enumerate(paras):
        text = p["text"].strip()
        style = p["style"]
        if style.startswith("Heading") or re.match(r"^(第[一二三四五六七八九十]+章|[0-9]+(?:\.[0-9]+){0,3}\s+)", text):
            headings.append({"index": idx, "block": p["block"], "style": style, "text": text})

    captions = [
        {"index": i, "block": p["block"], "text": p["text"]}
        for i, p in enumerate(paras)
        if re.match(r"^(图|表)\s*\d", p["text"].strip())
    ]

    terms = Counter()
    for term in [
        "GRU",
        "RUL",
        "DDQN",
        "DQN",
        "THDQN",
        "PPO",
        "POMCP",
        "DDT",
        "slack",
        "Differential_pressure",
        "two-hierarchy",
        "Temporal",
        "MDPI",
        "右设限",
        "补齐",
    ]:
        terms[term] = len(re.findall(re.escape(term), all_text, flags=re.IGNORECASE))

    styles = Counter(p["style"] for p in paras)
    duplicate_headings = [h for h, c in Counter(x["text"] for x in headings).items() if c > 1]

    return {
        "path": str(path),
        "paragraph_count": len(paras),
        "table_count": len(tables),
        "image_count": len(images),
        "headings": headings,
        "captions": captions,
        "references": refs,
        "reference_count": len(refs),
        "citations": citations,
        "citation_count": len(citations),
        "styles": dict(styles),
        "terms": dict(terms),
        "duplicate_headings": duplicate_headings,
        "headers": headers,
        "footers": footers,
        "footnotes": footnotes,
        "endnotes": endnotes,
        "paragraphs": paras,
        "tables": tables,
    }


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: audit_docx.py INPUT.docx OUTDIR")
    path = Path(sys.argv[1])
    outdir = Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    data = collect_document(path)
    stem = "thesis_2_0"

    (outdir / f"{stem}_audit.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (outdir / f"{stem}_paragraphs.txt").open("w", encoding="utf-8") as f:
        for i, p in enumerate(data["paragraphs"]):
            f.write(f"[{i:04d}] block={p['block']} type={p['type']} style={p['style']} sup={p['superscript']}\n")
            f.write(p["text"] + "\n\n")
    with (outdir / f"{stem}_headings.txt").open("w", encoding="utf-8") as f:
        for h in data["headings"]:
            f.write(f"{h['index']:04d}\t{h['style']}\t{h['text']}\n")
    with (outdir / f"{stem}_references.txt").open("w", encoding="utf-8") as f:
        for r in data["references"]:
            f.write(r + "\n")
    with (outdir / f"{stem}_captions.txt").open("w", encoding="utf-8") as f:
        for c in data["captions"]:
            f.write(f"{c['index']:04d}\t{c['text']}\n")

    print(
        json.dumps(
            {
                "paragraph_count": data["paragraph_count"],
                "table_count": data["table_count"],
                "image_count": data["image_count"],
                "heading_count": len(data["headings"]),
                "caption_count": len(data["captions"]),
                "reference_count": data["reference_count"],
                "citation_count": data["citation_count"],
                "terms": data["terms"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
