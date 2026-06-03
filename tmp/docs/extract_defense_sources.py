from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from docx import Document
from pypdf import PdfReader


ROOT = Path("/Users/Morty/Desktop/毕业设计")
PDF = ROOT / "答辩ppt 傅懋杰.pdf"
PPTX = ROOT / "答辩ppt 傅懋杰.pptx"
DOCX = ROOT / "個人陳述 講稿.docx"
OUT = Path("tmp/docs")


def clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf() -> None:
    reader = PdfReader(str(PDF))
    chunks = []
    for idx, page in enumerate(reader.pages, start=1):
        text = clean(page.extract_text() or "")
        chunks.append(f"=== PDF PAGE {idx} ===\n{text}")
    (OUT / "defense_ppt_pdf_text.txt").write_text("\n\n".join(chunks), encoding="utf-8")


def slide_order(zf: zipfile.ZipFile) -> list[str]:
    ns = {
        "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }
    pres = ET.fromstring(zf.read("ppt/presentation.xml"))
    rels = ET.fromstring(zf.read("ppt/_rels/presentation.xml.rels"))
    rid_to_target = {}
    for rel in rels:
        rid = rel.attrib.get("Id")
        target = rel.attrib.get("Target", "")
        if target.startswith("slides/"):
            rid_to_target[rid] = "ppt/" + target
    slides = []
    for sld_id in pres.find("p:sldIdLst", ns):
        rid = sld_id.attrib.get(f"{{{ns['r']}}}id")
        if rid in rid_to_target:
            slides.append(rid_to_target[rid])
    return slides


def extract_pptx() -> None:
    ns = {
        "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
        "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    }
    chunks = []
    with zipfile.ZipFile(PPTX) as zf:
        for idx, slide_path in enumerate(slide_order(zf), start=1):
            root = ET.fromstring(zf.read(slide_path))
            texts = []
            for tx_body in root.findall(".//p:txBody", ns):
                lines = []
                for para in tx_body.findall("a:p", ns):
                    runs = [t.text or "" for t in para.findall(".//a:t", ns)]
                    if runs:
                        lines.append("".join(runs))
                if lines:
                    texts.append("\n".join(lines))
            chunks.append(f"=== PPTX SLIDE {idx} ===\n{clean(chr(10).join(texts))}")
    (OUT / "defense_pptx_text.txt").write_text("\n\n".join(chunks), encoding="utf-8")


def extract_docx() -> None:
    doc = Document(str(DOCX))
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    (OUT / "original_script_text.txt").write_text("\n\n".join(paras), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    extract_pdf()
    extract_pptx()
    extract_docx()


if __name__ == "__main__":
    main()
