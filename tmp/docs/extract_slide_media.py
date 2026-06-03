from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

PPTX = Path("/Users/Morty/Desktop/毕业设计/答辩ppt 傅懋杰.pptx")
OUT = Path("tmp/docs/slide_media")


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
    return [
        rid_to_target[sld.attrib[f"{{{ns['r']}}}id"]]
        for sld in pres.find("p:sldIdLst", ns)
    ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(PPTX) as zf:
        slides = slide_order(zf)
        for idx, slide_path in enumerate(slides, start=1):
            rel_path = Path(slide_path).parent / "_rels" / (Path(slide_path).name + ".rels")
            if str(rel_path) not in zf.namelist():
                continue
            rels = ET.fromstring(zf.read(str(rel_path)))
            slide_dir = OUT / f"slide_{idx:02d}"
            slide_dir.mkdir(exist_ok=True)
            copied = []
            for rel in rels:
                typ = rel.attrib.get("Type", "")
                target = rel.attrib.get("Target", "")
                if "image" not in typ:
                    continue
                media_path = (Path(slide_path).parent / target).as_posix()
                while "/../" in media_path:
                    media_path = media_path.replace("/slides/../", "/")
                if media_path in zf.namelist():
                    dest = slide_dir / Path(media_path).name
                    dest.write_bytes(zf.read(media_path))
                    copied.append(dest.name)
            if copied:
                print(f"slide {idx}: " + ", ".join(copied))


if __name__ == "__main__":
    if OUT.exists():
        shutil.rmtree(OUT)
    main()
