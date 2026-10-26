"""
Build the textbook passages the hinter cites (Role A, week 1).

Downloads OpenStax Prealgebra 2e and Elementary Algebra 2e (CC BY 4.0) section by
section from the OpenStax archive, keeps the teaching text (drops end-of-section
exercises, "Try It" and "Be Prepared" boxes), turns MathML into plain text, and
splits each section at its subsection headings into passages of at most ~250 words.

Every passage carries the reference the hinter cites, e.g.
  "Prealgebra 2e, 4.2 Multiply and Divide Fractions > Simplify Fractions"

Downloads are cached in data/openstax/raw/ (git-ignored), so re-running is offline.

Usage: python -m matheasy.retrieval.openstax  -> data/openstax/passages.jsonl
"""

import json
import re
import time
import urllib.request
from pathlib import Path

from bs4 import BeautifulSoup

ARCHIVE = "https://openstax.org/apps/archive/20260604.144757/contents"
BOOKS = {  # title -> (book id, version), pinned so the passages never change under us
    "Prealgebra 2e": ("f0fa90be-fca8-43c9-9aad-715c0a2cee2b", "38cae45"),
    "Elementary Algebra 2e": ("55931856-c627-418b-a56f-1dd0007683a8", "38cae45"),
}
MAX_WORDS = 250

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "openstax" / "raw"
OUT = ROOT / "data" / "openstax" / "passages.jsonl"

# Blocks that are practice, not teaching text.
DROP = ["div.os-eos", "section.section-exercises", "div.try", "div.be-prepared",
        "div.manipulative-math", "figure", "annotation-xml", "div[data-type=media]"]


def fetch(url, path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "MathEasy course project"})
        path.write_bytes(urllib.request.urlopen(req, timeout=60).read())
        time.sleep(0.3)
    return json.loads(path.read_text(encoding="utf-8"))


def plain(html):
    return re.sub(r"\s+", " ", BeautifulSoup(html, "html.parser").get_text(" ")).strip()


def math_text(node):
    """MathML -> readable text: 3/4, x^2, sqrt(9)."""
    if isinstance(node, str):
        return node.strip()
    kids = [math_text(c) for c in node.children if not (isinstance(c, str) and not c.strip())]
    if node.name == "mfrac" and len(kids) == 2:
        return f"{kids[0]}/{kids[1]}"
    if node.name in ("msup", "msubsup") and len(kids) >= 2:
        return f"{kids[0]}^{kids[-1]}"
    if node.name == "msqrt":
        return f"sqrt({''.join(kids)})"
    return "".join(kids)


def sections(book_title, book_id, version):
    """Yield (number, title, html) for every numbered section, in book order."""
    toc = fetch(f"{ARCHIVE}/{book_id}@{version}.json", RAW / f"{book_id}.json")

    def walk(node):
        for child in node.get("contents", []):
            title = plain(child["title"])
            m = re.match(r"(\d+\.\d+) (.+)", title)
            if m:
                page = child["id"].split("@")[0]
                data = fetch(f"{ARCHIVE}/{book_id}@{version}:{page}.json", RAW / book_id / f"{page}.json")
                yield m.group(1), m.group(2), data["content"]
            yield from walk(child)

    yield from walk(toc["tree"])


def passages(book, number, title, html):
    soup = BeautifulSoup(html, "html.parser")
    for sel in DROP:
        for tag in soup.select(sel):
            tag.decompose()
    for m in soup.find_all("math"):
        m.replace_with(f" {math_text(m)} ")
    for t in soup.select("[data-type=document-title], .os-title-label"):
        t.decompose()

    # Walk headings and paragraphs in order; each h3 starts a new subsection.
    sub, words, out = "Introduction", [], []

    def flush():
        if len(words) < 10:  # empty or stray heading-only subsections
            return
        for i in range(0, len(words), MAX_WORDS):
            out.append({"book": book, "section": number, "section_title": title, "subsection": sub,
                        "ref": f"{book}, {number} {title} > {sub}",
                        "text": " ".join(words[i:i + MAX_WORDS])})

    for el in soup.find_all(["h3", "p", "li", "td"]):
        if el.find_parent(["li", "td"]) and el.name == "p":
            continue  # already counted through its list item or table cell
        text = re.sub(r"\s+", " ", el.get_text(" ")).strip()
        if el.name == "h3" and el.parent.name == "section":  # not a note or example title
            flush()
            sub, words = text, []
        elif text and sub != "Learning Objectives" and not text.startswith("Example "):
            words += text.split()
    flush()
    return out


if __name__ == "__main__":
    rows = []
    for book, (book_id, version) in BOOKS.items():
        n_sections = 0
        for number, title, html in sections(book, book_id, version):
            rows += passages(book, number, title, html)
            n_sections += 1
        print(f"{book}: {n_sections} sections")
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"{len(rows)} passages -> {OUT}")
