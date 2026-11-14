#!/usr/bin/env python3
"""Plain text out of a document, on stdout: extract.py FILE
pdf (needs pdftotext), docx, odt, pptx, epub, html/htm, xml, and anything text-like.
Paragraph breaks are kept as blank lines so vecdb can chunk on them."""
import html, re, subprocess, sys, zipfile
from html.parser import HTMLParser

def strip_html(s):
    class P(HTMLParser):
        def __init__(self):
            super().__init__(); self.out = []; self.skip = 0
        def handle_starttag(self, t, a):
            if t in ("script", "style"): self.skip += 1
            if t in ("p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section"): self.out.append("\n\n" if t != "br" else "\n")
        def handle_endtag(self, t):
            if t in ("script", "style"): self.skip = max(0, self.skip - 1)
        def handle_data(self, d):
            if not self.skip: self.out.append(d)
    p = P(); p.feed(s)
    return "".join(p.out)

def xml_paras(data, ptag):
    """text of every <ptag> element (namespaces ignored), one paragraph each"""
    out = []
    for m in re.finditer(rb"<(?:\w+:)?%s[ >].*?</(?:\w+:)?%s>" % (ptag, ptag), data, re.S):
        t = re.sub(rb"<(?:\w+:)?tab/>", b" ", m.group(0))
        t = re.findall(rb"<(?:\w+:)?t(?: [^>]*)?>([^<]*)</(?:\w+:)?t>", t) if ptag == b"p" and b"w:t" in m.group(0) or b"a:t" in m.group(0) else [re.sub(rb"<[^>]+>", b"", t)]
        s = html.unescape(b"".join(t).decode("utf-8", "replace")).strip()
        if s: out.append(s)
    return "\n\n".join(out)

def main(path):
    low = path.lower()
    if low.endswith(".pdf"):
        r = subprocess.run(["pdftotext", "-enc", "UTF-8", path, "-"], capture_output=True)
        t = r.stdout.decode("utf-8", "replace")
        t = re.sub(r"(?<=\S)-\n(?=\w)", "", t)            # de-hyphenate line breaks
        t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)            # single newlines are just wrapping
        return t.replace("\x0c", "\n\n")
    if low.endswith(".docx"):
        z = zipfile.ZipFile(path); return xml_paras(z.read("word/document.xml"), b"p")
    if low.endswith((".odt", ".ods", ".odp")):
        z = zipfile.ZipFile(path); d = z.read("content.xml")
        d = re.sub(rb"</text:(?:p|h)>", b"\n\n", d); d = re.sub(rb"<[^>]+>", b"", d)
        return html.unescape(d.decode("utf-8", "replace"))
    if low.endswith(".pptx"):
        z = zipfile.ZipFile(path); out = []
        for n in sorted((n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml", n)), key=lambda s: int(re.findall(r"\d+", s)[0])):
            out.append(xml_paras(z.read(n), b"p"))
        return "\n\n".join(out)
    if low.endswith(".epub"):
        z = zipfile.ZipFile(path)
        return "\n\n".join(strip_html(z.read(n).decode("utf-8", "replace")) for n in z.namelist() if n.lower().endswith((".xhtml", ".html", ".htm")))
    raw = open(path, "rb").read().decode("utf-8", "replace")
    if low.endswith((".html", ".htm", ".xml")):
        return strip_html(raw)
    return raw

if __name__ == "__main__":
    t = main(sys.argv[1])
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    sys.stdout.write(t.strip() + "\n")
