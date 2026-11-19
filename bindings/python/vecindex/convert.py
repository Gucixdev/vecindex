"""Other formats into Markdown, for DB.ingest_text: the C library reads only Markdown and JSON, and every other format
is turned into one of those two here, in Python, where the parsers already exist.

    to_markdown("notes.txt")   paragraphs as they are
    to_markdown("page.html")   headings h1-h6 as #, p/li/pre kept, script/style dropped (html.parser, stdlib)
    to_markdown("paper.pdf")   pdftotext -layout; each page becomes "# Page N", so a page is a section (a page id)

A new format is one function from a path to Markdown text:

    @vecindex.convert.register(".docx")
    def docx(path):
        return subprocess.run(["pandoc", "-t", "gfm", path], capture_output=True, text=True, check=True).stdout

Records go to JSONL instead (one object per line, ingest_text(name, text, "jsonl")): a CSV is
"\\n".join(json.dumps(r) for r in csv.DictReader(f)). Any language can do the same through the C API."""
import html.parser
import os
import shutil
import subprocess

_CONV = {}


def register(*exts):
    """Decorator: fn(path) -> Markdown is used for files ending in exts (".rst", ".docx", ...)."""
    def deco(fn):
        for e in exts:
            _CONV[e.lower()] = fn
        return fn
    return deco


def to_markdown(path):
    """The file at path as Markdown, by its extension. ValueError when no converter takes it."""
    ext = os.path.splitext(path)[1].lower()
    fn = _CONV.get(ext)
    if fn is None:
        raise ValueError("vecindex.convert: no converter for %r (register one: vecindex.convert.register)" % ext)
    return fn(path)


@register(".txt", ".text", "")
def text(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


class _Html(html.parser.HTMLParser):
    _DROP = {"script", "style", "head", "nav", "footer", "noscript", "svg"}
    _BLOCK = {"p", "div", "section", "article", "br", "tr", "table", "blockquote", "ul", "ol"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out = []
        self.drop = 0
        self.pre = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._DROP:
            self.drop += 1
        elif self.drop:
            return
        elif len(tag) == 2 and tag[0] == "h" and tag[1] in "123456":
            self.out.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "li":
            self.out.append("\n- ")
        elif tag == "pre":
            self.pre += 1
            self.out.append("\n\n```\n")
        elif tag in self._BLOCK:
            self.out.append("\n\n")

    def handle_endtag(self, tag):
        if tag in self._DROP:
            self.drop = max(0, self.drop - 1)
        elif self.drop:
            return
        elif len(tag) == 2 and tag[0] == "h" and tag[1] in "123456":
            self.out.append("\n\n")
        elif tag == "pre":
            self.pre = max(0, self.pre - 1)
            self.out.append("\n```\n\n")

    def handle_data(self, data):
        if self.drop:
            return
        if self.pre:
            self.out.append(data)
        else:
            self.out.append(" ".join(data.split()) + (" " if data[-1:].isspace() else ""))

    def markdown(self):
        lines = "".join(self.out).split("\n")
        md, blank = [], 0
        for ln in lines:
            ln = ln.rstrip()
            blank = blank + 1 if not ln else 0
            if blank < 2:
                md.append(ln)
        return "\n".join(md).strip() + "\n"


def html_to_markdown(src):
    """HTML text as Markdown: the headings stay headings, so they become sections in the store."""
    p = _Html()
    p.feed(src)
    p.close()
    return p.markdown()


@register(".html", ".htm", ".xhtml")
def html_file(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return html_to_markdown(f.read())


@register(".pdf")
def pdf(path):
    """pdftotext (poppler-utils) splits pages with a form feed; each page becomes a section, so a hit says its page."""
    exe = shutil.which("pdftotext")
    if exe is None:
        raise ValueError("vecindex.convert: .pdf needs pdftotext (poppler-utils) on PATH")
    out = subprocess.run([exe, "-layout", "-enc", "UTF-8", path, "-"], capture_output=True, check=True).stdout
    pages = out.decode("utf-8", "replace").split("\f")
    md = []
    for i, pg in enumerate(pages):
        body = pg.strip()
        if body:
            md.append("# Page %d\n\n%s\n" % (i + 1, ("\n" + body).replace("\n#", "\n\\#")[1:]))
    return "\n".join(md)
