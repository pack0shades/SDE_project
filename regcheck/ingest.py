"""Document Pre-processor (paper 3.1, 4.1): file -> sections (parents) -> chunks (children) -> index."""

import re
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pypdf import PdfReader

from . import llm, prompts
from .config import DATA, settings
from .schemas import Chunk, Section
from .store import VectorStore

Kind = Literal["regulation", "srs"]
Mode = Literal["agentic", "fixed"]

WINDOW = 8000  # chars per chunker call; long articles are read in windows but stay one parent

# EUR-Lex page furniture: GDPR-style and Data-Act-style page headers, and "(1) OJ C 229 ..." footnotes.
_HEADER = re.compile(r"(?m)^(ELI: http\S+ \d+/\d+|OJ L, [\d.]+ EN|.*Official Journal of the European Union EN)\s*$\n?")
# ponytail: strips only the first line of a multi-line footnote; the chunker ignores the remainder
_FOOTNOTE = re.compile(r"(?m)^\(\d+\)\s+(OJ|Directive|Regulation|Council|Commission|Position)\b.*$\n?")
_ARTICLE = re.compile(r"(?m)^Article (\d+)\s*$")
_ARTICLE_END = re.compile(r"(?m)^(CHAPTER [IVXLC]+|Section \d+|Done at )")
# "3 Specific Requirements" / "3.1 Functionality"; tabs mean TOC or table rows.
_SRS_HEADING = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2})?)\.?[ ]+([A-Z][^\t]{2,70}?)\s*$")
_NOT_HEADING = re.compile(r"\b(shall|must|should|will)\b|\s\d+$", re.IGNORECASE)


def load_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(errors="replace")


def _windows(text: str, size: int, overlap: int = 0) -> list[str]:
    """Split at line boundaries into pieces of at most `size` chars, repeating `overlap` chars of lines."""
    lines = [line[i:i + size] for line in text.splitlines() for i in range(0, max(len(line), 1), size)]
    out: list[str] = []
    cur: list[str] = []
    for line in lines:
        if cur and sum(map(len, cur)) + len(line) > size:
            out.append("\n".join(cur))
            keep: list[str] = []
            while cur and sum(map(len, keep)) + len(cur[-1]) <= overlap:
                keep.insert(0, cur.pop())
            cur = keep
        cur.append(line)
    if cur:
        out.append("\n".join(cur))
    return [w for w in out if w.strip()]


def _regulation_sections(text: str, doc: str) -> list[Section]:
    text = _FOOTNOTE.sub("", _HEADER.sub("", text))
    heads, want = [], 1
    for m in _ARTICLE.finditer(text):  # accept only the next number: skips recitals and stray lines
        if int(m.group(1)) == want:
            heads.append(m)
            want += 1
    sections = []
    for m, nxt in zip(heads, [*heads[1:], None]):
        body = text[m.end(): nxt.start() if nxt else len(text)]
        if end := _ARTICLE_END.search(body):
            body = body[: end.start()]
        title, _, rest = body.strip().partition("\n")
        n = m.group(1)
        sections.append(Section(id=f"{doc}:art{n}", doc=doc, ref=f"Article {n} - {title.strip()}", text=rest.strip()))
    return sections


def _srs_sections(text: str, doc: str) -> list[Section]:
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if _SRS_HEADING.match(line) and not _NOT_HEADING.search(line.strip())]
    if len(starts) < 3:
        return [Section(id=f"{doc}:w{i}", doc=doc, ref=f"Part {i + 1}", text=w)
                for i, w in enumerate(_windows(text, WINDOW))]
    # Text before the first heading (cover, revision history, TOC) is skipped: the chunker turned
    # TOC entries into invented requirements, and no PURE front matter held a real one.
    sections = []
    for n, (a, b) in enumerate(zip(starts, [*starts[1:], len(lines)]), 1):
        body = "\n".join(lines[a + 1: b]).strip()
        if len(body) >= 40:
            sections.append(Section(id=f"{doc}:s{n}", doc=doc, ref=" ".join(lines[a].split()), text=body))
    return sections


def split_sections(text: str, doc: str, kind: Kind) -> list[Section]:
    """Deterministic split into parent documents: articles for a regulation, numbered sections for an SRS."""
    return _regulation_sections(text, doc) if kind == "regulation" else _srs_sections(text, doc)


def chunk(section: Section, kind: Kind, mode: Mode, trace: list[dict]) -> list[Chunk]:
    """Children of a section. `agentic` is the paper's LLM chunking; `fixed` is the baseline RAG split."""
    max_chars = settings().retrieval.max_chunk_tokens * 4  # ponytail: chars/4 token estimate
    if mode == "fixed":
        texts = _windows(section.text, max_chars, overlap=max_chars // 8)
    else:
        template = prompts.CHUNK_POLICY if kind == "regulation" else prompts.CHUNK_SRS
        instruction = template.format(max_chars=max_chars, data_note=prompts.DATA_NOTE)

        def fits(statements: list[str]) -> None:
            if too_long := [s[:60] for s in statements if len(s) > max_chars * 1.25]:
                raise ValueError(f"statements longer than {max_chars} characters: {too_long}")

        texts = []
        for w in _windows(section.text, WINDOW):
            body = f'<section id="{section.id}" title="{section.ref}">\n{w}\n</section>'
            texts += llm.complete_json("chunker", prompts.alpaca(prompts.CHUNK_PERSONA, instruction, body),
                                       list[str], trace, validate=fits)
    texts = [t.replace("**", "").strip() for t in texts if t.strip()]  # models add bold despite the prompt
    return [Chunk(id=f"{section.id}:{i}", doc=section.doc, parent_id=section.id, text=t) for i, t in enumerate(texts)]


def store_path(doc: str, mode: Mode = "agentic") -> Path:
    return DATA / "index" / (doc if mode == "agentic" else f"{doc}-{mode}")


def index(path: Path, doc: str, kind: Kind, mode: Mode = "agentic",
          progress: Callable[[str], None] = print) -> VectorStore:
    """Index one document. Saves after every section, so an interrupted run resumes where it stopped."""
    store = VectorStore(store_path(doc, mode))
    sections = split_sections(load_text(path), doc, kind)
    trace: list[dict] = []
    for i, s in enumerate(sections, 1):
        if s.id not in store.sections:
            store.add(chunk(s, kind, mode, trace), [s])
        progress(f"{doc} [{mode}] {i}/{len(sections)} {s.ref[:60]}: {len(store.chunks)} chunks")
    return store
