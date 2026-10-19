"""Relative links in the shipped documentation resolve.

The READMEs (en, zh-CN), ``docs/**/*.md`` and the built-in extensions'
READMEs link files and sections of each other; a renamed heading or a moved
file breaks them silently. This checks every relative link and image (Markdown
``[text](target)`` and HTML ``src=`` / ``href=``) outside code: the file must
exist, and a ``#fragment`` into a Markdown file must name one of its headings
(GitHub's anchor: lower case, punctuation and symbols dropped, spaces to
``-``, ``-1``, ``-2`` for repeats) or an ``<a id=…>`` / ``<a name=…>``.
Text inside fenced code blocks and inline code is an example, not a link.
"""

from __future__ import annotations

import re
import unicodedata
from functools import cache
from pathlib import Path
from urllib.parse import unquote

REPO = Path(__file__).resolve().parents[1]

FENCE = re.compile(r"^\s*(?:>\s*)*(```|~~~)")
HEADING = re.compile(r"^(?:>\s*)*(#{1,6})\s+(.*?)\s*#*\s*$")
EXPLICIT_ANCHOR = re.compile(r"""<a\s+(?:id|name)=["']([^"']+)["']""")
MARKDOWN_LINK = re.compile(r"!?\[(?:[^\[\]]|\[[^\]]*\])*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
HTML_LINK = re.compile(r"""\b(?:src|href)=["']([^"']+)["']""")
SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*:", re.IGNORECASE)


def documents() -> list[Path]:
    found = [REPO / "README.md", REPO / "README.zh-CN.md"]
    found += sorted((REPO / "docs").rglob("*.md"))
    found += sorted((REPO / "extensions-builtin").glob("*/README*.md"))
    found += sorted((REPO / "extensions-builtin").glob("*/docs/**/*.md"))
    return [path for path in found if path.is_file()]


def prose_lines(text: str, *, inline_code: bool = False) -> list[str]:
    """``text``'s lines with fenced code blocks blanked and inline code
    removed (kept with ``inline_code``: headings take their anchor from it)."""

    lines, fenced = [], False
    for line in text.splitlines():
        if FENCE.match(line):
            fenced = not fenced
            lines.append("")
        else:
            lines.append("" if fenced else line if inline_code else re.sub(r"`[^`]*`", "", line))
    return lines


def github_slug(heading: str) -> str:
    text = re.sub(r"<[^>]+>", "", heading)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text).replace("`", "").strip().lower()
    kept = (char for char in text if char in " -_" or unicodedata.category(char)[0] in "LN" or unicodedata.category(char) == "Mn")
    return "".join("-" if char == " " else char for char in kept)


@cache
def anchors(path: Path) -> frozenset[str]:
    seen: dict[str, int] = {}
    found: set[str] = set()
    for line in prose_lines(path.read_text(encoding="utf-8"), inline_code=True):
        if match := HEADING.match(line):
            slug = github_slug(match.group(2))
            repeat = seen.get(slug, 0)
            seen[slug] = repeat + 1
            found.add(slug if repeat == 0 else f"{slug}-{repeat}")
        found.update(match.group(1) for match in EXPLICIT_ANCHOR.finditer(line))
    return frozenset(found)


def broken_links(path: Path, root: Path = REPO) -> list[str]:
    problems = []
    for number, line in enumerate(prose_lines(path.read_text(encoding="utf-8")), 1):
        targets = [match.group(1) for match in MARKDOWN_LINK.finditer(line)] + [match.group(1) for match in HTML_LINK.finditer(line)]
        for target in targets:
            if SCHEME.match(target) or target.startswith("//"):
                continue
            file_part, _, fragment = target.partition("#")
            linked = (path.parent / unquote(file_part)).resolve() if file_part else path
            where = f"{path.relative_to(root)}:{number}"
            if not linked.exists():
                problems.append(f"{where}: no file {target}")
            elif fragment and linked.suffix == ".md" and linked.is_file() and unquote(fragment).lower() not in anchors(linked):
                problems.append(f"{where}: no heading for {target}")
    return problems


def test_slugs_follow_github():
    assert github_slug("3. Procedure") == "3-procedure"
    assert github_slug("Ports from `ui()`") == "ports-from-ui"
    assert github_slug("14. Porting an A1111 / Forge extension") == "14-porting-an-a1111--forge-extension"
    assert github_slug("🙏 Native speakers — please help review!") == "-native-speakers--please-help-review"
    assert github_slug("快速开始") == "快速开始"


def test_checker_finds_broken_links(tmp_path):
    doc = tmp_path / "a.md"
    (tmp_path / "b.md").write_text("# Title\n\n## Same\n\n## Same\n", encoding="utf-8")
    doc.write_text(
        '[ok](b.md#title) [ok](b.md#same-1) [file](missing.md) [heading](b.md#nope) ![img](x.png)\n<img src="y.png"> [web](https://example.com/x.md#nope)\n```md\n[example](not-a-link.md)\n```\n`[code](also-not.md)`\n',
        encoding="utf-8",
    )
    assert [problem.split(": ", 1)[1] for problem in broken_links(doc, tmp_path)] == [
        "no file missing.md",
        "no heading for b.md#nope",
        "no file x.png",
        "no file y.png",
    ]


def test_documentation_links_resolve():
    docs = documents()
    assert len(docs) > 40  # the READMEs, docs/ and the extensions' READMEs were found
    problems = [problem for path in docs for problem in broken_links(path)]
    assert problems == []
