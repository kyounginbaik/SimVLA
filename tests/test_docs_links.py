"""Keep repository-local Markdown links and heading anchors valid."""
from __future__ import annotations

import re
import urllib.parse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
HTML_LINK = re.compile(r"(?:href|src)=[\"']([^\"']+)[\"']", re.IGNORECASE)
FENCE = re.compile(r"^\s*(```|~~~)")
HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")


def _without_fenced_code(text: str) -> str:
    output: list[str] = []
    marker: str | None = None
    for line in text.splitlines():
        match = FENCE.match(line)
        if match:
            token = match.group(1)
            marker = None if marker and token.startswith(marker[0]) else token
            continue
        if marker is None:
            output.append(line)
    return "\n".join(output)


def _slug(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text).strip().lower()
    text = re.sub(r"[^\w\- ]", "", text, flags=re.UNICODE)
    return re.sub(r"\s+", "-", text)


def _anchors(path: Path) -> set[str]:
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    for line in _without_fenced_code(path.read_text(encoding="utf-8")).splitlines():
        match = HEADING.match(line)
        if not match:
            continue
        base = _slug(match.group(1))
        count = counts.get(base, 0)
        counts[base] = count + 1
        anchors.add(base if count == 0 else f"{base}-{count}")
    return anchors


def test_repository_markdown_links_resolve():
    documents = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")),
                 *sorted((ROOT / "examples").rglob("*.md"))]
    failures: list[str] = []
    for document in documents:
        text = _without_fenced_code(document.read_text(encoding="utf-8"))
        targets = MARKDOWN_LINK.findall(text) + HTML_LINK.findall(text)
        for raw in targets:
            target = raw.strip().split(maxsplit=1)[0].strip("<>")
            parsed = urllib.parse.urlsplit(target)
            if parsed.scheme or target.startswith("//") or parsed.path.startswith("/"):
                continue
            path = document if not parsed.path else (document.parent / urllib.parse.unquote(parsed.path)).resolve()
            if not path.exists():
                failures.append(f"{document.relative_to(ROOT)} -> {target}: missing file")
                continue
            if parsed.fragment and path.suffix.lower() == ".md":
                anchor = urllib.parse.unquote(parsed.fragment).lower()
                if anchor not in _anchors(path):
                    failures.append(f"{document.relative_to(ROOT)} -> {target}: missing anchor")
    assert not failures, "\n" + "\n".join(failures)
