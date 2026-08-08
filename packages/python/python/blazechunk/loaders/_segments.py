"""Structural segmentation of GitHub-Flavored Markdown.

This module turns a Markdown document into a flat, ordered list of
:class:`Segment` values. A segment is a *structural span* — a heading, a
paragraph, a table, a fenced code block — recorded as offsets into the source
string rather than as a copy of it.

Nothing here imports :mod:`anydoc`. The segmenter's input contract is
"deterministic GFM", which is exactly what anydoc's single Markdown serializer
emits for every format it supports, and also what a hand-written ``.md`` file
looks like. That is what lets one code path serve all formats.

Why offsets and not text
------------------------
Every segment is a half-open span ``[start, end)``, so the text of a segment is
always ``markdown[seg.char_start:seg.char_end]`` — a *slice*, never a
re-rendering. Structural metadata can therefore never drift away from the text
it describes, and the reconstruction invariants in :mod:`blazechunk.loaders`
hold by construction instead of by careful bookkeeping.

Each segment carries both character offsets (to slice the ``str`` that gets
handed to a chunker) and byte offsets (blazechunk's public offset convention,
matching ``Chunk.start_index``). Both are computed in a single pass so neither
costs a rescan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

__all__ = [
    "Segment",
    "SegmentKind",
    "segment_markdown",
    "merge_small_segments",
    "group_span",
]

#: The structural kinds the segmenter distinguishes.
#:
#: * ``"heading"`` — an ATX heading line (``#`` … ``######``).
#: * ``"prose"``   — an ordinary paragraph.
#: * ``"table"``   — a GFM pipe table, header and delimiter row included.
#: * ``"code"``    — a fenced code block, fences included.
#: * ``"list"``    — a bullet or ordered list, including nested items.
#: * ``"quote"``   — a block quote.
#: * ``"rule"``    — a thematic break.
SegmentKind = str

_HEADING = re.compile(r"^ {0,3}(#{1,6})(?:\s+(.*?))?\s*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})\s*(\S*)")
_RULE = re.compile(r"^ {0,3}(?:-{3,}|\*{3,}|_{3,})\s*$")
_TABLE_DELIM = re.compile(r"^ {0,3}\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)*\|?\s*$")
_BULLET = re.compile(r"^ {0,3}[-*+]\s+")
_ORDERED = re.compile(r"^ {0,3}\d{1,9}[.)]\s+")


@dataclass(frozen=True)
class Segment:
    """One structural span of a Markdown document.

    Attributes:
        kind: What the span is — see :data:`SegmentKind`.
        char_start: Start offset in code points, for slicing the ``str``.
        char_end: End offset in code points, exclusive.
        byte_start: Start offset in UTF-8 bytes, for reporting to callers.
        byte_end: End offset in UTF-8 bytes, exclusive.
        heading_path: The enclosing heading titles, outermost first. A heading
            segment includes *itself* as the last element, so it shares a path
            with the prose it introduces and the two can merge.
        lang: The info string of a fenced code block, or ``None``.
    """

    kind: SegmentKind
    char_start: int
    char_end: int
    byte_start: int
    byte_end: int
    heading_path: tuple[str, ...] = ()
    lang: str | None = None

    def text(self, markdown: str) -> str:
        """The span's text: ``markdown[char_start:char_end]``."""
        return markdown[self.char_start : self.char_end]

    @property
    def is_empty(self) -> bool:
        """Whether the span covers no characters."""
        return self.char_end <= self.char_start


def _is_blank(line: str) -> bool:
    return not line.strip()


def _is_list_start(line: str) -> bool:
    return bool(_BULLET.match(line) or _ORDERED.match(line))


def _heading_title(line: str) -> tuple[int, str]:
    """Return ``(level, title)`` for an ATX heading line."""
    match = _HEADING.match(line)
    if match is None:  # pragma: no cover - callers check first
        return 0, ""
    hashes, title = match.group(1), match.group(2) or ""
    # A trailing closing sequence ("## Title ##") is decoration, not content.
    return len(hashes), title.rstrip("#").strip()


def segment_markdown(markdown: str) -> list[Segment]:
    """Split ``markdown`` into ordered structural segments.

    Segments never overlap and appear in document order. The text *between*
    consecutive segments is always whitespace (the blank lines that separate
    blocks), which is what lets a caller verify full coverage of the document.

    Args:
        markdown: A GFM document. anydoc's serializer output, or any Markdown.

    Returns:
        A list of :class:`Segment`, possibly empty for blank input.

    Example:
        >>> segs = segment_markdown("# Title\\n\\nSome prose.\\n")
        >>> [(s.kind, s.heading_path) for s in segs]
        [('heading', ('Title',)), ('prose', ('Title',))]
    """
    lines = markdown.splitlines(keepends=True)
    if not lines:
        return []

    # Line-start offsets in both coordinate systems, plus an end sentinel, so a
    # segment boundary is a cheap lookup rather than a re-scan. In an all-ASCII
    # document the two coordinate systems coincide, which is worth checking for:
    # one C-level scan of the whole string replaces a UTF-8 encode per line, and
    # most documents take that path.
    char_at: list[int] = []
    char_pos = 0
    for line in lines:
        char_at.append(char_pos)
        char_pos += len(line)
    char_at.append(char_pos)

    if markdown.isascii():
        byte_at = char_at
    else:
        byte_at = []
        byte_pos = 0
        for line in lines:
            byte_at.append(byte_pos)
            byte_pos += len(line.encode("utf-8"))
        byte_at.append(byte_pos)

    def span(first: int, last: int) -> tuple[int, int, int, int]:
        """Offsets covering lines ``[first, last]``, minus the final newline.

        A line terminator is always ASCII, so the same count of characters and
        of bytes comes off either end.
        """
        tail = lines[last]
        trimmed = len(tail) - len(tail.rstrip("\r\n"))
        return (
            char_at[first],
            char_at[last + 1] - trimmed,
            byte_at[first],
            byte_at[last + 1] - trimmed,
        )

    segments: list[Segment] = []
    stack: list[str] = []
    index = 0
    total = len(lines)

    while index < total:
        line = lines[index]

        if _is_blank(line):
            index += 1
            continue

        # Fenced code first: a fence can legally contain '#', '|' and '>' lines,
        # so every other rule below would misfire inside one.
        fence = _FENCE.match(line)
        if fence is not None:
            marker, info = fence.group(1), fence.group(2)
            start = index
            index += 1
            while index < total:
                closing = _FENCE.match(lines[index])
                if (
                    closing is not None
                    and closing.group(1)[0] == marker[0]
                    and len(closing.group(1)) >= len(marker)
                    and not closing.group(2)
                ):
                    index += 1
                    break
                index += 1
            c0, c1, b0, b1 = span(start, index - 1)
            segments.append(
                Segment("code", c0, c1, b0, b1, tuple(stack), info or None)
            )
            continue

        heading = _HEADING.match(line)
        if heading is not None:
            level, title = _heading_title(line)
            del stack[level - 1 :]
            stack.append(title)
            c0, c1, b0, b1 = span(index, index)
            segments.append(Segment("heading", c0, c1, b0, b1, tuple(stack)))
            index += 1
            continue

        if _RULE.match(line):
            c0, c1, b0, b1 = span(index, index)
            segments.append(Segment("rule", c0, c1, b0, b1, tuple(stack)))
            index += 1
            continue

        # A pipe table is a '|' row followed by a delimiter row. Without the
        # delimiter row it is just a paragraph that happens to contain pipes.
        if line.lstrip().startswith("|") and index + 1 < total and _TABLE_DELIM.match(lines[index + 1]):
            start = index
            index += 2
            while index < total and lines[index].lstrip().startswith("|"):
                index += 1
            c0, c1, b0, b1 = span(start, index - 1)
            segments.append(Segment("table", c0, c1, b0, b1, tuple(stack)))
            continue

        if line.lstrip().startswith(">"):
            start = index
            while index < total and not _is_blank(lines[index]):
                index += 1
            c0, c1, b0, b1 = span(start, index - 1)
            segments.append(Segment("quote", c0, c1, b0, b1, tuple(stack)))
            continue

        if _is_list_start(line):
            start = index
            index = _consume_list(lines, index, total)
            c0, c1, b0, b1 = span(start, index - 1)
            segments.append(Segment("list", c0, c1, b0, b1, tuple(stack)))
            continue

        # Paragraph: runs to the next blank line, or to the next line that
        # starts a different structure.
        start = index
        while index < total and not _is_blank(lines[index]):
            nxt = lines[index]
            if index > start and (
                _HEADING.match(nxt)
                or _FENCE.match(nxt)
                or _is_list_start(nxt)
                or nxt.lstrip().startswith(">")
            ):
                break
            index += 1
        c0, c1, b0, b1 = span(start, index - 1)
        segments.append(Segment("prose", c0, c1, b0, b1, tuple(stack)))

    return segments


def _consume_list(lines: list[str], index: int, total: int) -> int:
    """Return the line index just past a whole list.

    anydoc separates list items with blank lines, so a naive "stop at the first
    blank line" scan would shatter one list into several segments. A blank line
    only ends the list when the next non-blank line is neither another item nor
    an indented continuation of one.
    """
    index += 1
    while index < total:
        if not _is_blank(lines[index]):
            line = lines[index]
            if _is_list_start(line) or line.startswith((" ", "\t")):
                index += 1
                continue
            break

        look = index
        while look < total and _is_blank(lines[look]):
            look += 1
        if look >= total:
            break
        nxt = lines[look]
        if _is_list_start(nxt) or nxt.startswith((" ", "\t")):
            index = look + 1
            continue
        break
    return index


def descends_into(outer: tuple[str, ...], inner: tuple[str, ...]) -> bool:
    """Whether ``inner`` is ``outer`` or a subsection of it.

    This is what "a chunk never crosses a heading boundary" should mean.
    Requiring the two paths to be *equal* would strand every heading that is
    immediately followed by a deeper one: a document opening
    ``# Report`` / ``## Methods`` / prose would emit ``# Report`` and
    ``## Methods`` as two useless single-line chunks, because neither shares a
    path with what follows it.

    Moving *down* into a subsection keeps a heading with the content it
    introduces. Moving *sideways* to a sibling section — ``Methods`` to
    ``Results`` — is the boundary that actually matters, and is still refused.

    Because merging only ever proceeds forward and downward, a merged group's
    paths form a chain, so the last member's path contains every other's.
    """
    return inner[: len(outer)] == outer


def merge_small_segments(
    segments: list[Segment],
    markdown: str,
    *,
    min_size: int,
    respect_headings: bool = True,
) -> list[list[Segment]]:
    """Group undersized segments forward into their following sibling.

    A short segment — most often a heading, or a one-line paragraph — is poor
    retrieval material on its own. Merging it *forward* attaches a heading to
    the prose it introduces, which is almost always the intent; merging backward
    would attach it to the section it just ended.

    Tables and code blocks never join a group. A table with a stray sentence
    glued to it retrieves worse than a small table, and a code block's value
    depends on its fences staying intact.

    Args:
        segments: Ordered segments from :func:`segment_markdown`.
        markdown: The document the segments index into, for measuring sizes.
        min_size: Segments shorter than this many characters seek a partner.
        respect_headings: When true, a group never moves *sideways* across a
            heading boundary. Descending into a subsection is still allowed —
            see :func:`descends_into`.

    Returns:
        A list of groups, each a non-empty list of adjacent segments. Every
        input segment appears in exactly one group, in its original order.
    """
    groups: list[list[Segment]] = []
    pending: list[Segment] = []

    def joinable(seg: Segment) -> bool:
        return seg.kind not in ("table", "code")

    for seg in segments:
        if pending:
            same_section = descends_into(pending[-1].heading_path, seg.heading_path)
            if joinable(seg) and (same_section or not respect_headings):
                pending.append(seg)
                if len(markdown[pending[0].char_start : seg.char_end]) >= min_size:
                    groups.append(pending)
                    pending = []
                continue
            groups.append(pending)
            pending = []

        if not joinable(seg):
            groups.append([seg])
            continue

        if len(seg.text(markdown)) < min_size:
            pending = [seg]
            continue

        groups.append([seg])

    if pending:
        groups.append(pending)
    return groups


def group_span(group: list[Segment], markdown: str) -> Segment:
    """Collapse a merged group into the single segment that covers it.

    The result spans from the group's first character to its last, *including*
    the blank lines between members, so its text stays an exact contiguous
    slice of ``markdown``.

    The group takes the *deepest* heading path it contains, not the first. A
    group's paths form a chain (see :func:`descends_into`), so the last member's
    path has every other as a prefix and is the most specific description of
    where the content sits.
    """
    head, tail = group[0], group[-1]
    kind = head.kind
    if len(group) > 1:
        # A heading merged with what follows is named for the content, not for
        # the heading line, which is only a label for it.
        kinds = {seg.kind for seg in group} - {"heading", "rule"}
        kind = "prose" if not kinds else (kinds.pop() if len(kinds) == 1 else "prose")
    return replace(
        head,
        kind=kind,
        char_end=tail.char_end,
        byte_end=tail.byte_end,
        heading_path=tail.heading_path,
    )
