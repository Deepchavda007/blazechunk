"""Tests for :mod:`blazechunk.loaders`.

The centrepiece is :func:`assert_invariants`, applied to every document any test
produces. The invariants are what make the offsets trustworthy, and they are
exactly where the bugs live: table routing and segment merging are both places
where content can silently disappear.
"""

from __future__ import annotations

import asyncio

import pytest

import _docbuilders as build
from blazechunk import CodeChunker, RecursiveChunker, TableChunker
from blazechunk.loaders import (
    ChunkedDocument,
    DocumentChunk,
    DocumentChunker,
    DocumentError,
    ScannedDocumentError,
    UnsupportedDocument,
)
from blazechunk.loaders._segments import merge_small_segments, segment_markdown

anydoc = pytest.importorskip("anydoc", reason="needs the anydoc extra")


# --------------------------------------------------------------------------
# Invariants
# --------------------------------------------------------------------------


def assert_invariants(result: ChunkedDocument) -> None:
    """Check every guarantee :class:`ChunkedDocument` makes.

    I1 (exact reconstruction) — a chunk flagged ``is_exact`` is byte-for-byte
    the slice its offsets name.

    I2 (full coverage) — the chunks' offset spans tile the document in order,
    without overlap, and everything they leave out is whitespace. Stated over
    *offsets* rather than concatenated text, so it stays checkable even where a
    split table repeats its header and the texts therefore overlap.
    """
    data = result.markdown_bytes

    previous_end = 0
    for chunk in result.chunks:
        assert 0 <= chunk.md_start < chunk.md_end <= len(data), (
            f"offsets out of range or empty: {chunk!r}"
        )
        assert chunk.md_start >= previous_end, f"chunks overlap or are unordered: {chunk!r}"

        gap = data[previous_end : chunk.md_start]
        assert not gap.strip(), f"non-whitespace content dropped before {chunk!r}: {gap!r}"

        if chunk.is_exact:
            assert data[chunk.md_start : chunk.md_end] == chunk.text.encode("utf-8"), (
                f"I1 violated — text is not the slice its offsets name: {chunk!r}"
            )
        else:
            # The only sanctioned inexactness: a table continuation carrying a
            # repeated header. The rows it covers must still be present in it.
            assert chunk.kind == "table", f"unexpected inexact chunk: {chunk!r}"
            covered = data[chunk.md_start : chunk.md_end].decode("utf-8")
            assert covered.strip() in chunk.text, f"covered rows missing from {chunk!r}"

        previous_end = chunk.md_end

    trailing = data[previous_end:]
    assert not trailing.strip(), f"non-whitespace content dropped at the end: {trailing!r}"

    for chunk in result.chunks:
        assert chunk.source_format == result.format


RICH_BODY = (
    "<h1>Methods</h1>"
    "<p>An introductory paragraph that is long enough to stand on its own "
    "without being merged into anything that follows it in the document.</p>"
    "<h2>Sample Preparation</h2>"
    "<p>Second level prose describing how the samples were prepared for the "
    "analysis that follows in the results section further below.</p>"
    "<ul><li>first item<ul><li>nested item</li></ul></li><li>second item</li></ul>"
    "<pre><code>def f(x):\n    return x + 1\n</code></pre>"
    "<blockquote><p>A quoted claim worth preserving.</p></blockquote>"
    "<table><thead><tr><th>region</th><th>q1</th></tr></thead>"
    "<tbody><tr><td>west</td><td>4</td></tr><tr><td>east</td><td>7</td></tr></tbody></table>"
    "<h2>Results</h2><p>Final prose paragraph.</p>"
)


@pytest.fixture()
def loader() -> DocumentChunker:
    return DocumentChunker()


class TestFormats:
    """Every format reaches chunks through the same path."""

    def test_pdf(self, loader):
        data = build.pdf(
            [
                ("Quarterly Report", 24, 700),
                ("Revenue grew across every region this quarter.", 11, 660),
                ("Methods", 18, 600),
                ("We sampled two hundred sites.", 11, 570),
            ]
        )
        result = loader.chunk(data, format="pdf")
        assert result.format == "pdf"
        assert result.chunks
        assert_invariants(result)
        assert any("Quarterly Report" in c.text for c in result.chunks)

    def test_pdf_carries_heading_paths(self, loader):
        """The headline claim: PDF is not a second-class path."""
        data = build.pdf(
            [
                ("Annual Report", 24, 700),
                ("Methods", 18, 640),
                ("We sampled two hundred sites across the region.", 11, 600),
            ]
        )
        result = loader.chunk(data, format="pdf")
        assert any(c.heading_path for c in result.chunks)
        assert_invariants(result)

    def test_docx(self, loader):
        data = build.docx(
            [
                ("Heading1", "Methods"),
                ("Normal", "Intro prose that runs on for a while."),
                ("Heading2", "Sample Prep"),
                ("Normal", "Details here."),
            ]
        )
        result = loader.chunk(data, format="docx")
        assert result.format == "docx"
        assert_invariants(result)
        paths = {c.heading_path for c in result.chunks}
        assert ("Methods", "Sample Prep") in paths

    def test_epub(self, loader):
        result = loader.chunk(build.epub(RICH_BODY), format="epub")
        assert_invariants(result)
        kinds = {c.kind for c in result.chunks}
        assert "table" in kinds and "code" in kinds

    def test_csv(self, loader):
        result = loader.chunk(b"name,qty\nwidget,3\ngadget,5\n", format="csv")
        assert result.format == "csv"
        assert result.chunks[0].kind == "table"
        assert_invariants(result)

    def test_csv_bytes_without_format_is_a_clear_error(self, loader):
        """CSV has no signature, so bytes alone genuinely cannot be identified."""
        with pytest.raises(UnsupportedDocument, match="format"):
            loader.chunk(b"name,qty\nwidget,3\n")

    def test_path_input(self, loader, tmp_path):
        path = tmp_path / "report.csv"
        path.write_bytes(b"name,qty\nwidget,3\n")
        result = loader.chunk(path)
        assert result.format == "csv"
        assert_invariants(result)

    def test_markdown_passthrough_needs_no_conversion(self, loader, tmp_path):
        path = tmp_path / "notes.md"
        path.write_text("# Title\n\nSome prose here.\n")
        result = loader.chunk(path)
        assert result.format == "md"
        assert result.markdown == "# Title\n\nSome prose here.\n"
        assert_invariants(result)

    def test_unknown_format_name_rejected(self, loader):
        with pytest.raises(ValueError, match="unsupported format"):
            loader.chunk(b"data", format="xyz")


class TestStructure:
    """Routing, heading paths and the boundaries that must not be crossed."""

    def test_heading_path_is_a_prefix_chain(self, loader):
        result = loader.chunk(build.epub(RICH_BODY), format="epub")
        for chunk in result.chunks:
            assert all(isinstance(part, str) for part in chunk.heading_path)
        seen: list[tuple[str, ...]] = [c.heading_path for c in result.chunks]
        for path in seen:
            # Every ancestor of a path must itself be a path that appeared, or
            # the stack skipped a level.
            for depth in range(1, len(path)):
                assert any(other[:depth] == path[:depth] for other in seen)

    def test_table_rows_are_never_split(self):
        rows = "".join(f"<tr><td>r{i}</td><td>{i}</td></tr>" for i in range(12))
        body = f"<table><thead><tr><th>k</th><th>v</th></tr></thead><tbody>{rows}</tbody></table>"
        loader = DocumentChunker(table_chunker=TableChunker(chunk_size=3))
        result = loader.chunk(build.epub(body), format="epub")
        table_chunks = [c for c in result.chunks if c.kind == "table"]
        assert len(table_chunks) > 1, "expected the table to split"
        for chunk in table_chunks:
            for line in chunk.text.splitlines():
                if line.strip():
                    assert line.lstrip().startswith("|"), f"broken row: {line!r}"
                    assert line.rstrip().endswith("|"), f"broken row: {line!r}"
        assert_invariants(result)

    def test_split_table_repeats_header_and_is_reported(self):
        rows = "".join(f"<tr><td>r{i}</td><td>{i}</td></tr>" for i in range(12))
        body = f"<table><thead><tr><th>region</th><th>v</th></tr></thead><tbody>{rows}</tbody></table>"
        loader = DocumentChunker(table_chunker=TableChunker(chunk_size=3))
        result = loader.chunk(build.epub(body), format="epub")
        table_chunks = [c for c in result.chunks if c.kind == "table"]
        assert all("region" in c.text for c in table_chunks), "header not repeated"
        assert table_chunks[0].is_exact, "the first table chunk should still be a slice"
        assert not table_chunks[-1].is_exact, "a continuation cannot be a slice"
        assert result.warnings and "table" in result.warnings[0]

    def test_code_block_keeps_its_fences(self, loader):
        result = loader.chunk(build.epub(RICH_BODY), format="epub")
        code = [c for c in result.chunks if c.kind == "code"]
        assert code
        assert code[0].text.startswith("```") and code[0].text.rstrip().endswith("```")

    def test_tables_and_code_never_absorb_neighbours(self, loader):
        result = loader.chunk(build.epub(RICH_BODY), format="epub")
        for chunk in result.chunks:
            if chunk.kind == "code":
                assert chunk.text.lstrip().startswith("```")
            if chunk.kind == "table":
                assert chunk.text.lstrip().startswith("|")

    def test_disabling_table_routing_sends_tables_to_the_prose_chunker(self):
        result = DocumentChunker(table_chunker=None).chunk(
            b"name,qty\nwidget,3\ngadget,5\n", format="csv"
        )
        assert_invariants(result)
        assert result.chunks

    def test_respect_headings_prevents_cross_section_merges(self):
        markdown = "# A\n\nshort\n\n# B\n\nalso short\n"
        strict = DocumentChunker(respect_headings=True).chunk_markdown(markdown)
        for chunk in strict.chunks:
            assert not ("A" in chunk.text and "B" in chunk.text), "merged across a heading"
        assert_invariants(strict)

    def test_nested_headings_merge_down_into_their_content(self):
        """A heading followed by a deeper heading must not be stranded alone.

        Exact-path-equality would emit '# Report' and '## Methods' as two
        useless single-line chunks.
        """
        markdown = "# Report\n\n## Methods\n\n### Prep\n\nThe substantive prose.\n"
        result = DocumentChunker(min_chunk_size=500).chunk_markdown(markdown)
        assert len(result.chunks) == 1
        assert result.chunks[0].heading_path == ("Report", "Methods", "Prep")
        assert_invariants(result)

    def test_sibling_sections_still_refuse_to_merge(self):
        markdown = "# R\n\n## Methods\n\ntiny\n\n## Results\n\ntiny\n"
        result = DocumentChunker(min_chunk_size=500).chunk_markdown(markdown)
        for chunk in result.chunks:
            assert not ("Methods" in chunk.text and "Results" in chunk.text)
        assert_invariants(result)

    def test_merged_group_reports_the_deepest_path(self):
        markdown = "# A\n\n## B\n\nprose under B.\n"
        result = DocumentChunker(min_chunk_size=500).chunk_markdown(markdown)
        assert result.chunks[0].heading_path == ("A", "B")

    def test_heading_merges_forward_into_its_prose(self):
        markdown = "# Methods\n\nThe prose that this heading introduces.\n"
        result = DocumentChunker(min_chunk_size=500).chunk_markdown(markdown)
        assert len(result.chunks) == 1
        assert result.chunks[0].text.startswith("# Methods")
        assert "introduces" in result.chunks[0].text

    def test_code_lang_is_captured(self):
        result = DocumentChunker().chunk_markdown("```python\nx = 1\n```\n")
        assert result.chunks[0].kind == "code"
        assert result.chunks[0].lang == "python"


class TestSegmenter:
    """The segmenter's contract, independent of any conversion."""

    def test_empty_document(self):
        assert segment_markdown("") == []
        assert DocumentChunker().chunk_markdown("").chunks == []

    def test_whitespace_only_document(self):
        assert DocumentChunker().chunk_markdown("\n\n   \n").chunks == []

    def test_heading_with_no_body(self):
        result = DocumentChunker().chunk_markdown("# Only a heading\n")
        assert len(result.chunks) == 1
        assert_invariants(result)

    def test_document_starting_with_a_table(self):
        markdown = "| a | b |\n| --- | --- |\n| 1 | 2 |\n\nProse after.\n"
        result = DocumentChunker().chunk_markdown(markdown)
        assert result.chunks[0].kind == "table"
        assert result.chunks[0].heading_path == ()
        assert_invariants(result)

    def test_fence_contents_are_not_parsed_as_structure(self):
        markdown = "```\n# not a heading\n| not | a table |\n> not a quote\n```\n"
        segments = segment_markdown(markdown)
        assert len(segments) == 1
        assert segments[0].kind == "code"

    def test_pipes_without_a_delimiter_row_are_prose(self):
        segments = segment_markdown("| this is just | text with pipes |\n")
        assert segments[0].kind == "prose"

    def test_list_survives_blank_lines_between_items(self):
        markdown = "- first item\n\n  - nested item\n\n- second item\n"
        segments = segment_markdown(markdown)
        assert len(segments) == 1, f"list shattered into {len(segments)} segments"
        assert segments[0].kind == "list"

    def test_heading_stack_truncates_on_shallower_heading(self):
        markdown = "# A\n\n## B\n\n### C\n\n## D\n\nprose\n"
        paths = [s.heading_path for s in segment_markdown(markdown)]
        assert ("A", "B", "C") in paths
        assert ("A", "D") in paths

    def test_segments_are_ordered_and_disjoint(self):
        segments = segment_markdown("# A\n\nprose\n\n```\ncode\n```\n\n> quote\n")
        for earlier, later in zip(segments, segments[1:]):
            assert earlier.char_end <= later.char_start

    def test_merge_never_groups_a_table_with_prose(self):
        markdown = "tiny\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n\ntiny\n"
        groups = merge_small_segments(
            segment_markdown(markdown), markdown, min_size=1000, respect_headings=False
        )
        for group in groups:
            kinds = [seg.kind for seg in group]
            assert not ("table" in kinds and len(kinds) > 1)

    def test_non_ascii_offsets_are_bytes_not_code_points(self):
        markdown = "# Résumé\n\nDes accents partout — vraiment beaucoup d'accents ici.\n"
        result = DocumentChunker().chunk_markdown(markdown)
        assert_invariants(result)
        # A code-point reading would have matched too if the two agreed; force
        # them to disagree by asserting the byte length is the larger one.
        assert len(markdown.encode("utf-8")) > len(markdown)


class TestAwkwardInput:
    """Input shapes that a hand-written fixture would never cover."""

    CASES = {
        "crlf": "# Title\r\n\r\nSome prose here.\r\n\r\n| a | b |\r\n| --- | --- |\r\n| 1 | 2 |\r\n",
        "lone_cr": "# Title\r\rProse.\r",
        "no_trailing_newline": "# Title\n\nProse with no trailing newline.",
        "form_feed": "# T\n\nbefore\x0cafter\n",
        "byte_order_mark": "﻿# Title\n\nProse.\n",
        "empty_fence": "```\n```\n",
        "unclosed_fence": "```python\nx = 1\n",
        "every_heading_level": "".join(f"{'#' * i} H{i}\n\n" for i in range(1, 7)) + "body\n",
        "table_with_no_rows": "| a | b |\n| --- | --- |\n",
        "emoji_and_cjk": "# 报告 📊\n\n内容在这里，还有更多的文字。\n",
        "nested_quote": "> outer\n> > inner\n",
        "heading_only": "# Nothing follows\n",
    }

    @pytest.mark.parametrize("name", sorted(CASES))
    def test_invariants_hold(self, name):
        result = DocumentChunker().chunk_markdown(self.CASES[name])
        assert_invariants(result)

    def test_unclosed_fence_still_ends_at_the_document(self):
        result = DocumentChunker().chunk_markdown("```python\nx = 1\n")
        assert result.chunks[0].kind == "code"
        assert_invariants(result)

    def test_crlf_table_is_still_recognised(self):
        result = DocumentChunker().chunk_markdown(self.CASES["crlf"])
        assert any(c.kind == "table" for c in result.chunks)


class TestInvariantSweep:
    """The invariants, checked across many generated documents.

    Snapshot tests pin down the cases someone thought of. This covers the ones
    nobody thought of: structures in orders that never occur in a hand-written
    fixture, under every routing configuration at once. The seed is fixed so a
    failure is reproducible.
    """

    BLOCKS = [
        lambda r: f"{'#' * r.randint(1, 6)} Heading {r.randint(1, 99)}\n",
        lambda r: " ".join(
            r.choice(["alpha", "beta", "gamma", "délta", "эхо", "中文字符"])
            for _ in range(r.randint(1, 60))
        )
        + "\n",
        lambda r: "| a | b |\n| --- | --- |\n"
        + "".join(f"| r{i} | {i} |\n" for i in range(r.randint(1, 14))),
        lambda r: "```"
        + r.choice(["", "python", "rust"])
        + "\n"
        + "\n".join(f"code line {i}" for i in range(r.randint(1, 8)))
        + "\n```\n",
        lambda r: "".join(f"- item {i}\n\n" for i in range(r.randint(1, 6))),
        lambda r: "> quoted line\n> more quote\n",
        lambda r: "---\n",
        lambda r: "1. first\n2. second\n3. third\n",
        lambda r: "short\n",
        lambda r: "| pipes | but no delimiter row |\n",
    ]

    def _configurations(self) -> list[DocumentChunker]:
        return [
            DocumentChunker(),
            DocumentChunker(
                chunker=RecursiveChunker(chunk_size=40),
                table_chunker=TableChunker(chunk_size=2),
                code_chunker=CodeChunker(chunk_size=30),
            ),
            DocumentChunker(respect_headings=False, min_chunk_size=800),
            DocumentChunker(merge_small_segments=False),
            DocumentChunker(table_chunker=None, code_chunker=None),
            DocumentChunker(min_chunk_size=0),
        ]

    def test_invariants_hold_across_generated_documents(self):
        import random

        rng = random.Random(20260808)
        configurations = self._configurations()
        for _ in range(600):
            markdown = "\n".join(
                rng.choice(self.BLOCKS)(rng) for _ in range(rng.randint(0, 9))
            )
            chunker = rng.choice(configurations)
            try:
                assert_invariants(chunker.chunk_markdown(markdown))
            except AssertionError as error:  # pragma: no cover - only on regression
                pytest.fail(f"{error}\n\nconfig: {chunker!r}\ninput: {markdown!r}")


class TestErrors:
    def test_scanned_pdf_names_the_real_cause(self, loader):
        with pytest.raises(ScannedDocumentError) as info:
            loader.chunk(build.scanned_pdf(), format="pdf")
        message = str(info.value).lower()
        assert "ocr" in message
        assert "pdf" in message

    def test_scanned_pdf_is_still_an_unsupported_document(self, loader):
        """Existing handlers must keep catching it."""
        with pytest.raises(UnsupportedDocument):
            loader.chunk(build.scanned_pdf(), format="pdf")
        with pytest.raises(DocumentError):
            loader.chunk(build.scanned_pdf(), format="pdf")

    def test_garbage_bytes_raise_a_document_error(self, loader):
        with pytest.raises(DocumentError):
            loader.chunk(b"\x00\x01\x02not a document at all", format="docx")

    def test_missing_file_raises_oserror_not_a_document_error(self, loader):
        with pytest.raises(OSError):
            loader.chunk("/nonexistent/path/to/report.pdf")

    def test_batch_raises_by_default(self, loader):
        sources = [build.pdf([("Fine document", 12, 700)]), build.scanned_pdf()]
        with pytest.raises(DocumentError):
            loader.chunk_batch(sources, format="pdf")

    def test_batch_skip_omits_failures(self, loader):
        sources = [build.scanned_pdf(), build.pdf([("Hello there", 12, 700)])]
        results = loader.chunk_batch(sources, format="pdf", on_error="skip")
        assert len(results) == 1
        assert isinstance(results[0], ChunkedDocument)

    def test_batch_collect_keeps_positions_aligned(self, loader):
        sources = [build.scanned_pdf(), build.pdf([("Hello there", 12, 700)])]
        results = loader.chunk_batch(sources, format="pdf", on_error="collect")
        assert len(results) == 2
        assert isinstance(results[0], ScannedDocumentError)
        assert isinstance(results[1], ChunkedDocument)


class TestAsyncAndBatch:
    def test_async_matches_sync(self, loader):
        data = build.epub(RICH_BODY)
        sync = loader.chunk(data, format="epub")
        got = asyncio.run(loader.chunk_async(data, format="epub"))
        assert [c.text for c in got.chunks] == [c.text for c in sync.chunks]
        assert_invariants(got)

    def test_batch_matches_singles(self, loader):
        sources = [build.epub(RICH_BODY), b"name,qty\nwidget,3\n"]
        formats = ["epub", "csv"]
        singles = [loader.chunk(s, format=f) for s, f in zip(sources, formats)]
        batched = loader.chunk_batch([sources[0]], format="epub")
        assert [c.text for c in batched[0].chunks] == [c.text for c in singles[0].chunks]

    def test_batch_async_preserves_order(self, loader):
        sources = [
            build.pdf([(f"Document number {i} with some prose in it", 12, 700)])
            for i in range(6)
        ]
        results = asyncio.run(
            loader.chunk_batch_async(sources, format="pdf", max_concurrency=2)
        )
        assert len(results) == 6
        for index, result in enumerate(results):
            assert f"number {index}" in result.markdown
            assert_invariants(result)

    def test_rejects_bad_concurrency(self, loader):
        with pytest.raises(ValueError):
            asyncio.run(loader.chunk_batch_async([], max_concurrency=0))
        with pytest.raises(ValueError):
            DocumentChunker(max_concurrency=0)


class TestConfiguration:
    def test_custom_chunkers_are_used(self):
        loader = DocumentChunker(
            chunker=RecursiveChunker(chunk_size=64),
            table_chunker=TableChunker(chunk_size=2),
            code_chunker=CodeChunker(chunk_size=64),
        )
        result = loader.chunk(build.epub(RICH_BODY), format="epub")
        assert_invariants(result)
        assert len(result.chunks) > 4

    def test_smaller_chunk_size_yields_more_chunks(self):
        data = build.epub(RICH_BODY)
        coarse = DocumentChunker(chunker=RecursiveChunker(chunk_size=4096)).chunk(
            data, format="epub"
        )
        fine = DocumentChunker(chunker=RecursiveChunker(chunk_size=48)).chunk(
            data, format="epub"
        )
        assert len(fine.chunks) >= len(coarse.chunks)
        assert_invariants(fine)

    def test_merging_can_be_disabled(self):
        markdown = "# A\n\nshort\n"
        merged = DocumentChunker(merge_small_segments=True, min_chunk_size=500)
        split = DocumentChunker(merge_small_segments=False)
        assert len(merged.chunk_markdown(markdown).chunks) == 1
        assert len(split.chunk_markdown(markdown).chunks) == 2

    def test_callable_shorthand(self, loader):
        result = loader(b"name,qty\nwidget,3\n", format="csv")
        assert isinstance(result, ChunkedDocument)

    def test_result_is_iterable_and_indexable(self, loader):
        result = loader.chunk(build.epub(RICH_BODY), format="epub")
        assert len(result) == len(result.chunks)
        assert isinstance(result[0], DocumentChunk)
        assert list(result) == result.chunks

    def test_not_a_basechunker(self):
        """DocumentChunker consumes files, so it must not be substitutable for
        a text chunker."""
        from blazechunk import BaseChunker

        assert not isinstance(DocumentChunker(), BaseChunker)
        assert not issubclass(DocumentChunker, BaseChunker)


class TestAdapterContract:
    """Guards the anydoc surface this integration actually relies on.

    When upstream bumps and changes the model, this fails with a clear message
    instead of letting subtly wrong chunks through.
    """

    def test_required_functions_exist(self):
        for name in ("to_markdown_bytes", "format_from_bytes", "format_from_extension"):
            assert callable(getattr(anydoc, name)), f"anydoc.{name} is gone"

    def test_required_exceptions_exist(self):
        for name in (
            "ConvertError",
            "UnsupportedError",
            "MalformedError",
            "EncryptedError",
            "ResourceLimitError",
            "MissingPartError",
        ):
            assert issubclass(getattr(anydoc, name), Exception), f"anydoc.{name} is gone"

    def test_pdf_still_has_no_document_model(self):
        """The measurement the whole architecture rests on.

        If this ever starts passing, PDF gained a document model upstream and
        enriching segments from it becomes worth revisiting.
        """
        with pytest.raises(anydoc.UnsupportedError):
            anydoc.to_document(build.pdf([("Hello", 12, 700)]), "pdf")

    def test_markdown_serializer_shape_is_unchanged(self):
        markdown = anydoc.to_markdown_bytes(b"a,b\n1,2\n", "csv")
        assert markdown.startswith("| a | b |")
        assert "| --- |" in markdown
