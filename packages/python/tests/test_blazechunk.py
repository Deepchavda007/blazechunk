import asyncio
from pathlib import Path

import pytest
from blazechunk import Chunker, DEFAULT_TARGET_SIZE, DEFAULT_DELIMITERS


class TestChunker:
    def test_basic_chunking(self):
        text = b"Hello. World. Test."
        chunks = list(Chunker(text, size=10, delimiters=b"."))
        assert len(chunks) == 3
        assert chunks[0] == b"Hello."
        assert chunks[1] == b" World."
        assert chunks[2] == b" Test."

    def test_newline_delimiter(self):
        text = b"Line one\nLine two\nLine three"
        chunks = list(Chunker(text, size=15, delimiters=b"\n"))
        assert chunks[0] == b"Line one\n"
        assert chunks[1] == b"Line two\n"
        assert chunks[2] == b"Line three"

    def test_multiple_delimiters(self):
        text = b"Hello? World. Yes!"
        chunks = list(Chunker(text, size=10, delimiters=b".?!"))
        assert chunks[0] == b"Hello?"

    def test_no_delimiter_hard_split(self):
        text = b"abcdefghij"
        chunks = list(Chunker(text, size=5, delimiters=b"."))
        assert chunks[0] == b"abcde"
        assert chunks[1] == b"fghij"

    def test_empty_text(self):
        text = b""
        chunks = list(Chunker(text, size=10, delimiters=b"."))
        assert len(chunks) == 0

    def test_text_smaller_than_target(self):
        text = b"Small"
        chunks = list(Chunker(text, size=100, delimiters=b"."))
        assert len(chunks) == 1
        assert chunks[0] == b"Small"

    def test_total_bytes_preserved(self):
        text = b"The quick brown fox jumps over the lazy dog. How vexingly quick!"
        chunks = list(Chunker(text, size=20, delimiters=b"\n.?!"))
        total = sum(len(c) for c in chunks)
        assert total == len(text)

    def test_defaults(self):
        text = b"Hello world. This is a test."
        chunks = list(Chunker(text))
        assert len(chunks) > 0

    def test_iterator_protocol(self):
        text = b"Hello. World."
        chunker = Chunker(text, size=10, delimiters=b".")
        it = iter(chunker)
        assert next(it) == b"Hello."
        assert next(it) == b" World."
        with pytest.raises(StopIteration):
            next(it)

    def test_reset(self):
        text = b"Hello. World."
        chunker = Chunker(text, size=10, delimiters=b".")
        chunks1 = list(chunker)
        chunker.reset()
        chunks2 = list(chunker)
        assert chunks1 == chunks2

    def test_four_delimiters(self):
        """Test that 4+ delimiters work (uses lookup table internally)."""
        text = b"A. B? C! D; E"
        chunks = list(Chunker(text, size=5, delimiters=b".?!;"))
        assert len(chunks) >= 2


class TestStrInput:
    """Test that str input works (encoded as UTF-8)."""

    def test_str_text(self):
        text = "Hello. World. Test."
        chunks = list(Chunker(text, size=10, delimiters=b"."))
        assert len(chunks) == 3
        assert chunks[0] == b"Hello."

    def test_str_delimiters(self):
        text = b"Hello. World. Test."
        chunks = list(Chunker(text, size=10, delimiters="."))
        assert len(chunks) == 3
        assert chunks[0] == b"Hello."

    def test_str_both(self):
        text = "Hello. World. Test."
        chunks = list(Chunker(text, size=10, delimiters="."))
        assert len(chunks) == 3
        assert chunks[0] == b"Hello."

    def test_unicode(self):
        text = "Caf\u00e9. Tea."  # é is 2 bytes in UTF-8
        chunks = list(Chunker(text, size=10, delimiters="."))
        assert len(chunks) == 2
        # Verify UTF-8 encoding is preserved
        assert b"\xc3\xa9" in chunks[0]  # é in UTF-8
        assert chunks[0] == "Café.".encode("utf-8")


class TestPatterns:
    """Test multi-byte pattern support via .patterns() API."""

    def test_patterns_basic(self):
        text = "Hello。World，Test"
        chunks = list(Chunker(text, size=20, delimiters="\n.?!", patterns=["。", "，"]))
        total = sum(len(c) for c in chunks)
        assert total == len(text.encode("utf-8"))
        # Every chunk should be valid UTF-8
        for c in chunks:
            c.decode("utf-8")

    def test_patterns_composable_with_delimiters(self):
        text = "Hello. World。Test"
        chunks = list(Chunker(text, size=12, delimiters=".", patterns=["。"]))
        assert len(chunks) >= 2
        total = sum(len(c) for c in chunks)
        assert total == len(text.encode("utf-8"))

    def test_patterns_bytes_list(self):
        text = b"Hello\xe3\x80\x82World"  # 。is \xe3\x80\x82
        chunks = list(Chunker(text, size=10, delimiters=b"", patterns=["。"]))
        total = sum(len(c) for c in chunks)
        assert total == len(text)

    def test_patterns_chunk_offsets(self):
        from blazechunk import chunk_offsets

        text = "Hello。World。Test"
        offsets = chunk_offsets(text, size=15, delimiters="", patterns=["。"])
        assert len(offsets) >= 2
        total = sum(end - start for start, end in offsets)
        assert total == len(text.encode("utf-8"))

    def test_patterns_chunk_convenience(self):
        from blazechunk import chunk

        text = b"Hello. World\xe3\x80\x82Test"
        results = list(chunk(text, size=12, patterns=["\xe3\x80\x82"]))
        assert len(results) >= 2

    def test_patterns_utf8_safety(self):
        """Ensure multi-byte characters are not split mid-codepoint."""
        text = "It\u2019s a test。Done"  # \u2019 = right single quote
        chunks = list(
            Chunker(text, size=20, delimiters=".", patterns=["。"], forward_fallback=True)
        )
        total = sum(len(c) for c in chunks)
        assert total == len(text.encode("utf-8"))
        for c in chunks:
            c.decode("utf-8")  # raises if invalid UTF-8


class TestConstants:
    def test_default_target_size(self):
        assert DEFAULT_TARGET_SIZE == 4096

    def test_default_delimiters(self):
        assert DEFAULT_DELIMITERS == b"\n.?"


def _slice(text: str, chunk):
    """Reconstruct chunk.text from the original via byte offsets."""
    return text.encode("utf-8")[chunk.start_index : chunk.end_index].decode("utf-8")


class TestRecursiveChunker:
    def test_small_text_one_chunk(self):
        from blazechunk import RecursiveChunker

        out = RecursiveChunker().chunk("Hello world.")
        assert len(out) == 1
        assert out[0].text == "Hello world."
        assert out[0].token_count == 12

    def test_empty_returns_none(self):
        from blazechunk import RecursiveChunker

        assert RecursiveChunker().chunk("") == []

    def test_slice_invariant(self):
        from blazechunk import RecursiveChunker

        text = "Para one.\n\nPara two is longer. It has two sentences.\n\nThird one here."
        out = RecursiveChunker(chunk_size=20).chunk(text)
        assert len(out) > 1
        for c in out:
            assert c.text == _slice(text, c)

    def test_cjk_delimiters(self):
        from blazechunk import RecursiveChunker

        # Would have raised UnicodeDecodeError in the buggy upstream path (#536).
        text = "第一句。第二句。第三句。" * 5
        out = RecursiveChunker(chunk_size=8, min_characters_per_chunk=1).chunk(text)
        for c in out:
            c.text.encode("utf-8")  # valid str already
            assert c.text == _slice(text, c)

    def test_bad_chunk_size_raises(self):
        from blazechunk import RecursiveChunker

        with pytest.raises(ValueError):
            RecursiveChunker(chunk_size=0)

    def test_bad_min_characters_raises(self):
        from blazechunk import RecursiveChunker

        with pytest.raises(ValueError):
            RecursiveChunker(min_characters_per_chunk=0)

    def test_custom_rules_kwarg(self):
        from blazechunk import RecursiveChunker

        # A custom hierarchy: split on '|', then whitespace, then hard token split.
        text = "alpha|beta|gamma|delta|epsilon|zeta"
        rules = [
            {"delimiters": ["|"], "include_delim": "prev"},
            {"type": "whitespace"},
            {"type": "token"},
        ]
        out = RecursiveChunker(
            chunk_size=2, min_characters_per_chunk=1, rules=rules
        ).chunk(text)
        assert out
        # Byte-preserving and contiguous with the custom delimiter honored.
        raw = text.encode("utf-8")
        prev = 0
        for c in out:
            assert c.start_index == prev
            assert c.text == raw[c.start_index : c.end_index].decode("utf-8")
            prev = c.end_index
        assert prev == len(raw)

    def test_token_only_rule(self):
        from blazechunk import RecursiveChunker

        # A single token level hard-splits into fixed-size groups.
        out = RecursiveChunker(chunk_size=3, rules=[{"type": "token"}]).chunk(
            "abcdefgh"
        )
        assert [c.text for c in out] == ["abc", "def", "gh"]

    def test_bad_rules_raise(self):
        from blazechunk import RecursiveChunker

        with pytest.raises(ValueError):
            RecursiveChunker(rules=[])
        with pytest.raises(ValueError):
            RecursiveChunker(rules=[{"nonsense": True}])


class TestSentenceChunker:
    def test_packs_sentences(self):
        from blazechunk import SentenceChunker

        text = "One. Two. Three. Four. Five. "
        out = SentenceChunker(chunk_size=10, min_characters_per_sentence=1).chunk(text)
        assert len(out) >= 2
        for c in out:
            assert c.text == _slice(text, c)

    def test_overlap(self):
        from blazechunk import SentenceChunker

        text = "A. B. C. D. E. F. "
        out = SentenceChunker(
            chunk_size=9, chunk_overlap=4, min_characters_per_sentence=1
        ).chunk(text)
        assert any(out[i + 1].start_index < out[i].end_index for i in range(len(out) - 1))

    def test_overlap_ge_size_raises(self):
        from blazechunk import SentenceChunker

        with pytest.raises(ValueError):
            SentenceChunker(chunk_size=10, chunk_overlap=10)

    def test_min_sentences_zero_raises(self):
        from blazechunk import SentenceChunker

        with pytest.raises(ValueError):
            SentenceChunker(min_sentences_per_chunk=0)

    def test_min_characters_zero_raises(self):
        from blazechunk import SentenceChunker

        with pytest.raises(ValueError):
            SentenceChunker(min_characters_per_sentence=0)


class TestTokenChunker:
    def test_strides_with_overlap(self):
        from blazechunk import TokenChunker

        out = TokenChunker(chunk_size=4, chunk_overlap=1).chunk("abcdefghij")
        assert out[0].text == "abcd"
        assert out[1].text == "defg"  # step = 3

    def test_multibyte_slice_invariant(self):
        from blazechunk import TokenChunker

        # Regression for #629: boundaries never corrupt multi-byte characters.
        text = "a🩺bc🩺de"
        out = TokenChunker(chunk_size=2).chunk(text)
        for c in out:
            assert c.text != ""
            assert c.text == _slice(text, c)
        assert "".join(c.text for c in out) == text

    def test_float_overlap_full_raises(self):
        from blazechunk import TokenChunker

        with pytest.raises(ValueError):
            TokenChunker(chunk_size=4, chunk_overlap=1.0).chunk("abcd")


class TestTableChunker:
    MD = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n| 5 | 6 |\n"

    def test_row_split_reincludes_header(self):
        from blazechunk import TableChunker

        out = TableChunker(chunk_size=2).chunk(self.MD)
        assert len(out) >= 2
        for c in out:
            assert "| A | B |" in c.text
            assert "|---|---|" in c.text  # separator preserved (#582)

    def test_small_table_single_chunk(self):
        from blazechunk import TableChunker

        out = TableChunker(chunk_size=10).chunk(self.MD)
        assert len(out) == 1
        assert out[0].text == self.MD

    def test_empty_returns_none(self):
        from blazechunk import TableChunker

        assert TableChunker().chunk("   ") == []

    def test_bad_chunk_size_raises(self):
        from blazechunk import TableChunker

        with pytest.raises(ValueError):
            TableChunker(chunk_size=0)


class TestCodeChunker:
    CODE = "fn a() {\n    let x = 1;\n}\n\nfn b() {\n    let y = 2;\n}\n\nfn c() {\n    let z = 3;\n}\n"

    def test_slice_invariant(self):
        from blazechunk import CodeChunker

        out = CodeChunker(chunk_size=20, language="rust").chunk(self.CODE)
        assert len(out) > 1
        prev = 0
        for c in out:
            assert c.start_index == prev
            assert c.text == _slice(self.CODE, c)
            prev = c.end_index
        assert prev == len(self.CODE.encode("utf-8"))

    def test_auto_detect(self):
        from blazechunk import CodeChunker

        out = CodeChunker(chunk_size=200).chunk(self.CODE)
        assert len(out) == 1


class TestChunkType:
    def test_chunk_fields_and_repr(self):
        from blazechunk import RecursiveChunker

        c = RecursiveChunker().chunk("Hello world.")[0]
        assert c.text == "Hello world."
        assert c.start_index == 0
        assert c.end_index == 12
        assert c.token_count == 12
        assert len(c) == 12
        assert "Chunk(" in repr(c)


# Adversarial inputs exercised against every chunker's invariants.
STRESS_INPUTS = [
    "",
    "   \n\t ",
    "no delimiters at all just plain words here without punctuation",
    "Short. Sentences? Yes! Ok.\n\nA new paragraph follows here with text.",
    "第一句。第二句！第三句？第四句。第五句。",  # CJK, multi-byte
    "a🩺b🎉c👨‍👩‍👧d🩺e",  # emoji incl. ZWJ sequence
    "Café ☕ costs €3.50. Is that ok?\n\nDone。Really done！",  # mixed scripts
    "word " * 300,  # long
    "fn a() {\n    let x = 1;\n}\n\nfn b() {\n    return;\n}\n",  # code
]


class TestInvariants:
    """Property-style checks: no crashes, and the slice invariant holds."""

    def _slice_ok(self, text, chunks):
        raw = text.encode("utf-8")
        for c in chunks:
            assert 0 <= c.start_index <= c.end_index <= len(raw)
            assert c.text == raw[c.start_index : c.end_index].decode("utf-8")
            assert c.text != ""  # no empty chunks (incl. #629 multibyte case)

    def _contiguous_full(self, text, chunks):
        raw = text.encode("utf-8")
        prev = 0
        for c in chunks:
            assert c.start_index == prev
            prev = c.end_index
        assert prev == len(raw)

    @pytest.mark.parametrize("text", STRESS_INPUTS)
    def test_recursive(self, text):
        from blazechunk import RecursiveChunker

        out = RecursiveChunker(chunk_size=10, min_characters_per_chunk=1).chunk(text)
        if text == "":  # Recursive only short-circuits on truly empty input
            assert out == []
            return
        self._slice_ok(text, out)
        self._contiguous_full(text, out)

    @pytest.mark.parametrize("text", STRESS_INPUTS)
    def test_sentence(self, text):
        from blazechunk import SentenceChunker

        out = SentenceChunker(chunk_size=10, min_characters_per_sentence=1).chunk(text)
        if not text.strip():
            assert out == []
            return
        self._slice_ok(text, out)

    @pytest.mark.parametrize("text", STRESS_INPUTS)
    def test_token(self, text):
        from blazechunk import TokenChunker

        out = TokenChunker(chunk_size=5, chunk_overlap=1).chunk(text)
        if not text.strip():
            assert out == []
            return
        self._slice_ok(text, out)
        assert out[0].start_index == 0
        assert out[-1].end_index == len(text.encode("utf-8"))

    @pytest.mark.parametrize("text", STRESS_INPUTS)
    def test_code(self, text):
        from blazechunk import CodeChunker

        out = CodeChunker(chunk_size=10).chunk(text)
        if not text.strip():
            assert out == []
            return
        self._slice_ok(text, out)
        self._contiguous_full(text, out)


# --- Optional HuggingFace tokenizer (requires maturin --features hf-tokenizer) ---

_HF_FIXTURE = Path(__file__).parent / "fixtures" / "wordlevel_tokenizer.json"


def _require_hf_tokenizer():
    """Skip when the extension was built without the hf-tokenizer feature."""
    from blazechunk import TokenChunker

    try:
        TokenChunker(tokenizer=str(_HF_FIXTURE), chunk_size=8)
    except ValueError as e:
        if "unknown tokenizer" in str(e):
            pytest.skip("hf-tokenizer feature not built into this extension")
        raise


class TestHfTokenizer:
    """Real subword counts via a local tokenizer.json (Stream D)."""

    def test_token_chunker_slice_invariant(self):
        _require_hf_tokenizer()
        from blazechunk import TokenChunker

        text = "hello world foo bar the quick brown fox"
        out = TokenChunker(tokenizer=str(_HF_FIXTURE), chunk_size=2).chunk(text)
        assert out
        raw = text.encode("utf-8")
        for c in out:
            assert c.text == raw[c.start_index : c.end_index].decode("utf-8")
        assert out[0].start_index == 0
        assert out[-1].end_index == len(raw)

    def test_known_token_counts(self):
        _require_hf_tokenizer()
        from blazechunk import TokenChunker

        # WordLevel+Whitespace: one token per whitespace-separated word.
        out = TokenChunker(tokenizer=str(_HF_FIXTURE), chunk_size=100).chunk(
            "hello world"
        )
        assert len(out) == 1
        assert out[0].token_count == 2

    def test_missing_file_raises(self):
        _require_hf_tokenizer()
        from blazechunk import TokenChunker

        with pytest.raises(ValueError):
            TokenChunker(tokenizer="/nonexistent/path/tokenizer.json", chunk_size=8)


# --- Synchronous / asynchronous / batch API parity -------------------------


def _key(chunks):
    """Comparable representation of a chunk list (Chunk has no __eq__)."""
    return [(c.text, c.start_index, c.end_index, c.token_count) for c in chunks]


def _make_chunkers():
    """One configured instance of each high-level chunker, with sample text."""
    from blazechunk import (
        CodeChunker,
        RecursiveChunker,
        SentenceChunker,
        TableChunker,
        TokenChunker,
    )

    md = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n| 5 | 6 |\n"
    code = "fn a() {\n    let x = 1;\n}\n\nfn b() {\n    let y = 2;\n}\n"
    return [
        (RecursiveChunker(chunk_size=20), "Para one.\n\nPara two is longer here."),
        (SentenceChunker(chunk_size=10, min_characters_per_sentence=1), "One. Two. Three. Four."),
        (TokenChunker(chunk_size=4, chunk_overlap=1), "abcdefghij"),
        (TableChunker(chunk_size=2), md),
        (CodeChunker(chunk_size=20, language="rust"), code),
    ]


class TestAsyncParity:
    """Every chunker's async/batch/callable results must match plain chunk()."""

    def test_chunk_async_matches_sync(self):
        for chunker, text in _make_chunkers():
            expected = _key(chunker.chunk(text))
            got = _key(asyncio.run(chunker.chunk_async(text)))
            assert got == expected, type(chunker).__name__

    def test_call_matches_chunk(self):
        for chunker, text in _make_chunkers():
            assert _key(chunker(text)) == _key(chunker.chunk(text))

    def test_chunk_batch_matches_per_item(self):
        for chunker, text in _make_chunkers():
            texts = [text, text[: len(text) // 2], text]
            batched = chunker.chunk_batch(texts)
            assert [_key(b) for b in batched] == [_key(chunker.chunk(t)) for t in texts]

    def test_chunk_batch_async_matches_sync_batch(self):
        for chunker, text in _make_chunkers():
            texts = [text, "", text[:5], text]
            expected = [_key(b) for b in chunker.chunk_batch(texts)]
            got = [_key(b) for b in asyncio.run(chunker.chunk_batch_async(texts))]
            assert got == expected, type(chunker).__name__

    def test_chunk_batch_async_bounded_concurrency(self):
        chunker, text = _make_chunkers()[0]
        texts = [text] * 10
        expected = [_key(b) for b in chunker.chunk_batch(texts)]
        got = [
            _key(b)
            for b in asyncio.run(chunker.chunk_batch_async(texts, max_concurrency=3))
        ]
        assert got == expected

    def test_bad_max_concurrency_raises(self):
        chunker, text = _make_chunkers()[0]
        with pytest.raises(ValueError):
            asyncio.run(chunker.chunk_batch_async([text], max_concurrency=0))

    def test_empty_batch(self):
        chunker, _ = _make_chunkers()[0]
        assert chunker.chunk_batch([]) == []
        assert asyncio.run(chunker.chunk_batch_async([])) == []

    def test_all_are_base_chunkers(self):
        from blazechunk import BaseChunker

        for chunker, _ in _make_chunkers():
            assert isinstance(chunker, BaseChunker)

    def test_validation_still_raises_through_wrapper(self):
        from blazechunk import SentenceChunker, TokenChunker

        with pytest.raises(ValueError):
            TokenChunker(chunk_size=0)
        with pytest.raises(ValueError):
            SentenceChunker(chunk_size=10, chunk_overlap=10)


class TestFastPathAsync:
    """The low-level chunk()/chunk_async() convenience helpers."""

    def test_chunk_async_matches_chunk(self):
        from blazechunk import chunk, chunk_async

        text = b"Hello. World. Test. More text here."
        sync = [bytes(c) for c in chunk(text, size=10, delimiters=b".")]
        got = asyncio.run(chunk_async(text, size=10, delimiters=b"."))
        assert got == sync

    def test_chunk_async_returns_bytes(self):
        from blazechunk import chunk_async

        got = asyncio.run(chunk_async("Hello. World.", size=10, delimiters="."))
        assert all(isinstance(c, bytes) for c in got)
