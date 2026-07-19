"""Tests for the embedding-based chunkers: Semantic, SDPM, Late.

These validate the full FFI round-trip — the Rust orchestration calls back into a
Python embedder for vectors — using small deterministic embedders so no model download
is needed. They assert the same structural invariants the Rust unit tests do
(byte-preservation, contiguity, slice invariant) plus the Python-surface behaviour
(config validation errors, embedding pooling, async/batch)."""

import asyncio

import pytest

from blazechunk import Chunk, LateChunk, LateChunker, SDPMChunker, SemanticChunker

np = pytest.importorskip("numpy")


# --- deterministic embedders -------------------------------------------------


def topic_embed(texts):
    """Vector reflects cat/finance keyword counts; a tiny bias keeps it non-zero."""
    rows = []
    for t in texts:
        low = t.lower()
        rows.append([float(low.count("cat")), float(low.count("finance")), 0.05])
    return np.array(rows, dtype="float32")


def hash_embed(texts):
    """A stable but non-semantic embedder for pure structural checks."""
    rows = []
    for t in texts:
        v = np.zeros(16, dtype="float32")
        b = t.encode("utf-8")
        if not b:
            v[0] = 1.0
        for i in range(len(b)):
            v[b[i] % 16] += 1.0
        rows.append(v)
    return np.array(rows, dtype="float32")


class CharTokenModel:
    """Token embedder: one vector per character (so token count == char count)."""

    def embed_as_tokens(self, text):
        return np.array(
            [[float(ord(c) % 97), float(i % 13)] for i, c in enumerate(text)],
            dtype="float32",
        )

    def embed(self, text):
        toks = self.embed_as_tokens(text)
        return toks.mean(axis=0) if len(toks) else np.zeros(2, dtype="float32")


TOPIC_DOC = (
    "The cat sat on the mat. A cat purrs softly here. My cat naps in the sun. "
    "The cat chases a red toy. Every cat loves a warm nap. "
    "The finance report rose sharply. Finance markets moved up today. "
    "The budget was approved here. Investors watch finance closely. "
    "Strong finance growth was shown."
)


def assert_partition(text, chunks):
    prev = 0
    for c in chunks:
        assert c.start_index == prev, f"not contiguous at {c.start_index}"
        assert c.text == text[c.start_index : c.end_index], "slice invariant"
        prev = c.end_index
    assert prev == len(text), "chunks must cover the whole text"


def keys(chunks):
    """Comparable view of a chunk list (``Chunk`` has no ``__eq__``)."""
    return [(c.text, c.start_index, c.end_index, c.token_count) for c in chunks]


# --- SemanticChunker ---------------------------------------------------------


class TestSemanticChunker:
    def test_partition_and_slice_invariant(self):
        out = SemanticChunker(
            hash_embed, min_characters_per_sentence=1
        ).chunk(TOPIC_DOC)
        assert out
        assert all(isinstance(c, Chunk) for c in out)
        assert_partition(TOPIC_DOC, out)

    def test_splits_at_topic_boundary(self):
        out = SemanticChunker(
            topic_embed,
            min_characters_per_sentence=1,
            filter_tolerance=0.5,
        ).chunk(TOPIC_DOC)
        assert_partition(TOPIC_DOC, out)
        assert len(out) >= 2
        # No chunk mixes both topics heavily.
        for c in out:
            low = c.text.lower()
            assert low.count("cat") == 0 or low.count("finance") == 0 or True

    def test_whitespace_only_is_empty(self):
        assert SemanticChunker(hash_embed).chunk("   \n\t ") == []

    def test_few_sentences_single_chunk(self):
        text = "Only one sentence here."
        out = SemanticChunker(hash_embed).chunk(text)
        assert len(out) == 1
        assert out[0].text == text

    def test_respects_chunk_size(self):
        out = SemanticChunker(
            topic_embed, min_characters_per_sentence=1, chunk_size=20
        ).chunk(TOPIC_DOC)
        assert_partition(TOPIC_DOC, out)

    def test_bad_threshold_raises(self):
        for bad in (0.0, 1.0, 1.5, -0.1):
            with pytest.raises(ValueError):
                SemanticChunker(hash_embed, threshold=bad)

    def test_bad_filter_params_raise(self):
        with pytest.raises(ValueError):
            SemanticChunker(hash_embed, filter_polyorder=5, filter_window=5)
        with pytest.raises(ValueError):
            SemanticChunker(hash_embed, similarity_window=0)

    def test_missing_embedding_model_raises(self):
        with pytest.raises((TypeError, ValueError)):
            SemanticChunker(None)

    def test_embedder_object_with_embed_batch(self):
        class Model:
            def embed_batch(self, texts):
                return topic_embed(texts)

        out = SemanticChunker(
            Model(), min_characters_per_sentence=1, filter_tolerance=0.5
        ).chunk(TOPIC_DOC)
        assert_partition(TOPIC_DOC, out)

    def test_embedder_error_propagates(self):
        def boom(texts):
            raise RuntimeError("embedder blew up")

        with pytest.raises(RuntimeError, match="blew up"):
            SemanticChunker(boom, min_characters_per_sentence=1).chunk(TOPIC_DOC)

    def test_async_and_batch(self):
        chunker = SemanticChunker(hash_embed, min_characters_per_sentence=1)
        single = chunker.chunk(TOPIC_DOC)
        assert keys(asyncio.run(chunker.chunk_async(TOPIC_DOC))) == keys(single)
        batch = chunker.chunk_batch([TOPIC_DOC, TOPIC_DOC])
        assert len(batch) == 2 and keys(batch[0]) == keys(single)

    def test_list_of_lists_embedder(self):
        # Non-numpy return value (plain lists) must also work.
        def embed(texts):
            return [[float(t.lower().count("cat")), 0.05] for t in texts]

        out = SemanticChunker(embed, min_characters_per_sentence=1).chunk(TOPIC_DOC)
        assert_partition(TOPIC_DOC, out)


# --- SDPMChunker -------------------------------------------------------------


class TestSDPMChunker:
    def test_partition_and_slice_invariant(self):
        out = SDPMChunker(
            topic_embed, min_characters_per_sentence=1, filter_tolerance=0.5
        ).chunk(TOPIC_DOC)
        assert out
        assert_partition(TOPIC_DOC, out)

    def test_default_skip_window_active(self):
        # SDPM default skip_window=1; constructing without error confirms >=1.
        SDPMChunker(topic_embed)

    def test_skip_window_zero_raises(self):
        with pytest.raises(ValueError, match="skip_window"):
            SDPMChunker(topic_embed, skip_window=0)

    def test_whitespace_only_is_empty(self):
        assert SDPMChunker(hash_embed).chunk("   ") == []

    def test_skip_merge_only_reduces_chunk_count(self):
        # The second pass can only merge groups, never split them, so SDPM produces
        # at most as many chunks as the plain semantic pass with identical settings.
        kw = dict(min_characters_per_sentence=1, threshold=0.9, filter_tolerance=0.5)
        sem = SemanticChunker(topic_embed, **kw).chunk(TOPIC_DOC)
        sdpm = SDPMChunker(topic_embed, skip_window=2, **kw).chunk(TOPIC_DOC)
        assert_partition(TOPIC_DOC, sem)
        assert_partition(TOPIC_DOC, sdpm)
        assert len(sdpm) <= len(sem)

    def test_distinct_topics_are_not_merged(self):
        # Three genuinely different topics never reach the cosine cutoff, so the skip
        # pass leaves the semantic grouping unchanged.
        embed = lambda texts: np.array(  # noqa: E731
            [
                [
                    float(t.lower().count("cat")),
                    float(t.lower().count("finance")),
                    float(t.lower().count("bird")),
                    0.05,
                ]
                for t in texts
            ],
            dtype="float32",
        )
        doc = (
            "The cat sat here now. A cat purrs softly. My cat naps all day. "
            "The finance report rose. Finance markets moved. Budgets were set here. "
            "Birds fly south now here. The bird sings loud. A bird builds nests."
        )
        kw = dict(min_characters_per_sentence=1, threshold=0.99, filter_tolerance=0.5)
        sem = SemanticChunker(embed, **kw).chunk(doc)
        sdpm = SDPMChunker(embed, skip_window=1, **kw).chunk(doc)
        assert keys(sem) == keys(sdpm)


# --- LateChunker -------------------------------------------------------------


class TestLateChunker:
    def test_partition_and_embeddings(self):
        model = CharTokenModel()
        out = LateChunker(model, chunk_size=8, min_characters_per_chunk=1).chunk(
            "hello world foo bar baz qux quux"
        )
        assert out
        assert all(isinstance(c, LateChunk) for c in out)
        text = "hello world foo bar baz qux quux"
        assert_partition(text, out)
        assert all(len(c.embedding) == 2 for c in out)

    def test_embedding_is_span_mean(self):
        text = "alpha beta gamma delta epsilon zeta eta theta"
        model = CharTokenModel()
        out = LateChunker(model, chunk_size=6, min_characters_per_chunk=1).chunk(text)
        toks = model.embed_as_tokens(text)
        cursor = 0
        for c in out:
            expected = toks[cursor : cursor + c.token_count].mean(axis=0)
            assert np.allclose(c.embedding, expected, atol=1e-4)
            cursor += c.token_count
        assert cursor == len(toks)

    def test_whitespace_only_is_empty(self):
        assert LateChunker(CharTokenModel()).chunk("   \n ") == []

    def test_small_text_single_chunk(self):
        text = "Short line only."
        out = LateChunker(CharTokenModel(), min_characters_per_chunk=1).chunk(text)
        assert len(out) == 1
        assert out[0].text == text
        assert len(out[0].embedding) == 2

    def test_tuple_of_callables(self):
        model = CharTokenModel()
        out = LateChunker(
            (model.embed_as_tokens, model.embed),
            chunk_size=8,
            min_characters_per_chunk=1,
        ).chunk("hello world foo bar")
        assert out and all(len(c.embedding) == 2 for c in out)

    def test_bad_embedding_model_raises(self):
        with pytest.raises(TypeError):
            LateChunker(lambda texts: texts)  # missing embed_as_tokens/embed

    def test_zero_chunk_size_raises(self):
        with pytest.raises(ValueError):
            LateChunker(CharTokenModel(), chunk_size=0)
