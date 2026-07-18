# Contributing to blazechunk

First off — thank you! 🎉 blazechunk is open source and **contributions are very
welcome**, whether that's a bug report, a docs fix, a new chunker, or a performance
improvement. This guide gets you set up and explains how we work.

## Ways to contribute

- 🐛 **Report a bug** — [open an issue](https://github.com/Deepchavda007/blazechunk/issues)
  with a minimal reproduction (input text + the chunker config + what you expected).
- 💡 **Request a feature** — open an issue describing the use case.
- 📖 **Improve the docs** — even fixing a typo in a docstring or the README helps.
- 🔧 **Send a pull request** — see the workflow below.

If you're planning a large change, please open an issue first so we can agree on the
approach before you invest the time.

## Project layout

blazechunk is a SIMD-accelerated Rust core with a thin, well-typed Python API on top:

```
src/                         # the Rust core crate (`chunk`) — chunking primitives
  chunkers/                  # the five high-level chunkers
packages/python/             # the `blazechunk` Python package (PyO3 bindings)
  src/lib.rs                 # the #[pyclass] wrappers
  python/blazechunk/         # pure-Python API: chunkers.py, __init__.py, _chunk.pyi
  tests/                     # pytest suite
benches/                     # criterion benchmarks
```

The Python chunkers are thin wrappers over the compiled `_chunk` extension; the real
logic lives in Rust. Most behavior changes therefore start in `src/`.

## Development setup

You'll need the [Rust toolchain](https://rustup.rs/) (edition 2024, so Rust ≥ 1.85) and
Python ≥ 3.10. [`uv`](https://github.com/astral-sh/uv) is recommended for the Python env.

```bash
# clone your fork
git clone https://github.com/<you>/blazechunk.git
cd blazechunk

# set up the Python env and build the extension into it
cd packages/python
uv venv
uv pip install pytest numpy maturin
maturin develop --release
# optional: build with the real HuggingFace tokenizer support
#   maturin develop --release --features hf-tokenizer
```

## Running the tests

```bash
# Python tests (from packages/python, with the venv active)
python -m pytest tests/ -v

# Rust tests (from the repo root)
cargo test
```

After any change to the Rust source, re-run `maturin develop` so the Python side picks
it up.

## Code style & checks

Please make sure these pass before opening a PR:

```bash
cargo fmt --check            # Rust formatting
cargo clippy -- -D warnings  # Rust lints (warnings are errors)
python -m pytest tests/      # Python tests green
```

- **Rust:** format with `cargo fmt`; keep `clippy` clean.
- **Python:** format with [black](https://github.com/psf/black); add type hints and a
  docstring to every public function/method (see `packages/python/python/blazechunk/chunkers.py`
  for the house style).

## Invariants to preserve

These are load-bearing — new chunkers and offset changes must keep them:

- **Byte offsets everywhere.** Every `Chunk` carries byte offsets into the original text.
- **Slice invariant.** For every chunker *except* `TableChunker`,
  `chunk.text == original_text[chunk.start_index:chunk.end_index]`. Never rebuild chunk
  text from decoded lengths (that corrupts multi-byte/CJK text).
- **Non-ASCII / multi-byte delimiters** must go through pattern matching, never a
  per-byte split.

When you fix a bug, add a regression test that would have caught it.

## Pull request workflow

1. Fork the repo and create a branch: `git checkout -b my-feature`.
2. Make your change, with tests. Keep commits focused.
3. Run the checks above (`cargo fmt --check`, `cargo clippy -- -D warnings`, `pytest`).
4. Push and open a PR against `main` with a clear description of *what* and *why*.
5. CI runs the build + test matrix on your PR; address any failures.

## Code of conduct

Be kind and constructive. We want blazechunk to be a welcoming project for contributors
of all backgrounds and experience levels — assume good faith, keep discussion technical,
and help newcomers.

## License

By contributing, you agree that your contributions are dual-licensed under the
[MIT](LICENSE-MIT) and [Apache-2.0](LICENSE-APACHE) licenses, matching the project.
