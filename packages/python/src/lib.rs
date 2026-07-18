#[cfg(feature = "hf-tokenizer")]
use chunk::HfTokenCounter;
use chunk::{
    ByteCounter, CharCounter, Chunk as RsChunk, ChunkError, CodeChunker as RsCode, Language,
    Overlap as RsOverlap, RecursiveChunker as RsRecursive, RecursiveLevel, RecursiveRules,
    RowCounter, SentenceChunker as RsSentence, TableChunker as RsTable, TokenChunker as RsToken,
    TokenCounter, WordCounter,
};
use chunk::{
    DEFAULT_DELIMITERS, DEFAULT_TARGET_SIZE, IncludeDelim, OwnedChunker,
    PatternSplitter as RustPatternSplitter, find_merge_indices as rust_find_merge_indices,
    merge_splits as rust_merge_splits, split_at_delimiters, split_at_patterns,
};
#[cfg(feature = "numpy-support")]
use chunk::{
    filter_split_indices as rust_filter_split_indices,
    find_local_minima_interpolated as rust_find_local_minima, savgol_filter as rust_savgol_filter,
    windowed_cross_similarity as rust_windowed_cross_similarity,
};
#[cfg(feature = "numpy-support")]
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyString};

/// Extract bytes from either bytes or str Python object.
fn extract_bytes(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    if obj.is_instance_of::<PyBytes>() {
        Ok(obj.extract::<Vec<u8>>()?)
    } else if obj.is_instance_of::<PyString>() {
        let s: String = obj.extract()?;
        Ok(s.into_bytes())
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "expected bytes or str",
        ))
    }
}

/// Chunker splits text at delimiter boundaries.
///
/// Example with single-byte delimiters:
///     >>> from blazechunk import Chunker
///     >>> text = b"Hello. World. Test."
///     >>> for chunk in Chunker(text, size=10, delimiters=b"."):
///     ...     print(chunk)
///
/// Example with multi-byte pattern (e.g., metaspace for SentencePiece):
///     >>> text = "Hello▁World▁Test"
///     >>> metaspace = "▁"
///     >>> for chunk in Chunker(text, size=15, pattern=metaspace, prefix=True):
///     ...     print(chunk)
///
/// Example with consecutive pattern handling:
///     >>> text = "word   next"  # Three spaces
///     >>> for chunk in Chunker(text, pattern=" ", consecutive=True):
///     ...     print(chunk)  # Splits at START of "   ", not middle
///
/// Also accepts str (encoded as UTF-8):
///     >>> text = "Hello. World. Test."
///     >>> for chunk in Chunker(text, size=10, delimiters="."):
///     ...     print(chunk)
#[pyclass]
pub struct Chunker {
    inner: OwnedChunker,
}

#[pymethods]
impl Chunker {
    #[new]
    #[pyo3(signature = (text, size=DEFAULT_TARGET_SIZE, delimiters=None, pattern=None, patterns=None, prefix=false, consecutive=false, forward_fallback=false))]
    fn new(
        text: &Bound<'_, PyAny>,
        size: usize,
        delimiters: Option<&Bound<'_, PyAny>>,
        pattern: Option<&Bound<'_, PyAny>>,
        patterns: Option<Vec<Bound<'_, PyAny>>>,
        prefix: bool,
        consecutive: bool,
        forward_fallback: bool,
    ) -> PyResult<Self> {
        let text_bytes = extract_bytes(text)?;

        let mut inner = OwnedChunker::new(text_bytes).size(size);

        // Pattern (singular) takes precedence over delimiters if both specified
        if let Some(p) = pattern {
            let pattern_bytes = extract_bytes(p)?;
            inner = inner.pattern(pattern_bytes);
        } else {
            let delims = match delimiters {
                Some(d) => extract_bytes(d)?,
                None => DEFAULT_DELIMITERS.to_vec(),
            };
            inner = inner.delimiters(delims);
        }

        // Patterns (plural) is composable with delimiters — applied after
        if let Some(pats) = patterns {
            let pattern_strings: Vec<String> = pats
                .iter()
                .map(|p| {
                    extract_bytes(p).and_then(|b| {
                        String::from_utf8(b).map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                                "patterns must be valid UTF-8: {}",
                                e
                            ))
                        })
                    })
                })
                .collect::<PyResult<Vec<String>>>()?;
            let pattern_refs: Vec<&str> = pattern_strings.iter().map(|s| s.as_str()).collect();
            inner = inner.patterns(&pattern_refs);
        }

        if prefix {
            inner = inner.prefix();
        }
        if consecutive {
            inner = inner.consecutive();
        }
        if forward_fallback {
            inner = inner.forward_fallback();
        }

        Ok(Self { inner })
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<Py<PyBytes>> {
        slf.inner
            .next_chunk()
            .map(|chunk| PyBytes::new(slf.py(), &chunk).unbind())
    }

    /// Reset the chunker to iterate from the beginning.
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Collect all chunk offsets as a list of (start, end) tuples.
    /// This is faster than iterating as it makes a single Rust call.
    fn collect_offsets(&mut self) -> Vec<(usize, usize)> {
        self.inner.collect_offsets()
    }
}

/// Fast chunking function that returns offsets in a single call.
/// Use this with slicing for maximum performance.
///
/// Example with single-byte delimiters:
///     >>> text = b"Hello. World. Test."
///     >>> offsets = chunk_offsets(text, size=10, delimiters=b".")
///     >>> chunks = [text[start:end] for start, end in offsets]
///
/// Example with multi-byte pattern:
///     >>> text = "Hello▁World▁Test".encode()
///     >>> offsets = chunk_offsets(text, size=15, pattern="▁", prefix=True)
///     >>> chunks = [text[start:end] for start, end in offsets]
#[pyfunction]
#[pyo3(signature = (text, size=DEFAULT_TARGET_SIZE, delimiters=None, pattern=None, patterns=None, prefix=false, consecutive=false, forward_fallback=false))]
fn chunk_offsets(
    text: &Bound<'_, PyAny>,
    size: usize,
    delimiters: Option<&Bound<'_, PyAny>>,
    pattern: Option<&Bound<'_, PyAny>>,
    patterns: Option<Vec<Bound<'_, PyAny>>>,
    prefix: bool,
    consecutive: bool,
    forward_fallback: bool,
) -> PyResult<Vec<(usize, usize)>> {
    let text_bytes = extract_bytes(text)?;

    let mut chunker = OwnedChunker::new(text_bytes).size(size);

    // Pattern (singular) takes precedence over delimiters if both specified
    if let Some(p) = pattern {
        let pattern_bytes = extract_bytes(p)?;
        chunker = chunker.pattern(pattern_bytes);
    } else {
        let delims = match delimiters {
            Some(d) => extract_bytes(d)?,
            None => DEFAULT_DELIMITERS.to_vec(),
        };
        chunker = chunker.delimiters(delims);
    }

    // Patterns (plural) is composable with delimiters — applied after
    if let Some(pats) = patterns {
        let pattern_strings: Vec<String> = pats
            .iter()
            .map(|p| {
                extract_bytes(p).and_then(|b| {
                    String::from_utf8(b).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "patterns must be valid UTF-8: {}",
                            e
                        ))
                    })
                })
            })
            .collect::<PyResult<Vec<String>>>()?;
        let pattern_refs: Vec<&str> = pattern_strings.iter().map(|s| s.as_str()).collect();
        chunker = chunker.patterns(&pattern_refs);
    }

    if prefix {
        chunker = chunker.prefix();
    }
    if consecutive {
        chunker = chunker.consecutive();
    }
    if forward_fallback {
        chunker = chunker.forward_fallback();
    }

    Ok(chunker.collect_offsets())
}

/// Split text at every delimiter occurrence, returning offsets.
///
/// This is the Rust equivalent of Cython's `split_text` function.
/// Unlike chunk_offsets() which creates size-based chunks, this splits at
/// **every** delimiter occurrence.
///
/// Args:
///     text: bytes or str to split
///     delimiters: bytes or str of delimiter characters (default: "\\n.?")
///     include_delim: Where to attach delimiter - "prev" (default), "next", or "none"
///     min_chars: Minimum characters per segment (default: 0). Shorter segments are merged.
///
/// Returns:
///     List of (start, end) byte offsets for each segment.
///
/// Example:
///     >>> text = b"Hello. World. Test."
///     >>> offsets = split_offsets(text, delimiters=b".")
///     >>> segments = [text[start:end] for start, end in offsets]
///     >>> # ["Hello.", " World.", " Test."]
///
/// Example with include_delim="next":
///     >>> offsets = split_offsets(text, delimiters=b".", include_delim="next")
///     >>> segments = [text[start:end] for start, end in offsets]
///     >>> # ["Hello", ". World", ". Test", "."]
#[pyfunction]
#[pyo3(signature = (text, delimiters=None, include_delim="prev", min_chars=0))]
fn split_offsets(
    text: &Bound<'_, PyAny>,
    delimiters: Option<&Bound<'_, PyAny>>,
    include_delim: &str,
    min_chars: usize,
) -> PyResult<Vec<(usize, usize)>> {
    let text_bytes = extract_bytes(text)?;

    let delims = match delimiters {
        Some(d) => extract_bytes(d)?,
        None => DEFAULT_DELIMITERS.to_vec(),
    };

    let include = match include_delim {
        "prev" => IncludeDelim::Prev,
        "next" => IncludeDelim::Next,
        "none" => IncludeDelim::None,
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "include_delim must be 'prev', 'next', or 'none'",
            ));
        }
    };

    Ok(split_at_delimiters(
        &text_bytes,
        &delims,
        include,
        min_chars,
    ))
}

/// Split text at every occurrence of multi-byte patterns, returning offsets.
///
/// Unlike split_offsets() which only handles single-byte delimiters,
/// this function supports multi-byte patterns like ". ", "? ", "\n\n", etc.
/// using the Aho-Corasick algorithm for efficient matching.
///
/// Args:
///     text: bytes or str to split
///     patterns: List of bytes or str patterns to split on (e.g., [". ", "? ", "! "])
///     include_delim: Where to attach pattern - "prev" (default), "next", or "none"
///     min_chars: Minimum characters per segment (default: 0). Shorter segments are merged.
///
/// Returns:
///     List of (start, end) byte offsets for each segment.
///
/// Example:
///     >>> text = b"Hello. World? Test!"
///     >>> offsets = split_pattern_offsets(text, patterns=[b". ", b"? ", b"! "])
///     >>> segments = [text[start:end] for start, end in offsets]
///     >>> # [b"Hello. ", b"World? ", b"Test!"]
///
/// Example with paragraph splitting:
///     >>> text = b"Para 1\n\nPara 2\n\nPara 3"
///     >>> offsets = split_pattern_offsets(text, patterns=[b"\n\n"])
///     >>> segments = [text[start:end] for start, end in offsets]
///     >>> # [b"Para 1\n\n", b"Para 2\n\n", b"Para 3"]
#[pyfunction]
#[pyo3(signature = (text, patterns, include_delim="prev", min_chars=0))]
fn split_pattern_offsets(
    text: &Bound<'_, PyAny>,
    patterns: Vec<Bound<'_, PyAny>>,
    include_delim: &str,
    min_chars: usize,
) -> PyResult<Vec<(usize, usize)>> {
    let text_bytes = extract_bytes(text)?;

    // Convert Python patterns to Vec<Vec<u8>>
    let pattern_bytes: Vec<Vec<u8>> = patterns
        .iter()
        .map(|p| extract_bytes(p))
        .collect::<PyResult<Vec<Vec<u8>>>>()?;

    // Convert to slice of slices for the Rust function
    let pattern_slices: Vec<&[u8]> = pattern_bytes.iter().map(|p| p.as_slice()).collect();

    let include = match include_delim {
        "prev" => IncludeDelim::Prev,
        "next" => IncludeDelim::Next,
        "none" => IncludeDelim::None,
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "include_delim must be 'prev', 'next', or 'none'",
            ));
        }
    };

    Ok(split_at_patterns(
        &text_bytes,
        &pattern_slices,
        include,
        min_chars,
    ))
}

/// A compiled multi-pattern splitter for efficient repeated splitting.
///
/// Unlike split_pattern_offsets() which rebuilds the automaton on each call,
/// PatternSplitter compiles once and reuses. This is ~25x faster when splitting
/// multiple texts with the same patterns.
///
/// Example:
///     >>> from blazechunk import PatternSplitter
///     >>> splitter = PatternSplitter([b". ", b"? ", b"! "])
///     >>> offsets1 = splitter.split(b"Hello. World?")
///     >>> offsets2 = splitter.split(b"Another. Text!")
#[pyclass]
pub struct PatternSplitter {
    inner: RustPatternSplitter,
}

#[pymethods]
impl PatternSplitter {
    #[new]
    fn new(patterns: Vec<Bound<'_, PyAny>>) -> PyResult<Self> {
        let pattern_bytes: Vec<Vec<u8>> = patterns
            .iter()
            .map(|p| extract_bytes(p))
            .collect::<PyResult<Vec<Vec<u8>>>>()?;

        let pattern_slices: Vec<&[u8]> = pattern_bytes.iter().map(|p| p.as_slice()).collect();
        let inner = RustPatternSplitter::new(&pattern_slices);

        Ok(Self { inner })
    }

    /// Split text using the compiled patterns.
    ///
    /// Args:
    ///     text: bytes or str to split
    ///     include_delim: Where to attach pattern - "prev" (default), "next", or "none"
    ///     min_chars: Minimum characters per segment (default: 0)
    ///
    /// Returns:
    ///     List of (start, end) byte offsets for each segment.
    #[pyo3(signature = (text, include_delim="prev", min_chars=0))]
    fn split(
        &self,
        text: &Bound<'_, PyAny>,
        include_delim: &str,
        min_chars: usize,
    ) -> PyResult<Vec<(usize, usize)>> {
        let text_bytes = extract_bytes(text)?;

        let include = match include_delim {
            "prev" => IncludeDelim::Prev,
            "next" => IncludeDelim::Next,
            "none" => IncludeDelim::None,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "include_delim must be 'prev', 'next', or 'none'",
                ));
            }
        };

        Ok(self.inner.split(&text_bytes, include, min_chars))
    }
}

/// Result of merge_splits operation.
///
/// Attributes:
///     merged: List of merged text strings.
///     token_counts: List of token counts for each merged chunk.
#[pyclass]
#[derive(Clone)]
pub struct MergeResult {
    #[pyo3(get)]
    merged: Vec<String>,
    #[pyo3(get)]
    token_counts: Vec<usize>,
}

#[pymethods]
impl MergeResult {
    fn __repr__(&self) -> String {
        format!(
            "MergeResult(merged=[...{} items], token_counts={:?})",
            self.merged.len(),
            self.token_counts
        )
    }

    fn __len__(&self) -> usize {
        self.merged.len()
    }
}

/// Find merge indices for combining segments within token limits.
///
/// Returns indices marking where to split segments into chunks that
/// respect the token budget. Use this when you only need indices,
/// not the actual merged text.
///
/// Args:
///     token_counts: List of token counts for each segment.
///     chunk_size: Maximum tokens per merged chunk.
///
/// Returns:
///     List of end indices (exclusive) for each chunk.
///
/// Example:
///     >>> from blazechunk import find_merge_indices
///     >>> token_counts = [1, 1, 1, 1, 1, 1, 1]
///     >>> indices = find_merge_indices(token_counts, chunk_size=3)
///     >>> indices  # [3, 6, 7]
#[pyfunction]
#[pyo3(signature = (token_counts, chunk_size))]
fn find_merge_indices(token_counts: Vec<usize>, chunk_size: usize) -> Vec<usize> {
    rust_find_merge_indices(&token_counts, chunk_size)
}

/// Merge text segments based on token counts, respecting chunk size limits.
///
/// This is the Rust equivalent of Chonkie's Cython `_merge_splits` function.
/// Performs string concatenation in Rust for optimal performance.
///
/// Args:
///     splits: List of text segments to merge.
///     token_counts: List of token counts for each segment.
///     chunk_size: Maximum tokens per merged chunk.
///
/// Returns:
///     MergeResult with:
///     - merged: List of merged text strings
///     - token_counts: Token count for each merged chunk
///
/// Example:
///     >>> from blazechunk import merge_splits
///     >>> splits = ["Hello", "world", "!", "How", "are", "you"]
///     >>> token_counts = [1, 1, 1, 1, 1, 1]
///     >>> result = merge_splits(splits, token_counts, chunk_size=3)
///     >>> result.merged  # ["Helloworld!", "Howareyou"]
///     >>> result.token_counts  # [3, 3]
#[pyfunction]
#[pyo3(signature = (splits, token_counts, chunk_size))]
fn merge_splits(splits: Vec<String>, token_counts: Vec<usize>, chunk_size: usize) -> MergeResult {
    let split_refs: Vec<&str> = splits.iter().map(|s| s.as_str()).collect();
    let result = rust_merge_splits(&split_refs, &token_counts, chunk_size);
    MergeResult {
        merged: result.merged,
        token_counts: result.token_counts,
    }
}

// =============================================================================
// Savitzky-Golay Filter Functions (NumPy-optimized)
// =============================================================================

#[cfg(feature = "numpy-support")]
/// Apply Savitzky-Golay filter to data.
///
/// This filter is used for smoothing signals and computing derivatives.
/// It fits a polynomial to a sliding window of data points.
///
/// Args:
///     data: Input signal as numpy array of floats.
///     window_length: Filter window length (must be odd and > poly_order). Default: 5.
///     poly_order: Polynomial order for fitting. Default: 3.
///     deriv: Derivative order (0=smoothing, 1=first, 2=second). Default: 0.
///
/// Returns:
///     Filtered data as numpy array.
///
/// Example:
///     >>> import numpy as np
///     >>> from blazechunk import savgol_filter
///     >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
///     >>> smoothed = savgol_filter(data, window_length=5, poly_order=2)
#[pyfunction]
#[pyo3(signature = (data, window_length=5, poly_order=3, deriv=0))]
fn savgol_filter<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    window_length: usize,
    poly_order: usize,
    deriv: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data_slice = data.as_slice()?;
    let result =
        rust_savgol_filter(data_slice, window_length, poly_order, deriv).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Invalid parameters: window_length must be odd and > poly_order",
            )
        })?;
    Ok(PyArray1::from_vec(py, result))
}

#[cfg(feature = "numpy-support")]
/// Find local minima with sub-sample accuracy using Savitzky-Golay derivatives.
///
/// A point is considered a minimum if its first derivative is near zero
/// and its second derivative is positive (concave up).
///
/// Args:
///     data: Input signal as numpy array.
///     window_size: Savitzky-Golay window size (must be odd). Default: 11.
///     poly_order: Polynomial order. Default: 2.
///     tolerance: Tolerance for first derivative near zero. Default: 0.2.
///
/// Returns:
///     Tuple of (indices, values) as numpy arrays where minima were found.
///
/// Example:
///     >>> import numpy as np
///     >>> from blazechunk import find_local_minima_interpolated
///     >>> data = np.array([x**2 for x in range(-10, 11)], dtype=np.float64)
///     >>> indices, values = find_local_minima_interpolated(data)
#[pyfunction]
#[pyo3(signature = (data, window_size=11, poly_order=2, tolerance=0.2))]
fn find_local_minima_interpolated<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    window_size: usize,
    poly_order: usize,
    tolerance: f64,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)> {
    let data_slice = data.as_slice()?;
    let result = rust_find_local_minima(data_slice, window_size, poly_order, tolerance)
        .ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Invalid parameters: window_size must be odd and > poly_order",
            )
        })?;
    let indices: Vec<i64> = result.indices.into_iter().map(|i| i as i64).collect();
    Ok((
        PyArray1::from_vec(py, indices),
        PyArray1::from_vec(py, result.values),
    ))
}

#[cfg(feature = "numpy-support")]
/// Compute windowed cross-similarity for semantic chunking.
///
/// For each position, computes the average cosine similarity between
/// consecutive embedding vectors within a sliding window.
///
/// Args:
///     embeddings: 2D numpy array of embeddings (n_sentences x embedding_dim).
///     window_size: Size of sliding window (must be odd and >= 3). Default: 3.
///
/// Returns:
///     Numpy array of average similarities (length n_sentences - 1).
///
/// Example:
///     >>> import numpy as np
///     >>> from blazechunk import windowed_cross_similarity
///     >>> embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
///     >>> similarities = windowed_cross_similarity(embeddings, window_size=3)
#[pyfunction]
#[pyo3(signature = (embeddings, window_size=3))]
fn windowed_cross_similarity<'py>(
    py: Python<'py>,
    embeddings: PyReadonlyArray2<'py, f64>,
    window_size: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let shape = embeddings.shape();
    let n = shape[0];
    let d = shape[1];

    if n == 0 {
        return Ok(PyArray1::from_vec(py, vec![]));
    }

    // Get flattened view
    let flat = embeddings.as_slice()?;

    let result = rust_windowed_cross_similarity(flat, n, d, window_size).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Invalid parameters: window_size must be odd and >= 3, and need at least 2 embeddings",
        )
    })?;
    Ok(PyArray1::from_vec(py, result))
}

#[cfg(feature = "numpy-support")]
/// Filter split indices by percentile threshold and minimum distance.
///
/// This is used in semantic chunking to select optimal split points
/// from candidate minima, ensuring splits are spread out and have
/// low enough similarity values.
///
/// Args:
///     indices: Candidate split indices as numpy array.
///     values: Similarity values at those indices as numpy array.
///     threshold: Percentile threshold (0.0-1.0). Default: 0.5.
///     min_distance: Minimum distance between selected splits. Default: 2.
///
/// Returns:
///     Tuple of (filtered_indices, filtered_values) as numpy arrays.
///
/// Example:
///     >>> import numpy as np
///     >>> from blazechunk import filter_split_indices
///     >>> indices = np.array([0, 5, 8, 15, 20])
///     >>> values = np.array([0.1, 0.3, 0.2, 0.5, 0.4])
///     >>> filtered_idx, filtered_val = filter_split_indices(indices, values, threshold=0.5)
#[pyfunction]
#[pyo3(signature = (indices, values, threshold=0.5, min_distance=2))]
fn filter_split_indices<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
    threshold: f64,
    min_distance: usize,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)> {
    let indices_slice = indices.as_slice()?;
    let values_slice = values.as_slice()?;

    // Convert i64 to usize for Rust function
    let indices_usize: Vec<usize> = indices_slice.iter().map(|&i| i as usize).collect();

    let result = rust_filter_split_indices(&indices_usize, values_slice, threshold, min_distance);

    // Convert back to i64 for numpy
    let result_indices: Vec<i64> = result.indices.into_iter().map(|i| i as i64).collect();
    Ok((
        PyArray1::from_vec(py, result_indices),
        PyArray1::from_vec(py, result.values),
    ))
}

// =============================================================================
// High-level chunkers (Recursive / Sentence / Token / Table / Code)
// =============================================================================

/// Build a token counter from a tokenizer name.
/// Resolved tokenizer for the Python chunkers: a built-in name, or (with the
/// `hf-tokenizer` feature) a locally loaded HuggingFace `tokenizer.json`.
/// Loaded once at construction so HF models aren't re-parsed on every `chunk()`.
enum BoundCounter {
    Char,
    Word,
    Byte,
    Row,
    #[cfg(feature = "hf-tokenizer")]
    Hf(HfTokenCounter),
}

impl BoundCounter {
    fn resolve(name: &str) -> PyResult<(String, Self)> {
        let label = name.to_string();
        let counter = match name.to_ascii_lowercase().as_str() {
            "character" | "char" => Self::Char,
            "word" => Self::Word,
            "byte" => Self::Byte,
            "row" => Self::Row,
            _ => {
                #[cfg(feature = "hf-tokenizer")]
                if looks_like_tokenizer_path(name) {
                    let hf = HfTokenCounter::from_file(name)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
                    return Ok((label, Self::Hf(hf)));
                }
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "unknown tokenizer '{name}' (expected 'character', 'word', 'byte', 'row'{})",
                    if cfg!(feature = "hf-tokenizer") {
                        ", or a path to a local tokenizer.json"
                    } else {
                        ""
                    }
                )));
            }
        };
        Ok((label, counter))
    }

    fn as_dyn(&self) -> &dyn TokenCounter {
        match self {
            Self::Char => &CharCounter,
            Self::Word => &WordCounter,
            Self::Byte => &ByteCounter,
            Self::Row => &RowCounter,
            #[cfg(feature = "hf-tokenizer")]
            Self::Hf(hf) => hf,
        }
    }
}

/// True when `name` looks like a filesystem path to a tokenizer rather than a
/// built-in counter name. Checked against the original (non-lowercased) string
/// so case-sensitive paths keep working.
fn looks_like_tokenizer_path(name: &str) -> bool {
    let path = std::path::Path::new(name);
    name.ends_with(".json") || path.exists()
}

fn parse_include_delim(s: &str) -> PyResult<IncludeDelim> {
    match s.to_ascii_lowercase().as_str() {
        "prev" => Ok(IncludeDelim::Prev),
        "next" => Ok(IncludeDelim::Next),
        "none" => Ok(IncludeDelim::None),
        other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "include_delim must be 'prev', 'next', or 'none', got '{other}'"
        ))),
    }
}

/// Map a core [`ChunkError`] to a Python `ValueError`. The Rust core is the
/// single source of truth for configuration validity; the binding just surfaces it.
fn chunk_err_to_py(e: ChunkError) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
}

/// Build a [`RecursiveRules`] from a Python description of the delimiter hierarchy.
///
/// `levels` is a list of dicts, one per level, evaluated top to bottom:
///   * `{"delimiters": ["\n\n", "\n"], "include_delim": "prev"}` — split at delimiters
///   * `{"type": "whitespace", "include_delim": "prev"}` — split on ASCII spaces
///   * `{"type": "token"}` — terminal hard-split into token groups
///
/// `include_delim` is optional and defaults to `"prev"`. `None` yields the default
/// five-level hierarchy.
fn parse_recursive_rules(levels: Option<Vec<Bound<'_, PyAny>>>) -> PyResult<RecursiveRules> {
    let Some(levels) = levels else {
        return Ok(RecursiveRules::default());
    };
    if levels.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "rules must contain at least one level",
        ));
    }
    let mut parsed = Vec::with_capacity(levels.len());
    for level in &levels {
        let dict = level.cast::<PyDict>().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "each rules level must be a dict, e.g. {'delimiters': ['\\n\\n']} or {'type': 'token'}",
            )
        })?;
        let include_delim = match dict.get_item("include_delim")? {
            Some(v) => parse_include_delim(&v.extract::<String>()?)?,
            None => IncludeDelim::Prev,
        };
        let level_type: Option<String> = match dict.get_item("type")? {
            Some(v) => Some(v.extract()?),
            None => None,
        };
        if let Some(delims) = dict.get_item("delimiters")? {
            let delimiters: Vec<String> = delims.extract().map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "'delimiters' must be a list of strings",
                )
            })?;
            parsed.push(RecursiveLevel::Delimiters {
                delimiters,
                include_delim,
            });
        } else if level_type.as_deref() == Some("whitespace") {
            parsed.push(RecursiveLevel::Whitespace { include_delim });
        } else if level_type.as_deref() == Some("token") {
            parsed.push(RecursiveLevel::Token);
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "each rules level needs a 'delimiters' list, or type 'whitespace'/'token'",
            ));
        }
    }
    Ok(RecursiveRules { levels: parsed })
}

/// A chunk of text with byte-offset indices into the original input.
#[pyclass(name = "Chunk")]
#[derive(Clone)]
pub struct PyChunk {
    #[pyo3(get)]
    text: String,
    #[pyo3(get)]
    start_index: usize,
    #[pyo3(get)]
    end_index: usize,
    #[pyo3(get)]
    token_count: usize,
}

#[pymethods]
impl PyChunk {
    fn __len__(&self) -> usize {
        self.text.chars().count()
    }

    fn __repr__(&self) -> String {
        let preview: String = self.text.chars().take(40).collect();
        let ellipsis = if self.text.chars().count() > 40 {
            "…"
        } else {
            ""
        };
        format!(
            "Chunk(text={:?}{}, start_index={}, end_index={}, token_count={})",
            preview, ellipsis, self.start_index, self.end_index, self.token_count
        )
    }
}

fn to_py_chunks(text: &str, chunks: Vec<RsChunk>) -> Vec<PyChunk> {
    chunks
        .into_iter()
        .map(|c| PyChunk {
            text: text[c.start..c.end].to_string(),
            start_index: c.start,
            end_index: c.end,
            token_count: c.token_count,
        })
        .collect()
}

/// Recursive chunking through a hierarchy of delimiter levels.
#[pyclass(name = "RecursiveChunker")]
pub struct PyRecursiveChunker {
    chunk_size: usize,
    min_characters_per_chunk: usize,
    tokenizer: String,
    counter: BoundCounter,
    rules: RecursiveRules,
}

#[pymethods]
impl PyRecursiveChunker {
    #[new]
    #[pyo3(signature = (tokenizer=None, chunk_size=2048, min_characters_per_chunk=24, rules=None))]
    fn new(
        tokenizer: Option<String>,
        chunk_size: usize,
        min_characters_per_chunk: usize,
        rules: Option<Vec<Bound<'_, PyAny>>>,
    ) -> PyResult<Self> {
        let name = tokenizer.unwrap_or_else(|| "character".to_string());
        let (tokenizer, counter) = BoundCounter::resolve(&name)?;
        let rules = parse_recursive_rules(rules)?;
        // The Rust core owns validation — build and validate once here so bad
        // config raises at construction, then reuse the same rules per chunk.
        RsRecursive::new()
            .chunk_size(chunk_size)
            .min_characters_per_chunk(min_characters_per_chunk)
            .rules(rules.clone())
            .validate()
            .map_err(chunk_err_to_py)?;
        Ok(Self {
            chunk_size,
            min_characters_per_chunk,
            tokenizer,
            counter,
            rules,
        })
    }

    fn chunk(&self, text: &str) -> PyResult<Vec<PyChunk>> {
        let inner = RsRecursive::new()
            .chunk_size(self.chunk_size)
            .min_characters_per_chunk(self.min_characters_per_chunk)
            .rules(self.rules.clone());
        let chunks = inner
            .chunk(text, self.counter.as_dyn())
            .map_err(chunk_err_to_py)?;
        Ok(to_py_chunks(text, chunks))
    }

    fn __repr__(&self) -> String {
        format!(
            "RecursiveChunker(chunk_size={}, min_characters_per_chunk={}, tokenizer='{}', rules={} levels)",
            self.chunk_size,
            self.min_characters_per_chunk,
            self.tokenizer,
            self.rules.levels.len()
        )
    }
}

/// Sentence-aware chunking with token overlap.
#[pyclass(name = "SentenceChunker")]
pub struct PySentenceChunker {
    chunk_size: usize,
    chunk_overlap: usize,
    min_sentences_per_chunk: usize,
    min_characters_per_sentence: usize,
    delim: Vec<String>,
    include_delim: IncludeDelim,
    tokenizer: String,
    counter: BoundCounter,
}

#[pymethods]
impl PySentenceChunker {
    #[new]
    #[pyo3(signature = (
        tokenizer=None,
        chunk_size=2048,
        chunk_overlap=0,
        min_sentences_per_chunk=1,
        min_characters_per_sentence=12,
        delim=None,
        include_delim=None,
    ))]
    fn new(
        tokenizer: Option<String>,
        chunk_size: usize,
        chunk_overlap: usize,
        min_sentences_per_chunk: usize,
        min_characters_per_sentence: usize,
        delim: Option<Vec<String>>,
        include_delim: Option<String>,
    ) -> PyResult<Self> {
        let name = tokenizer.unwrap_or_else(|| "character".to_string());
        let (tokenizer, counter) = BoundCounter::resolve(&name)?;
        let include_delim = parse_include_delim(&include_delim.unwrap_or_else(|| "prev".into()))?;
        let delim =
            delim.unwrap_or_else(|| vec![". ".into(), "! ".into(), "? ".into(), "\n".into()]);
        // The Rust core owns validation — raise at construction on bad config.
        RsSentence::new()
            .chunk_size(chunk_size)
            .chunk_overlap(chunk_overlap)
            .min_sentences_per_chunk(min_sentences_per_chunk)
            .min_characters_per_sentence(min_characters_per_sentence)
            .delim(delim.clone())
            .include_delim(include_delim)
            .validate()
            .map_err(chunk_err_to_py)?;
        Ok(Self {
            chunk_size,
            chunk_overlap,
            min_sentences_per_chunk,
            min_characters_per_sentence,
            delim,
            include_delim,
            tokenizer,
            counter,
        })
    }

    fn chunk(&self, text: &str) -> PyResult<Vec<PyChunk>> {
        let inner = RsSentence::new()
            .chunk_size(self.chunk_size)
            .chunk_overlap(self.chunk_overlap)
            .min_sentences_per_chunk(self.min_sentences_per_chunk)
            .min_characters_per_sentence(self.min_characters_per_sentence)
            .delim(self.delim.clone())
            .include_delim(self.include_delim);
        let chunks = inner
            .chunk(text, self.counter.as_dyn())
            .map_err(chunk_err_to_py)?;
        Ok(to_py_chunks(text, chunks))
    }

    fn __repr__(&self) -> String {
        format!(
            "SentenceChunker(chunk_size={}, chunk_overlap={}, tokenizer='{}')",
            self.chunk_size, self.chunk_overlap, self.tokenizer
        )
    }
}

/// Fixed-size token chunking with token/fraction overlap.
#[pyclass(name = "TokenChunker")]
pub struct PyTokenChunker {
    chunk_size: usize,
    overlap: RsOverlap,
    tokenizer: String,
    counter: BoundCounter,
}

#[pymethods]
impl PyTokenChunker {
    #[new]
    #[pyo3(signature = (tokenizer=None, chunk_size=2048, chunk_overlap=None))]
    fn new(
        tokenizer: Option<String>,
        chunk_size: usize,
        chunk_overlap: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let name = tokenizer.unwrap_or_else(|| "character".to_string());
        if chunk_size == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "chunk_size must be > 0",
            ));
        }
        let (tokenizer, counter) = BoundCounter::resolve(&name)?;
        let overlap = match chunk_overlap {
            None => RsOverlap::Tokens(0),
            Some(obj) => {
                if obj.is_instance_of::<pyo3::types::PyFloat>() {
                    RsOverlap::Fraction(obj.extract::<f64>()?)
                } else {
                    let n = obj.extract::<usize>()?;
                    if n >= chunk_size {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "chunk_overlap must be < chunk_size",
                        ));
                    }
                    RsOverlap::Tokens(n)
                }
            }
        };
        Ok(Self {
            chunk_size,
            overlap,
            tokenizer,
            counter,
        })
    }

    fn chunk(&self, text: &str) -> PyResult<Vec<PyChunk>> {
        let inner = RsToken::new().chunk_size(self.chunk_size);
        let inner = match self.overlap {
            RsOverlap::Tokens(n) => inner.chunk_overlap_tokens(n),
            RsOverlap::Fraction(f) => inner.chunk_overlap_fraction(f),
        };
        let chunks = inner
            .chunk(text, self.counter.as_dyn())
            .map_err(chunk_err_to_py)?;
        Ok(to_py_chunks(text, chunks))
    }

    fn __repr__(&self) -> String {
        format!(
            "TokenChunker(chunk_size={}, tokenizer='{}')",
            self.chunk_size, self.tokenizer
        )
    }
}

/// Table chunking that re-includes the header in every chunk.
#[pyclass(name = "TableChunker")]
pub struct PyTableChunker {
    chunk_size: usize,
    tokenizer: String,
    counter: BoundCounter,
}

#[pymethods]
impl PyTableChunker {
    #[new]
    #[pyo3(signature = (tokenizer=None, chunk_size=3))]
    fn new(tokenizer: Option<String>, chunk_size: usize) -> PyResult<Self> {
        let name = tokenizer.unwrap_or_else(|| "row".to_string());
        let (tokenizer, counter) = BoundCounter::resolve(&name)?;
        // The Rust core owns validation — raise at construction on bad config.
        RsTable::new()
            .chunk_size(chunk_size)
            .validate()
            .map_err(chunk_err_to_py)?;
        Ok(Self {
            chunk_size,
            tokenizer,
            counter,
        })
    }

    fn chunk(&self, text: &str) -> PyResult<Vec<PyChunk>> {
        let inner = RsTable::new().chunk_size(self.chunk_size);
        let chunks = inner
            .chunk(text, self.counter.as_dyn())
            .map_err(chunk_err_to_py)?;
        Ok(chunks
            .into_iter()
            .map(|c| PyChunk {
                text: c.text,
                start_index: c.start,
                end_index: c.end,
                token_count: c.token_count,
            })
            .collect())
    }

    fn __repr__(&self) -> String {
        format!(
            "TableChunker(chunk_size={}, tokenizer='{}')",
            self.chunk_size, self.tokenizer
        )
    }
}

/// Structure-aware code chunking.
#[pyclass(name = "CodeChunker")]
pub struct PyCodeChunker {
    chunk_size: usize,
    language: String,
    tokenizer: String,
    counter: BoundCounter,
}

#[pymethods]
impl PyCodeChunker {
    #[new]
    #[pyo3(signature = (tokenizer=None, chunk_size=2048, language=None))]
    fn new(
        tokenizer: Option<String>,
        chunk_size: usize,
        language: Option<String>,
    ) -> PyResult<Self> {
        let name = tokenizer.unwrap_or_else(|| "character".to_string());
        if chunk_size == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "chunk_size must be > 0",
            ));
        }
        let (tokenizer, counter) = BoundCounter::resolve(&name)?;
        Ok(Self {
            chunk_size,
            language: language.unwrap_or_else(|| "auto".to_string()),
            tokenizer,
            counter,
        })
    }

    fn chunk(&self, text: &str) -> PyResult<Vec<PyChunk>> {
        let inner = RsCode::new()
            .chunk_size(self.chunk_size)
            .language(Language::from_name(&self.language));
        Ok(to_py_chunks(text, inner.chunk(text, self.counter.as_dyn())))
    }

    fn __repr__(&self) -> String {
        format!(
            "CodeChunker(chunk_size={}, language='{}', tokenizer='{}')",
            self.chunk_size, self.language, self.tokenizer
        )
    }
}

#[pymodule]
fn _chunk(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Chunker>()?;
    m.add_class::<PyChunk>()?;
    m.add_class::<PyRecursiveChunker>()?;
    m.add_class::<PySentenceChunker>()?;
    m.add_class::<PyTokenChunker>()?;
    m.add_class::<PyTableChunker>()?;
    m.add_class::<PyCodeChunker>()?;
    m.add_class::<MergeResult>()?;
    m.add_class::<PatternSplitter>()?;
    m.add_function(wrap_pyfunction!(chunk_offsets, m)?)?;
    m.add_function(wrap_pyfunction!(split_offsets, m)?)?;
    m.add_function(wrap_pyfunction!(split_pattern_offsets, m)?)?;
    m.add_function(wrap_pyfunction!(find_merge_indices, m)?)?;
    m.add_function(wrap_pyfunction!(merge_splits, m)?)?;
    #[cfg(feature = "numpy-support")]
    {
        m.add_function(wrap_pyfunction!(savgol_filter, m)?)?;
        m.add_function(wrap_pyfunction!(find_local_minima_interpolated, m)?)?;
        m.add_function(wrap_pyfunction!(windowed_cross_similarity, m)?)?;
        m.add_function(wrap_pyfunction!(filter_split_indices, m)?)?;
    }
    m.add("DEFAULT_TARGET_SIZE", DEFAULT_TARGET_SIZE)?;
    m.add("DEFAULT_DELIMITERS", DEFAULT_DELIMITERS)?;
    Ok(())
}
