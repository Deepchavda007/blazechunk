"""The one place blazechunk touches :mod:`anydoc`.

Everything upstream of this module deals in ``bytes``; everything downstream
deals in Markdown and :class:`~blazechunk.loaders._segments.Segment`. No anydoc
type crosses either edge — not a ``Document``, not a ``Format``, not an
exception. When anydoc changes, this file changes and nothing else does.

Why Markdown and not the document model
---------------------------------------
anydoc exposes a block tree via ``to_document()``, and the obvious design is to
walk it. It is the wrong choice here, for one measured reason: **PDF has no
document-model form.** ``to_document()`` on a PDF raises ``UnsupportedError``
("PDF converts directly to Markdown"), because the PDF path renders Markdown
directly. Building on the model would make PDF — the format most people arrive
with — the one format on a second-class path.

The model turns out to carry almost nothing extra that chunking needs. Code
blocks report ``lang=None`` in the model *and* in the Markdown; headings, tables,
lists, quotes and fences all survive serialization. Only ``Table.kind``
(data vs layout), multi-row headers, and merged-cell spans are model-only, and
none of them change where a chunk boundary belongs.

So the canonical text is the Markdown, for every format alike. That is not the
usual "just parse the Markdown" compromise: anydoc renders every format through
a *single* serializer, so this is a known machine-generated dialect rather than
arbitrary Markdown of unknown provenance.
"""

from __future__ import annotations

import os
from pathlib import Path

from blazechunk.loaders._errors import (
    DocumentResourceLimit,
    EncryptedDocument,
    MalformedDocument,
    ScannedDocumentError,
    UnsupportedDocument,
)

__all__ = ["convert", "resolve_format", "SUPPORTED_FORMATS", "PASSTHROUGH_FORMATS"]

#: Formats anydoc converts. Mirrors its ``Format`` literal.
SUPPORTED_FORMATS = frozenset(
    {"doc", "docx", "odt", "pdf", "ppt", "pptx", "rtf", "epub", "xlsx", "ods", "odp", "csv"}
)

#: Formats that are *already* Markdown (or close enough to treat as such) and so
#: skip conversion entirely. Handling these here means a pipeline mixing ``.md``
#: notes with ``.pdf`` reports takes one code path, and that the anydoc extra is
#: needed only by the documents that actually require it.
PASSTHROUGH_FORMATS = frozenset({"md", "markdown", "txt", "text"})

_EXTENSION_ALIASES = {
    ".md": "md",
    ".markdown": "md",
    ".mdown": "md",
    ".txt": "txt",
    ".text": "txt",
}

_INSTALL_HINT = (
    "Document loading needs the anydoc extra, which is not installed.\n"
    '    pip install "blazechunk[anydoc]"\n'
    "Markdown and plain-text input work without it."
)


def _anydoc():  # type: ignore[no-untyped-def]
    """Import anydoc, turning its absence into an actionable message."""
    try:
        import anydoc
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(_INSTALL_HINT) from exc
    return anydoc


def resolve_format(
    data: bytes,
    *,
    format: str | None = None,
    filename: str | os.PathLike[str] | None = None,
) -> str:
    """Decide which format ``data`` is, without converting it.

    Resolution runs cheapest-and-most-certain first:

    1. an explicit ``format`` argument always wins;
    2. a Markdown or text *extension*, which needs no anydoc at all;
    3. anydoc's content sniffing, which reads the signature each container
       specification designates;
    4. the file extension, which is the only way to name the signature-less
       formats (CSV).

    Args:
        data: The document bytes.
        format: An explicit format name, or ``None`` to detect.
        filename: The originating path, used for the extension fallback.

    Returns:
        A lowercase format name.

    Raises:
        ValueError: If ``format`` names something unsupported.
        UnsupportedDocument: If detection found nothing and no extension helped.
    """
    if format is not None:
        name = format.lower().lstrip(".")
        name = {"markdown": "md", "text": "txt"}.get(name, name)
        if name not in SUPPORTED_FORMATS and name not in PASSTHROUGH_FORMATS:
            supported = ", ".join(sorted(SUPPORTED_FORMATS | {"md", "txt"}))
            raise ValueError(f"unsupported format {format!r}; expected one of: {supported}")
        return name

    suffix = Path(os.fspath(filename)).suffix.lower() if filename is not None else ""
    if suffix in _EXTENSION_ALIASES:
        return _EXTENSION_ALIASES[suffix]

    anydoc = _anydoc()
    detected = anydoc.format_from_bytes(data)
    if detected is not None:
        return str(detected)

    if suffix:
        from_extension = anydoc.format_from_extension(suffix)
        if from_extension is not None:
            return str(from_extension)

    raise UnsupportedDocument(
        "could not determine the document format from its content"
        + (f" or from the extension {suffix!r}" if suffix else "")
        + ". Pass format=... explicitly (signature-less formats such as CSV "
        "always need it)."
    )


def convert(
    data: bytes,
    *,
    format: str | None = None,
    filename: str | os.PathLike[str] | None = None,
) -> tuple[str, str]:
    """Convert ``data`` to canonical Markdown.

    Args:
        data: The document bytes.
        format: An explicit format name, or ``None`` to detect one.
        filename: The originating path, used for extension-based detection.

    Returns:
        ``(markdown, format)`` — the canonical text and the format it came from.

    Raises:
        ScannedDocumentError: A PDF with no text layer.
        UnsupportedDocument: An unknown format, or one with no extractable text.
        MalformedDocument: Structurally unusable, or missing a required part.
        EncryptedDocument: Password-protected.
        DocumentResourceLimit: A parser safety limit was crossed.
    """
    resolved = resolve_format(data, format=format, filename=filename)

    if resolved in PASSTHROUGH_FORMATS:
        return data.decode("utf-8", errors="replace"), resolved

    anydoc = _anydoc()
    try:
        markdown = anydoc.to_markdown_bytes(data, resolved)
    except anydoc.UnsupportedError as exc:
        # On a PDF this means no text layer at all. The file opens fine in a
        # reader, so the generic message would send people hunting for a bug.
        if resolved == "pdf":
            raise ScannedDocumentError(
                f"no text could be extracted from this PDF ({exc}). It is almost "
                "certainly a scan or an image-only export; anydoc does not do OCR. "
                "Run an OCR step first and pass the result in as text or Markdown."
            ) from exc
        raise UnsupportedDocument(str(exc)) from exc
    except anydoc.EncryptedError as exc:
        raise EncryptedDocument(str(exc)) from exc
    except anydoc.ResourceLimitError as exc:
        raise DocumentResourceLimit(str(exc), limit=getattr(exc, "limit", None)) from exc
    except anydoc.MissingPartError as exc:
        # Collapsed into MalformedDocument: a caller can do nothing different
        # about a missing part than about a corrupt one, and `part` survives.
        raise MalformedDocument(str(exc), part=getattr(exc, "part", None)) from exc
    except anydoc.MalformedError as exc:
        raise MalformedDocument(str(exc), part=getattr(exc, "part", None)) from exc

    return markdown, resolved
