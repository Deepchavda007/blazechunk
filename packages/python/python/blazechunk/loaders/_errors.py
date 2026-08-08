"""Errors raised while loading a document.

These mirror anydoc's failure modes but are blazechunk's own types, so callers
never import :mod:`anydoc` to write an ``except`` clause and code that catches
them keeps working if the conversion backend is ever swapped.

Every failure below subclasses :class:`DocumentError`, so a batch job that wants
to skip unreadable files and keep going needs exactly one handler::

    try:
        result = loader.chunk(path)
    except DocumentError as error:
        skipped.append((path, str(error)))

An unreadable *file* (missing, no permission) raises :class:`OSError`, which is
deliberately left alone: it is a filesystem problem, not a document problem, and
collapsing the two would hide a typo'd path behind a "malformed document"
message.
"""

from __future__ import annotations

__all__ = [
    "DocumentError",
    "UnsupportedDocument",
    "ScannedDocumentError",
    "MalformedDocument",
    "EncryptedDocument",
    "DocumentResourceLimit",
]


class DocumentError(Exception):
    """Base class for every document-loading failure."""


class UnsupportedDocument(DocumentError):
    """The format is unknown, or carries no extractable text."""


class ScannedDocumentError(UnsupportedDocument):
    """A PDF that holds no text layer — almost always a scan or an export of
    images.

    This is by far the most common real-world failure, and the least obvious:
    the file opens perfectly in any PDF reader, so a bare "unsupported format"
    reads as a bug in the library rather than a property of the file. Reading it
    needs OCR, which anydoc does not do; run an OCR step upstream and pass the
    result in.

    It subclasses :class:`UnsupportedDocument`, so handlers written before this
    distinction existed still catch it.
    """


class MalformedDocument(DocumentError):
    """The document is structurally unusable, or a part required for any
    meaningful output is missing.

    Attributes:
        part: The package part or stream at fault, when the backend named one.
    """

    def __init__(self, message: str, *, part: str | None = None) -> None:
        super().__init__(message)
        self.part = part


class EncryptedDocument(DocumentError):
    """The document is encrypted or password-protected."""


class DocumentResourceLimit(DocumentError):
    """A fixed safety limit was crossed while parsing.

    Attributes:
        limit: The limit that was crossed, e.g. ``"max_entry_bytes"``.
    """

    def __init__(self, message: str, *, limit: str | None = None) -> None:
        super().__init__(message)
        self.limit = limit
