"""Domain exceptions for private document management."""


class DocumentError(Exception):
    """Base exception for private document operations."""


class DocumentNotFoundError(DocumentError):
    """Raised when a requested document does not exist."""


class UnsupportedDocumentFormatError(DocumentError):
    """Raised when a document format is unsupported."""


class EmptyDocumentError(DocumentError):
    """Raised when an uploaded document contains no data."""


class DocumentQuotaExceededError(DocumentError):
    """Raised when document storage quota would be exceeded."""

    def __init__(
        self,
        *,
        current_bytes: int,
        incoming_bytes: int,
        maximum_bytes: int,
    ) -> None:
        self.current_bytes = current_bytes
        self.incoming_bytes = incoming_bytes
        self.maximum_bytes = maximum_bytes

        super().__init__(
            "Document upload would exceed the configured "
            "document storage quota."
        )


class DocumentFileTooLargeError(DocumentError):
    """Raised while streaming a file that exceeds its allowed size."""

    def __init__(
        self,
        *,
        bytes_read: int,
        maximum_bytes: int,
    ) -> None:
        self.bytes_read = bytes_read
        self.maximum_bytes = maximum_bytes

        super().__init__(
            "Uploaded document exceeds its allowed storage size."
        )


class DocumentStorageError(DocumentError):
    """Raised when document file storage fails."""

class DocumentParsingError(DocumentError):
    """Raised when text cannot be extracted from a document."""

    