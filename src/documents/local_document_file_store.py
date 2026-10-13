"""Local filesystem storage for original private documents."""

import asyncio
import hashlib
import os
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

from src.documents.document_exceptions import (
    DocumentFileTooLargeError,
    DocumentStorageError,
)
from src.models.document_models import StoredDocumentFile


_COPY_CHUNK_SIZE = 1024 * 1024


class LocalDocumentFileStore:
    """
    Stores original private documents on the local filesystem.

    Files are written to a temporary location first and atomically
    moved into place after successful validation.
    """

    def __init__(
        self,
        *,
        root_directory: Path,
    ) -> None:
        self._root_directory = root_directory.resolve()

        self._root_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    async def save(
        self,
        *,
        source: BinaryIO,
        storage_key: str,
        max_bytes: int,
    ) -> StoredDocumentFile:
        """Stream a file to disk without loading it entirely into RAM."""

        return await asyncio.to_thread(
            self._save_sync,
            source,
            storage_key,
            max_bytes,
        )

    def _save_sync(
        self,
        source: BinaryIO,
        storage_key: str,
        max_bytes: int,
    ) -> StoredDocumentFile:
        destination = self._resolve_storage_key(
            storage_key
        )

        temporary_path = self._root_directory / (
            f".{storage_key}.{uuid4().hex}.part"
        )

        hasher = hashlib.sha256()
        total_bytes = 0

        try:
            source.seek(0)

            with temporary_path.open("xb") as output:
                while True:
                    chunk = source.read(
                        _COPY_CHUNK_SIZE
                    )

                    if not chunk:
                        break

                    total_bytes += len(chunk)

                    if total_bytes > max_bytes:
                        raise DocumentFileTooLargeError(
                            bytes_read=total_bytes,
                            maximum_bytes=max_bytes,
                        )

                    hasher.update(chunk)
                    output.write(chunk)

            os.replace(
                temporary_path,
                destination,
            )

            return StoredDocumentFile(
                storage_key=storage_key,
                size_bytes=total_bytes,
                content_hash=hasher.hexdigest(),
            )

        except DocumentFileTooLargeError:
            raise

        except Exception as exc:
            raise DocumentStorageError(
                "Failed to store uploaded document."
            ) from exc

        finally:
            temporary_path.unlink(
                missing_ok=True
            )

    def get_path(
        self,
        *,
        storage_key: str,
    ) -> Path | None:
        """Return the physical file path if it exists."""

        path = self._resolve_storage_key(
            storage_key
        )

        if not path.is_file():
            return None

        return path

    async def delete(
        self,
        *,
        storage_key: str,
    ) -> None:
        """Delete a stored document file."""

        path = self._resolve_storage_key(
            storage_key
        )

        await asyncio.to_thread(
            path.unlink,
            missing_ok=True,
        )

    def _resolve_storage_key(
        self,
        storage_key: str,
    ) -> Path:
        """
        Resolve a storage key while preventing path traversal.
        """

        path = (
            self._root_directory
            / storage_key
        ).resolve()

        if path.parent != self._root_directory:
            raise DocumentStorageError(
                "Invalid document storage key."
            )

        return path

    