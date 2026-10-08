"""Structured logging configuration for the application."""

import json
import logging
import sys
from datetime import UTC, datetime

from src.config.settings import ObservabilitySettings


class JsonFormatter(logging.Formatter):
    """Formats application log records as JSON."""

    _structured_fields = (
        "event",
        "job_id",
        "research_session_id",
        "node",
        "iteration",
        "query_id",
        "source_id",
        "error_type",
        "resource",
        "data",
    )

    def format(self, record: logging.LogRecord) -> str:
        """Converts a log record into a JSON string."""

        log_data: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for field_name in self._structured_fields:
            field_value = getattr(record, field_name, None)

            if field_value is not None:
                log_data[field_name] = field_value

        if record.exc_info:
            log_data["exception"] = self.formatException(
                record.exc_info
            )

        return json.dumps(
            log_data,
            ensure_ascii=False,
        )


def configure_logging(
    settings: ObservabilitySettings,
) -> None:
    """Configures the root logger for structured application logs."""

    log_level = getattr(
        logging,
        settings.log_level.upper(),
        logging.INFO,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)
    root_logger.addHandler(handler)

