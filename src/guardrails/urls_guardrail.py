"""Security validation for URLs sent to web extraction providers."""

from ipaddress import ip_address
from urllib.parse import urlsplit, urlunsplit


class URLValidationError(ValueError):
    """Raised when a URL is not safe for web extraction."""


_BLOCKED_HOSTNAMES = {
    "localhost",
    "metadata",
    "metadata.google.internal",
    "metadata.azure.internal",
    "instance-data",
}


def validate_public_http_url(
    url: str,
) -> str:
    """
    Validates that a URL is suitable for external web extraction.

    This blocks obvious SSRF-style targets such as localhost, private
    literal IP addresses, cloud metadata hosts, and non-HTTP schemes.

    The validated URL is returned unchanged except for surrounding
    whitespace removal.
    """

    cleaned_url = url.strip()

    if not cleaned_url:
        raise URLValidationError(
            "URL cannot be empty."
        )

    try:
        parsed = urlsplit(cleaned_url)

    except ValueError as exc:
        raise URLValidationError(
            "URL could not be parsed."
        ) from exc

    if parsed.scheme.lower() not in {
        "http",
        "https",
    }:
        raise URLValidationError(
            "Only HTTP and HTTPS URLs are allowed."
        )

    if not parsed.hostname:
        raise URLValidationError(
            "URL must contain a hostname."
        )

    if parsed.username or parsed.password:
        raise URLValidationError(
            "URLs containing embedded credentials are not allowed."
        )

    hostname = (
        parsed.hostname
        .lower()
        .rstrip(".")
    )

    if "%" in hostname:
        raise URLValidationError(
            "Scoped or malformed hostnames are not allowed."
        )

    if hostname in _BLOCKED_HOSTNAMES:
        raise URLValidationError(
            "Local or metadata hostnames are not allowed."
        )

    if (
        hostname.endswith(".localhost")
        or hostname.endswith(".local")
    ):
        raise URLValidationError(
            "Local network hostnames are not allowed."
        )

    try:
        parsed_ip = ip_address(hostname)

    except ValueError:
        parsed_ip = None

    if parsed_ip is not None:
        if not parsed_ip.is_global:
            raise URLValidationError(
                "Non-public IP addresses are not allowed."
            )

    elif "." not in hostname:
        # Single-label names are commonly internal/intranet hosts.
        raise URLValidationError(
            "Public web hostnames must be fully qualified."
        )

    return cleaned_url


def url_dedup_key(
    url: str,
) -> str:
    """
    Produces a lightweight canonical key for URL comparison.

    Host and scheme are case-insensitive. Path case is intentionally
    preserved because some web servers treat it as significant.
    URL fragments are ignored.
    """

    parsed = urlsplit(url.strip())
    scheme = parsed.scheme.lower()

    hostname = (parsed.hostname or "").lower()
    port = parsed.port
    if (
        port is not None
        and not (scheme == "http" and port == 80)
        and not (scheme == "https" and port == 443)
    ):
        netloc = f"{hostname}:{port}"
    else:
        netloc = hostname

    path = parsed.path or "/"

    if path != "/":
        path = path.rstrip("/")

    return urlunsplit(
        (
            scheme,
            netloc,
            path,
            parsed.query,
            "",
        )
    )

