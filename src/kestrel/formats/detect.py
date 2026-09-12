"""Identify common byte signatures without changing their source."""

from pathlib import Path


def decode_text(data: bytes) -> tuple[str, str] | None:
    """Decode UTF-8 text, allowing a byte order mark."""
    encoding = "utf-8-sig" if data.startswith(b"\xef\xbb\xbf") else "utf-8"
    try:
        text = data.decode(encoding)
    except UnicodeDecodeError:
        return None
    if any(ord(char) < 32 and char not in "\t\r\n" for char in text):
        return None
    return text, encoding


def detect_bytes(data: bytes) -> str:
    """Classify signatures; structured text is validated during inspection."""
    signatures = (
        ((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"), "zip"),
        ((b"\x1f\x8b",), "gzip"),
        ((b"7z\xbc\xaf\x27\x1c",), "7z"),
        ((b"SQLite format 3\x00",), "sqlite"),
    )
    for prefixes, kind in signatures:
        if data.startswith(prefixes):
            return kind
    decoded = decode_text(data)
    if decoded is None:
        return "binary/unknown"
    text = decoded[0].lstrip()
    if text.startswith("<"):
        return "xml"
    if text.startswith(("{", "[", '"')) or text in {"null", "true", "false"}:
        return "json"
    if text and text[0] in "-0123456789":
        import json

        try:
            json.loads(text)
        except ValueError:
            pass
        else:
            return "json"
    return "text"


def detect_file(path: str | Path) -> str:
    """Read and classify a file."""
    return detect_bytes(Path(path).read_bytes())
