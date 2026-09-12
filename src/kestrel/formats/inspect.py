"""Inspect containers and structured data entirely in memory."""

import hashlib
import io
import json
import math
import sqlite3
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any
from xml.etree import ElementTree as ET

from kestrel.formats.detect import decode_text, detect_bytes

MAX_ENTRY_SIZE = 16 * 1024 * 1024
MAX_ARCHIVE_SIZE = 64 * 1024 * 1024
MAX_ENTRIES = 10000


def parse_json(data: bytes) -> Any:
    """Parse JSON, rejecting nonstandard constants and duplicate keys."""

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value: str) -> Any:
        raise ValueError(f"Invalid JSON constant: {value}")

    def number(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("JSON number exceeds finite floating-point range")
        return parsed

    return json.loads(
        data, object_pairs_hook=pairs, parse_constant=constant, parse_float=number
    )


def parse_xml(data: bytes) -> ET.Element:
    """Parse XML without accepting document type or entity declarations."""
    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise ValueError("XML declarations for document types or entities are disabled")
    decoded = decode_text(data)
    if decoded is None:
        raise ValueError("XML must be UTF-8 text")
    return ET.fromstring(decoded[0])


def safe_entry(name: str) -> bool:
    """Flag absolute paths, drive paths and traversal components."""
    normalized = name.replace("\\", "/")
    return (
        bool(name)
        and not normalized.startswith("/")
        and ":" not in normalized
        and ".." not in PurePosixPath(normalized).parts
        and "\x00" not in name
    )


def zip_entries(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    entries = archive.infolist()
    if len(entries) > MAX_ENTRIES:
        raise ValueError("ZIP entry count exceeds inspection limit")
    names = [entry.filename for entry in entries]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate ZIP entry names are ambiguous")
    return entries


def read_entry(archive: zipfile.ZipFile, entry: zipfile.ZipInfo) -> bytes:
    if not safe_entry(entry.filename):
        raise ValueError(f"Unsafe ZIP entry path: {entry.filename}")
    if entry.file_size > MAX_ENTRY_SIZE:
        raise ValueError("ZIP entry exceeds comparison size limit")
    with archive.open(entry) as stream:
        data = stream.read(MAX_ENTRY_SIZE + 1)
    if len(data) > MAX_ENTRY_SIZE:
        raise ValueError("ZIP entry exceeds comparison size limit")
    return data


def inspect_bytes(data: bytes, name: str = "<memory>") -> dict[str, Any]:
    """Return serializable metadata, including structured parsing errors."""
    kind = detect_bytes(data)
    result: dict[str, Any] = {
        "name": name,
        "size": len(data),
        "type": kind,
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    decoded = decode_text(data) if kind in {"text", "json", "xml"} else None
    if decoded is not None:
        result["encoding"] = decoded[1]
    try:
        if kind == "json":
            value = parse_json(data)
            types = {
                dict: "object",
                list: "array",
                str: "string",
                bool: "boolean",
                int: "number",
                float: "number",
                type(None): "null",
            }
            summary: dict[str, Any] = {"type": types[type(value)]}
            if isinstance(value, (dict, list)):
                summary["length"] = len(value)
            if isinstance(value, dict):
                summary["keys"] = sorted(value)
            result["json"] = summary
        elif kind == "xml":
            root = parse_xml(data)
            result["xml"] = {
                "root_tag": root.tag,
                "element_count": sum(1 for _ in root.iter()),
            }
        elif kind == "zip":
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                result["entries"] = [
                    {
                        "name": entry.filename,
                        "size": entry.file_size,
                        "compressed_size": entry.compress_size,
                        "safe_path": safe_entry(entry.filename),
                    }
                    for entry in zip_entries(archive)
                ]
        elif kind == "sqlite":
            connection = sqlite3.connect(":memory:")
            try:
                connection.deserialize(data)
                connection.execute("PRAGMA query_only = ON")
                connection.execute("PRAGMA trusted_schema = OFF")
                result["tables"] = [
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_schema "
                        "WHERE type = 'table' ORDER BY name"
                    )
                ]
            finally:
                connection.close()
        else:
            result["prefix_hex"] = data[:32].hex()
    except (
        ValueError,
        ET.ParseError,
        zipfile.BadZipFile,
        sqlite3.Error,
        RecursionError,
        OverflowError,
    ) as error:
        result["error"] = str(error)
    return result


def inspect_file(path: str | Path) -> dict[str, Any]:
    """Read a file once, then inspect its immutable byte snapshot."""
    source = Path(path)
    return inspect_bytes(source.read_bytes(), source.name)
