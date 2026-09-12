"""Structural comparisons of immutable byte snapshots."""

import difflib
import hashlib
import io
import zipfile
import zlib
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from kestrel.formats.detect import decode_text, detect_bytes
from kestrel.formats.inspect import (
    MAX_ARCHIVE_SIZE,
    parse_json,
    parse_xml,
    read_entry,
    safe_entry,
    zip_entries,
)


def json_changes(left: Any, right: Any, path: str = "$") -> list[dict[str, Any]]:
    """Compare JSON recursively using escaped JSON Pointer components."""
    changes: list[dict[str, Any]] = []
    if type(left) is not type(right):
        return [{"path": path, "change": "changed", "left": left, "right": right}]
    if isinstance(left, dict):
        for key in sorted(left.keys() | right.keys()):
            child = path + "/" + key.replace("~", "~0").replace("/", "~1")
            if key not in left:
                changes.append({"path": child, "change": "added", "right": right[key]})
            elif key not in right:
                changes.append({"path": child, "change": "removed", "left": left[key]})
            else:
                changes.extend(json_changes(left[key], right[key], child))
    elif isinstance(left, list):
        for index in range(max(len(left), len(right))):
            child = f"{path}/{index}"
            if index >= len(left):
                changes.append(
                    {"path": child, "change": "added", "right": right[index]}
                )
            elif index >= len(right):
                changes.append(
                    {"path": child, "change": "removed", "left": left[index]}
                )
            else:
                changes.extend(json_changes(left[index], right[index], child))
    elif left != right:
        changes.append(
            {"path": path, "change": "changed", "left": left, "right": right}
        )
    return changes


def xml_structure(root: ET.Element) -> dict[str, Any]:
    result: dict[str, Any] = {}
    pending = [(root, f"/{root.tag}[1]")]
    while pending:
        node, path = pending.pop()
        result[path] = {
            "attributes": dict(node.attrib),
            "text": node.text or "",
            "tail": node.tail or "",
            "children": [child.tag for child in node],
        }
        counts: dict[str, int] = {}
        for child in node:
            counts[child.tag] = counts.get(child.tag, 0) + 1
            pending.append((child, f"{path}/{child.tag}[{counts[child.tag]}]"))
    return result


def binary_metadata(left: bytes, right: bytes) -> dict[str, Any]:
    offset = next(
        (i for i, (a, b) in enumerate(zip(left, right, strict=False)) if a != b), None
    )
    if offset is None and len(left) != len(right):
        offset = min(len(left), len(right))
    return {
        "left_size": len(left),
        "right_size": len(right),
        "size_difference": len(right) - len(left),
        "left_sha256": hashlib.sha256(left).hexdigest(),
        "right_sha256": hashlib.sha256(right).hexdigest(),
        "first_differing_byte": offset,
    }


def diff_bytes(
    left: bytes,
    right: bytes,
    left_name: str = "left",
    right_name: str = "right",
    *,
    allow_zip: bool = True,
) -> dict[str, Any]:
    """Compare content; malformed structures produce an error and binary fallback."""
    left_type, right_type = detect_bytes(left), detect_bytes(right)
    kind = left_type if left_type == right_type else "binary/unknown"
    result: dict[str, Any] = {
        "left": left_name,
        "right": right_name,
        "left_type": left_type,
        "right_type": right_type,
        "type": kind,
        "equal": left == right,
    }
    try:
        if kind == "json":
            changes = json_changes(parse_json(left), parse_json(right))
            result.update(changes=changes, equal=not changes)
        elif kind == "xml":
            a, b = xml_structure(parse_xml(left)), xml_structure(parse_xml(right))
            changes = []
            for path in sorted(a.keys() | b.keys()):
                if path not in a:
                    changes.append({"path": path, "change": "added", "right": b[path]})
                elif path not in b:
                    changes.append({"path": path, "change": "removed", "left": a[path]})
                elif a[path] != b[path]:
                    changes.append(
                        {
                            "path": path,
                            "change": "changed",
                            "left": a[path],
                            "right": b[path],
                        }
                    )
            result.update(changes=changes, equal=not changes)
        elif kind == "text":
            a_text, b_text = decode_text(left), decode_text(right)
            assert a_text is not None and b_text is not None
            lines = difflib.unified_diff(
                a_text[0].splitlines(keepends=True),
                b_text[0].splitlines(keepends=True),
                fromfile=left_name,
                tofile=right_name,
            )
            result["unified_diff"] = "".join(
                line
                if line.endswith("\n")
                else line + "\n\\ No newline at end of file\n"
                for line in lines
            )
            result["equal"] = a_text[0] == b_text[0]
        elif kind == "zip" and allow_zip:
            with (
                zipfile.ZipFile(io.BytesIO(left)) as a_zip,
                zipfile.ZipFile(io.BytesIO(right)) as b_zip,
            ):
                a_entries = {entry.filename: entry for entry in zip_entries(a_zip)}
                b_entries = {entry.filename: entry for entry in zip_entries(b_zip)}
                if (
                    sum(e.file_size for e in [*a_entries.values(), *b_entries.values()])
                    > MAX_ARCHIVE_SIZE
                ):
                    raise ValueError("ZIP contents exceed total comparison size limit")
                for name in a_entries.keys() | b_entries.keys():
                    if not safe_entry(name):
                        raise ValueError(f"Unsafe ZIP entry path: {name}")
                added = sorted(b_entries.keys() - a_entries.keys())
                removed = sorted(a_entries.keys() - b_entries.keys())
                comparisons = {}
                for name in sorted(a_entries.keys() & b_entries.keys()):
                    comparisons[name] = diff_bytes(
                        read_entry(a_zip, a_entries[name]),
                        read_entry(b_zip, b_entries[name]),
                        name,
                        name,
                        allow_zip=False,
                    )
                result.update(
                    added=added,
                    removed=removed,
                    entries=comparisons,
                    equal=not added
                    and not removed
                    and all(item["equal"] for item in comparisons.values()),
                )
                if any("error" in item for item in comparisons.values()):
                    result["error"] = (
                        "One or more ZIP entries could not be compared structurally"
                    )
        else:
            result["binary"] = binary_metadata(left, right)
    except (
        ValueError,
        ET.ParseError,
        zipfile.BadZipFile,
        zlib.error,
        EOFError,
        RuntimeError,
        NotImplementedError,
        OSError,
        RecursionError,
        OverflowError,
    ) as error:
        result["error"] = str(error)
        result["binary"] = binary_metadata(left, right)
    return result


def diff_files(left: str | Path, right: str | Path) -> dict[str, Any]:
    a, b = Path(left), Path(right)
    return diff_bytes(a.read_bytes(), b.read_bytes(), a.name, b.name)
