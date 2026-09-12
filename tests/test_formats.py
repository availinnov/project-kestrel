"""Generated fixtures for format detection, inspection and comparison."""

import hashlib
import io
import json
import os
import sqlite3
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import pytest

from kestrel.formats.detect import detect_bytes, detect_file
from kestrel.formats.diff import diff_bytes, diff_files
from kestrel.formats.inspect import inspect_bytes, inspect_file


def archive(entries: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as container:
        for name, data in entries.items():
            container.writestr(name, data)
    return stream.getvalue()


@pytest.mark.parametrize(
    ("data", "kind"),
    [
        (b"PK\x03\x04", "zip"),
        (b"PK\x05\x06", "zip"),
        (b"\x1f\x8b", "gzip"),
        (b"7z\xbc\xaf\x27\x1c", "7z"),
        (b"SQLite format 3\x00", "sqlite"),
        (b" <root/>", "xml"),
        (b' {"a": 1}', "json"),
        (b"[]", "json"),
        (b"true", "json"),
        (b"42", "json"),
        (b'"value"', "json"),
        (b"hello", "text"),
        (b"\xef\xbb\xbfhello", "text"),
        (b"", "text"),
        (b"\x00abc", "binary/unknown"),
        (b"\xff", "binary/unknown"),
    ],
)
def test_detection(data: bytes, kind: str) -> None:
    assert detect_bytes(data) == kind


def test_json_inspect_and_diff() -> None:
    result = inspect_bytes(b'{"items": [1], "ready": true}')
    assert result["json"] == {"type": "object", "length": 2, "keys": ["items", "ready"]}
    assert result["encoding"] == "utf-8"
    changes = diff_bytes(b'{"a/b": [1, 2], "old": 0}', b'{"a/b": [true], "new": 3}')[
        "changes"
    ]
    assert {item["path"] for item in changes} == {
        "$/a~1b/0",
        "$/a~1b/1",
        "$/old",
        "$/new",
    }
    assert diff_bytes(b'{"a":1,"b":2}', b'{ "b": 2, "a": 1 }')["equal"]
    assert not diff_bytes(b"true", b"1")["equal"]


def test_xml_inspect_and_diff() -> None:
    left = b'<root a="1"><item>one</item><item/>tail</root>'
    right = b'<root a="2"><item>two</item><extra/></root>'
    assert inspect_bytes(left)["xml"] == {"root_tag": "root", "element_count": 3}
    result = diff_bytes(left, right)
    assert not result["equal"]
    assert {item["path"] for item in result["changes"]} == {
        "/root[1]",
        "/root[1]/item[1]",
        "/root[1]/item[2]",
        "/root[1]/extra[1]",
    }
    assert diff_bytes(b'<r a="1" b="2"/>', b'<r b="2" a="1"></r>')["equal"]
    assert not diff_bytes(b"<r><a/><b/></r>", b"<r><b/><a/></r>")["equal"]
    assert not diff_bytes(b"<r><a/>one</r>", b"<r><a/>two</r>")["equal"]


def test_zip_inspect_and_diff() -> None:
    left = archive(
        {"data.json": b'{"n":1}', "data.xml": b"<r/>", "note.txt": b"one\n", "old": b""}
    )
    right = archive(
        {"data.json": b'{"n":2}', "data.xml": b"<s/>", "note.txt": b"two\n", "new": b""}
    )
    metadata = inspect_bytes(left)
    assert len(metadata["entries"]) == 4
    assert metadata["entries"][0]["size"] == 7
    assert metadata["entries"][0]["compressed_size"] > 0
    result = diff_bytes(left, right)
    assert result["added"] == ["new"]
    assert result["removed"] == ["old"]
    assert result["entries"]["data.json"]["changes"][0]["path"] == "$/n"
    assert result["entries"]["data.xml"]["changes"]
    assert "-one" in result["entries"]["note.txt"]["unified_diff"]
    assert diff_bytes(left, left)["equal"]


@pytest.mark.parametrize(
    "name", ["../escape", "/absolute", "C:/absolute", "a/../../escape", "..\\escape"]
)
def test_zip_unsafe_paths(name: str) -> None:
    data = archive({name: b"content"})
    assert not inspect_bytes(data)["entries"][0]["safe_path"]
    assert "Unsafe ZIP entry" in diff_bytes(data, data)["error"]


def test_zip_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kestrel.formats.inspect.MAX_ENTRY_SIZE", 4)
    data = archive({"large": b"12345"})
    assert "size limit" in diff_bytes(data, data)["error"]


def test_zip_duplicates() -> None:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as container:
        container.writestr("a", b"1")
        with pytest.warns(UserWarning):
            container.writestr("a", b"2")
    assert "Duplicate" in inspect_bytes(stream.getvalue())["error"]


@pytest.mark.parametrize(
    ("left", "right", "offset"),
    [
        (b"\x00abc", b"\x00axc", 2),
        (b"\x00a", b"\x00abc", 2),
        (b"\x00abc", b"\x00a", 2),
        (b"\x00abc", b"\x00abc", None),
    ],
)
def test_binary_diff(left: bytes, right: bytes, offset: int | None) -> None:
    result = diff_bytes(left, right)["binary"]
    assert result["first_differing_byte"] == offset
    assert result["size_difference"] == len(right) - len(left)
    assert result["left_sha256"] == hashlib.sha256(left).hexdigest()
    assert result["right_sha256"] == hashlib.sha256(right).hexdigest()


@pytest.mark.parametrize(
    "data",
    [
        b'{"broken":',
        b"<root>",
        b"PK\x03\x04bad",
        b"SQLite format 3\x00bad",
        b'{"a":1,"a":2}',
        b'{"a":NaN}',
        b'<!DOCTYPE r [<!ENTITY x "text">]><r>&x;</r>',
    ],
)
def test_malformed(data: bytes) -> None:
    result = inspect_bytes(data)
    assert "error" in result
    assert result["sha256"] == hashlib.sha256(data).hexdigest()
    if result["type"] != "sqlite":
        comparison = diff_bytes(data, data)
        assert "error" in comparison
        assert comparison["binary"]["first_differing_byte"] is None


def test_sqlite(tmp_path: Path) -> None:
    source = tmp_path / "database.db"
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE example (value TEXT)")
    connection.close()
    before = source.read_bytes()
    assert inspect_file(source)["tables"] == ["example"]
    assert source.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["database.db"]


def test_sources_unchanged(tmp_path: Path) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_bytes(b'{"value": 1}')
    b.write_bytes(b'{"value": 2}')
    before = [(p.read_bytes(), p.stat().st_mtime_ns) for p in (a, b)]
    assert detect_file(a) == "json"
    assert inspect_file(a)["name"] == "a.json"
    assert not diff_files(a, b)["equal"]
    assert before == [(p.read_bytes(), p.stat().st_mtime_ns) for p in (a, b)]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["kestrel", *args],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


@pytest.mark.parametrize("machine", [False, True])
def test_cli(tmp_path: Path, machine: bool) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text('{"count":1}', encoding="utf-8")
    b.write_text('{"count":2}', encoding="utf-8")
    options = ["--json"] if machine else []
    inspected = run_cli("inspect", str(a), *options)
    compared = run_cli("diff", str(a), str(b), *options)
    assert inspected.returncode == compared.returncode == 0
    assert inspected.stderr == compared.stderr == ""
    if machine:
        inspection: dict[str, Any] = json.loads(inspected.stdout)
        assert inspection["type"] == "json"
        assert json.loads(compared.stdout)["changes"][0]["path"] == "$/count"
    else:
        assert "name: a.json" in inspected.stdout
        assert "$/count" in compared.stdout


def test_cli_errors(tmp_path: Path) -> None:
    missing = run_cli("inspect", str(tmp_path / "missing"), "--json")
    assert missing.returncode == 1
    assert "error" in json.loads(missing.stdout)
    malformed = tmp_path / "bad.json"
    malformed.write_bytes(b"{")
    result = run_cli("inspect", str(malformed), "--json")
    assert result.returncode == 1
    assert "error" in json.loads(result.stdout)
    assert "Traceback" not in result.stderr


def test_text_diff() -> None:
    result = diff_bytes(b"alpha\nbeta\n", b"alpha\ngamma\n")
    assert "--- left\n+++ right\n" in result["unified_diff"]
    assert "-beta\n+gamma\n" in result["unified_diff"]


def test_text_missing_newline() -> None:
    result = diff_bytes(b"one", b"two")
    assert "-one\n\\ No newline at end of file\n+two\n" in result["unified_diff"]
    assert not diff_bytes(b"one", b"one\n")["equal"]


def test_zip_corrupt_payload() -> None:
    data = bytearray(archive({"a": b"contents"}))
    data[31] = 255  # First compressed byte after the local header and one-byte name.
    result = diff_bytes(bytes(data), bytes(data))
    assert "error" in result
    assert "binary" in result


def test_zip_added_unsafe_entry() -> None:
    assert (
        "Unsafe ZIP entry" in diff_bytes(archive({}), archive({"../bad": b""}))["error"]
    )


def test_json_extreme_number() -> None:
    assert "error" in inspect_bytes(b'{"n":1e9999}')
    assert "error" in diff_bytes(b'{"n":1e9999}', b'{"n":1}')


def test_zip_total_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kestrel.formats.diff.MAX_ARCHIVE_SIZE", 3)
    data = archive({"a": b"12"})
    assert "total comparison" in diff_bytes(data, data)["error"]


def test_zip_entry_count_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kestrel.formats.inspect.MAX_ENTRIES", 1)
    assert "entry count" in inspect_bytes(archive({"a": b"", "b": b""}))["error"]
