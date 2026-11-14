"""Retained native modules are whole whenever a concurrent reader sees them."""

import hashlib
import threading

import pytest

from demos.integrations.mlx.portable_host.retained_modules import retain


def test_retain_names_a_module_by_its_hash(tmp_path):
    data = b"\x00module\xff" * 64
    record = retain(tmp_path / "native-modules", data, ".dxil")
    digest = hashlib.sha256(data).hexdigest()
    assert record == {
        "file": str(tmp_path / "native-modules" / (digest + ".dxil")),
        "sha256": digest,
    }
    assert (tmp_path / "native-modules" / (digest + ".dxil")).read_bytes() == data
    # A second retention keeps the module and leaves no temporary file.
    assert retain(tmp_path / "native-modules", data, ".dxil") == record
    assert [path.name for path in (tmp_path / "native-modules").iterdir()] == [
        digest + ".dxil"
    ]


def test_retain_refuses_an_empty_module(tmp_path):
    with pytest.raises(RuntimeError, match="empty module"):
        retain(tmp_path, b"", ".air")
    assert list(tmp_path.iterdir()) == []


def test_concurrent_retention_never_exposes_a_partial_module(tmp_path):
    # test_threads.TestThreads.test_concurrent_transforms compiles one module
    # on three threads while each thread's audit reads it back.
    data = bytes(range(256)) * 4096
    digest = hashlib.sha256(data).hexdigest()
    directory = tmp_path / "native-modules"
    errors = []
    start = threading.Barrier(8)

    def worker():
        try:
            start.wait()
            for _ in range(50):
                record = retain(directory, data, ".glsl")
                read = open(record["file"], "rb").read()
                if hashlib.sha256(read).hexdigest() != digest:
                    errors.append(len(read))
        except Exception as error:  # surfaced through the errors list
            errors.append(repr(error))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert [path.name for path in directory.iterdir()] == [digest + ".glsl"]
