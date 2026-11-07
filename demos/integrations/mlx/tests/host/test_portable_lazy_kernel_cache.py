"""Host checks of the shared lazy kernel cache, with a faked translator."""

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from demos.integrations.mlx.portable_host import lazy_kernel_cache

SOURCE = f"{lazy_kernel_cache.KERNELS}/fake.metal"
ENTRY = "fake_entry"


class _Report:
    def __init__(self, entry):
        self.payload = {
            "summary": {"failedCount": 0},
            "artifacts": [{"entryPoint": {"source": entry}}],
            "diagnostics": [],
            # A whole-source unit list, as the real report carries.
            "units": [{"path": SOURCE, "text": "x" * 4096} for _ in range(64)],
        }

    def write_json(self, path):
        Path(path).write_text(json.dumps(self.payload, indent=2), encoding="utf-8")

    def to_json(self):
        return self.payload


def _fake_translator(monkeypatch, root, calls):
    source_hash = hashlib.sha256((root / SOURCE).read_bytes()).hexdigest()
    loader = {
        "success": True,
        "diagnostics": [],
        "loadUnits": [{"entryPoint": {"source": ENTRY}}],
    }

    def translate(config, format_output):
        calls.append(config)
        return _Report(ENTRY)

    def package(manifest, output):
        Path(output).mkdir(parents=True)
        (Path(output) / "runtime-package.json").write_text("{}", encoding="utf-8")
        return {"success": True}

    fakes = {
        "require_revision": lambda path: None,
        "translation_implementation_hash": lambda: "implementation",
        "load_project_config": lambda path, config: Path(config).read_text(),
        "translate_project": translate,
        "build_runtime_artifact_manifest": lambda path: {"success": True},
        "build_runtime_package": package,
        "build_runtime_loader_manifest": lambda path: loader,
        "build_native_loader_abi_descriptor": lambda manifest: {
            "target": "opengl",
            "stage": "compute",
            "source": {"hash": {"algorithm": "sha256", "value": source_hash}},
        },
    }
    for name, fake in fakes.items():
        monkeypatch.setattr(lazy_kernel_cache, name, fake)
    monkeypatch.setattr(
        lazy_kernel_cache.subprocess, "check_output", lambda *args, **kwargs: ""
    )


@pytest.fixture
def cache(monkeypatch, tmp_path):
    root = tmp_path / "mlx"
    (root / SOURCE).parent.mkdir(parents=True)
    (root / SOURCE).write_text("kernel void fake_entry() {}\n", encoding="utf-8")
    calls = []
    _fake_translator(monkeypatch, root, calls)
    recipe = lazy_kernel_cache.KernelRecipe(
        label="Fake",
        source=SOURCE,
        signature=lambda entry: None,
        options=lambda target, entry: "",
        module=__file__,
    )
    return (
        lazy_kernel_cache.LazyKernelCache(root, tmp_path / "cache", "opengl", recipe),
        calls,
    )


def test_entries_retain_a_compressed_report_and_translate_once(cache):
    lazy, calls = cache
    descriptor, package = lazy.get(ENTRY)
    translation = package.parent / "translation"
    assert descriptor["target"] == "opengl" and len(calls) == 1
    assert not (translation / "report.json").exists()
    compressed = (translation / "report.json.gz").read_bytes()
    report = json.loads(gzip.decompress(compressed))
    assert report["artifacts"][0]["entryPoint"]["source"] == ENTRY
    assert len(compressed) * 20 < len(gzip.decompress(compressed))
    assert f'"{SOURCE}" = "{ENTRY}"' in (translation / "crosstl.toml").read_text()
    lazy._loaded.clear()
    assert lazy.get(ENTRY) == (descriptor, package) and len(calls) == 1


def test_compressed_reports_are_byte_identical(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "report.json").write_text('{"units": []}', encoding="utf-8")
    first, second = tmp_path / "first", tmp_path / "second"
    lazy_kernel_cache.retain_translation(work, first)
    lazy_kernel_cache.retain_translation(work, second)
    assert (first / "report.json.gz").read_bytes() == (
        second / "report.json.gz"
    ).read_bytes()
    assert sorted(path.name for path in first.iterdir()) == ["report.json.gz"]
