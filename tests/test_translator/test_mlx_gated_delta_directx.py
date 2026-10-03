"""Native DirectX compilation of the pinned MLX gated-delta backward entry."""

import hashlib
import json
import os

import pytest

from tests.test_translator.test_directx_float_atomics import _compile
from tests.test_translator.test_mlx_gated_delta_metal import ENTRY, _translate_pinned

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_GATED_DELTA_DIRECTX"


def test_pinned_gated_delta_backward_compiles_with_float_storage(tmp_path, monkeypatch):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 to require the pinned DirectX compiler gate")
    monkeypatch.setenv("CROSTL_REQUIRE_DIRECTX_FLOAT_ATOMICS", "1")
    revision, original, generated = _translate_pinned(tmp_path, "directx")
    source = generated.read_text(encoding="utf-8")
    assert "InterlockedCompareExchangeFloatBitwise(" in source
    assert "__crossgl_float_atomic_" in source
    artifact, module = _compile(
        source, tmp_path, profile="cs_6_2", flags=("-enable-16bit-types",)
    )
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "commit": revision,
                "entry": ENTRY,
                "targetEntry": "CSMain",
                "execution": "not-tested",
                "originalSha256": hashlib.sha256(original.read_bytes()).hexdigest(),
                "artifactSha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "moduleSha256": hashlib.sha256(module.read_bytes()).hexdigest(),
                "moduleSizeBytes": module.stat().st_size,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
