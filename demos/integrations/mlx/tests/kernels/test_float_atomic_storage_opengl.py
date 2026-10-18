"""Word-backed OpenGL storage for unchanged upstream atomic operations."""

import json
import os
import sys
from pathlib import Path

import pytest

from demos.integrations.mlx.tests.kernels.test_float_scatter_runtime import (
    CASES as SCATTER_CASES,
)
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_opengl_float_atomic_storage import REQUIRE_ENV


@pytest.mark.parametrize(
    "case", ("load", "store", "compare", "multiply", "minimum", "maximum", "contention")
)
def test_unchanged_mlx_atomic_helpers_use_word_storage(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native unchanged MLX helpers")
    assert sys.platform == "linux"
    from demos.integrations.mlx.portable_host.prepare import COMMIT
    from demos.integrations.mlx.tests.kernels.test_atomic_load_runtime import HEADER
    from demos.integrations.mlx.tests.kernels.test_float_atomic_compare_exchange import (
        _request as compare_request,
    )
    from demos.integrations.mlx.tests.kernels.test_float_atomic_memory import (
        _request as memory_request,
    )
    from demos.integrations.mlx.tests.kernels.test_general_scatter_runtime import (
        _verify_source,
    )

    root = Path(os.environ["CROSTL_MLX_CURRENT_ROOT"]).resolve()
    hashes = _verify_source(root, (HEADER,))
    try:
        request_builder = (
            memory_request if case in {"load", "store"} else compare_request
        )
        _, request, expected = request_builder(root, "opengl", tmp_path, case)
        (tmp_path / "workload.json").write_text(
            json.dumps({"commit": COMMIT, "headers": hashes, "case": case}),
            encoding="utf-8",
        )
        _execute(request, expected, tmp_path)
    finally:
        assert _verify_source(root, (HEADER,)) == hashes


@pytest.mark.parametrize("operation,layout", SCATTER_CASES)
def test_unchanged_mlx_float_scatter_uses_word_storage(tmp_path, operation, layout):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native unchanged MLX scatter")
    assert sys.platform == "linux"
    from demos.integrations.mlx.portable_host.prepare import COMMIT
    from demos.integrations.mlx.tests.kernels.test_atomic_load_runtime import HEADER
    from demos.integrations.mlx.tests.kernels.test_float_scatter_runtime import (
        _request as scatter_request,
    )
    from demos.integrations.mlx.tests.kernels.test_general_scatter_runtime import (
        HEADERS,
        JIT,
        _verify_source,
    )

    root = Path(os.environ["CROSTL_MLX_CURRENT_ROOT"]).resolve()
    hashes = _verify_source(root, (*HEADERS, JIT, HEADER))
    try:
        entry, _, request, expected = scatter_request(
            root, "opengl", tmp_path, operation, layout
        )
        (tmp_path / "workload.json").write_text(
            json.dumps(
                {
                    "commit": COMMIT,
                    "headers": hashes,
                    "entry": entry,
                    "operation": operation,
                    "layout": layout,
                }
            ),
            encoding="utf-8",
        )
        _execute(request, expected, tmp_path)
    finally:
        assert _verify_source(root, (*HEADERS, JIT, HEADER)) == hashes
