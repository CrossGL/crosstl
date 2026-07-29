import importlib.util
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DISCOVERY_PATH = ROOT / "demos" / "integrations" / "mlx" / "discover_entries.py"


def _load_discovery_tool():
    spec = importlib.util.spec_from_file_location(
        "mlx_entry_discovery",
        DISCOVERY_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_mlx_entry_discovery_tool_reports_deterministic_units(tmp_path):
    module = _load_discovery_tool()
    mlx_root = tmp_path / "mlx"
    kernels = mlx_root / module.MLX_KERNEL_ROOT
    kernels.mkdir(parents=True)
    (kernels / "first.metal").write_text(
        textwrap.dedent("""
            kernel void first(device float* output [[buffer(0)]]) {
              output[0] = 1.0f;
            }
            """),
        encoding="utf-8",
    )
    (kernels / "second.metal").write_text(
        textwrap.dedent("""
            template <typename T>
            [[kernel]] void second(device T* output [[buffer(0)]]) {
              output[0] = T(2);
            }

            template [[host_name("second_float")]] [[kernel]]
            decltype(second<float>) second<float>;
            """),
        encoding="utf-8",
    )

    report = module.discover_mlx_entries(mlx_root)

    assert report["unitCount"] == 2
    assert report["entryCount"] == 2
    assert report["diagnosticCount"] == 0
    assert [unit["source"] for unit in report["units"]] == [
        "mlx/backend/metal/kernels/first.metal",
        "mlx/backend/metal/kernels/second.metal",
    ]
    assert [entry["name"] for unit in report["units"] for entry in unit["entries"]] == [
        "first",
        "second_float",
    ]


def test_mlx_entry_discovery_cli_defaults_to_pinned_contract():
    module = _load_discovery_tool()

    args = module.parse_args(["--mlx-root", "/tmp/mlx"])

    assert args.expected_commit == module.MLX_COMMIT
    assert args.expected_unit_count == module.EXPECTED_SOURCE_UNIT_COUNT
    assert args.expected_entry_count == module.EXPECTED_ENTRY_COUNT
