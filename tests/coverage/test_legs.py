import io
from pathlib import Path

import pytest
import yaml

from tools.coverage import legs

REPO = Path(__file__).resolve().parents[2]
NONE = {"metal": False, "compute": False, "rust": False, "d3d12": False}

# A small CrossTL: the dispatcher imports every generator, as crosstl/_crosstl.py does, so only the walk that stops
# at the languages' own modules keeps the pipeline from reaching everything.
TREE = {
    "__init__.py": "from . import _crosstl\n",
    "_crosstl.py": (
        "from .translator import parser, semantics\nfrom .translator.codegen import registry\n"
    ),
    "translator/__init__.py": "",
    "translator/parser.py": "",
    "translator/semantics/__init__.py": "from .lower import Lowerer\n",
    "translator/semantics/lower.py": "",
    "translator/codegen/__init__.py": "",
    "translator/codegen/registry.py": (
        "from . import GLSL_codegen, metal_codegen, cuda_codegen, hip_codegen, rust_codegen\n"
        "from . import directx_codegen\n"
    ),
    "translator/codegen/shared_names.py": "",
    "translator/codegen/texture_names.py": "",
    "translator/codegen/pointer_math.py": "",
    "translator/codegen/char_names.py": "",
    "translator/codegen/GLSL_codegen.py": (
        "from crosstl.translator import parser\nfrom .shared_names import name\n"
    ),
    "translator/codegen/metal_codegen.py": (
        "from ..semantics import Lowerer\nfrom .shared_names import name\n\n"
        "def emit():\n    from .texture_names import sample\n"
    ),
    "translator/codegen/cuda_codegen.py": (
        "from crosstl.translator.codegen.pointer_math import offset\n"
    ),
    "translator/codegen/hip_codegen.py": (
        "from . import pointer_math\nfrom .char_names import quote\n"
    ),
    "translator/codegen/rust_codegen.py": (
        "import crosstl.translator.codegen.char_names\n"
    ),
    # Only HLSL imports these, yet HLSL is a Linux-leg language, so they are not the D3D12 leg's alone.
    "translator/codegen/directx_codegen.py": "from .hlsl_buffer_layout import layout\n",
    "translator/codegen/hlsl_buffer_layout.py": "from .wave_names import wave\n",
    "translator/codegen/wave_names.py": "",
    "backend/__init__.py": "",
    "backend/common_ast.py": "",
    "backend/GLSL/__init__.py": "from ..common_ast import Node\n",
    "backend/Metal/__init__.py": "from ..common_ast import Node\n",
    "backend/CUDA/__init__.py": "",
    "backend/DirectX/__init__.py": "from ..common_ast import Node\n",
}


@pytest.fixture
def tree(tmp_path):
    for name, text in TREE.items():
        path = tmp_path / "crosstl" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return tmp_path


def _legs(*runs):
    return {**NONE, **dict.fromkeys(runs, True)}


@pytest.mark.parametrize(
    "changed,runs",
    [
        ("crosstl/translator/codegen/metal_codegen.py", ["metal"]),
        ("crosstl/backend/Metal/__init__.py", ["metal"]),
        ("crosstl/backend/CUDA/__init__.py", ["compute"]),
        # Imported inside a function, by the Metal generator only.
        ("crosstl/translator/codegen/texture_names.py", ["metal"]),
        ("crosstl/translator/codegen/pointer_math.py", ["compute"]),
        ("crosstl/translator/codegen/char_names.py", ["compute", "rust"]),
        # Shared with GLSL, the Linux leg measures them.
        ("crosstl/translator/codegen/shared_names.py", []),
        ("crosstl/translator/parser.py", []),
        ("crosstl/backend/common_ast.py", []),
        # The pipeline lowers for every target, although only Metal's generator imports it.
        ("crosstl/translator/semantics/lower.py", []),
        ("tests/conformance/vkrun.c", ["rust"]),
        ("tests/conformance/runtimes.py", ["metal", "rust", "d3d12"]),
        ("tools/coverage/probes.py", ["metal", "compute", "rust", "d3d12"]),
        (".github/workflows/coverage.yml", ["metal", "compute", "rust", "d3d12"]),
        ("README.md", []),
        # HLSL's own front end and generator modules, and the files the D3D12 runtime executes with.
        ("crosstl/translator/codegen/directx_codegen.py", ["d3d12"]),
        ("crosstl/translator/codegen/hlsl_buffer_layout.py", ["d3d12"]),
        ("crosstl/backend/DirectX/__init__.py", ["d3d12"]),
        ("tests/conformance/toolchains.py", ["d3d12"]),
        ("crosstl/project/directx_runtime_worker.cpp", ["d3d12"]),
        (".github/actions/install-windows-d3d12/action.yml", ["d3d12"]),
        # A module only HLSL imports, but under no HLSL prefix: the Linux leg measures HLSL on it.
        ("crosstl/translator/codegen/wave_names.py", []),
    ],
)
def test_a_change_runs_the_legs_whose_rows_it_can_move(tree, changed, runs):
    assert legs.plan([changed], tree) == _legs(*runs)


def test_a_change_runs_every_leg_any_of_its_files_needs(tree):
    changed = [
        "crosstl/translator/codegen/texture_names.py\n",
        "crosstl/translator/codegen/pointer_math.py\n",
        "\n",
    ]
    assert legs.plan(changed, tree) == _legs("metal", "compute")


def test_a_deleted_module_runs_the_legs_its_path_names(tree):
    assert legs.plan(["crosstl/translator/codegen/metal_removed.py"], tree) == _legs(
        "metal"
    )
    assert legs.plan(["crosstl/translator/codegen/removed.py"], tree) == NONE


def test_a_leg_whose_generators_were_renamed_runs_for_any_translator_change(tree):
    (tree / "crosstl/translator/codegen/rust_codegen.py").rename(
        tree / "crosstl/translator/codegen/rustgen.py"
    )
    assert legs.plan(["crosstl/translator/parser.py"], tree) == _legs("rust")
    assert legs.plan(["README.md"], tree) == NONE


def test_the_command_prints_github_outputs(tree, monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.stdin", io.StringIO("crosstl/translator/codegen/char_names.py\n")
    )
    assert legs.main([str(tree)]) == 0
    assert (
        capsys.readouterr().out == "metal=false\ncompute=true\nrust=true\nd3d12=false\n"
    )


@pytest.mark.parametrize(
    "changed,runs",
    [
        ("crosstl/translator/codegen/directx_codegen.py", []),
        ("tools/coverage/probes.py", ["metal", "compute", "rust"]),
        ("crosstl/backend/Metal/__init__.py", ["metal"]),
    ],
)
def test_the_merge_queue_leaves_out_the_d3d12_leg(tree, changed, runs):
    assert legs.plan([changed], tree, merge_group=True) == _legs(*runs)


def test_the_command_plans_a_merge_group(tree, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("tools/coverage/probes.py\n"))
    assert legs.main(["--merge-group", str(tree)]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "d3d12=false"


def test_every_leg_has_generators_in_crosstl_and_shares_the_parser():
    exclusive = legs.exclusive_modules(REPO)
    assert set(exclusive) == {name for name, leg in legs.LEGS.items() if leg.alone}
    assert all(modules is not None for modules in exclusive.values()), exclusive
    assert all(
        "crosstl/translator/parser.py" not in modules for modules in exclusive.values()
    )


def test_the_workflow_plans_every_leg_with_the_planner():
    workflow = yaml.safe_load(
        (REPO / ".github/workflows/coverage.yml").read_text(encoding="utf-8")
    )
    plan = workflow["jobs"]["plan"]
    assert set(plan["outputs"]) == set(legs.LEGS)
    (step,) = (step for step in plan["steps"] if step.get("id") == "plan")
    assert f"legs=({' '.join(legs.LEGS)})" in step["run"]
    assert "python3 -m tools.coverage.legs" in step["run"]
    assert '[ "$EVENT" = merge_group ]; then queue=(--merge-group)' in step["run"]


def test_every_d3d12_prefix_names_modules_in_crosstl():
    leg = legs.LEGS["d3d12"]
    for prefix in leg.generators + leg.frontends:
        assert list(REPO.glob(prefix + "*")), prefix
