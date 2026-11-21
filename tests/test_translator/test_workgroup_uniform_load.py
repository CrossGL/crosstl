"""CrossGL's ``workgroupUniformLoad`` reads workgroup memory between two workgroup barriers.

WGSL output keeps the call; every other target calls a CrossGL helper that ``crosstl/translator/semantics/
workgroup_uniform_load.py`` adds: ``workgroupBarrier(); T value = x; workgroupBarrier(); return value;``.
``tests/conformance/cases/x_workgroup_uniform_load`` executes the helpers; these tests pin the rewriting and its
refusals.
"""

from __future__ import annotations

import re

import pytest

import crosstl
from crosstl.translator.semantics.builtin_function_names import BUILTIN_FUNCTION_NAMES
from crosstl.translator.semantics.workgroup_uniform_load import (
    FUNCTION,
    PREFIX,
    WorkgroupUniformLoadError,
)

LOWERED_TARGETS = ("opengl", "vulkan", "directx", "metal", "cuda", "hip", "slang")
_BARRIERS = {
    "opengl": "barrier();",
    "directx": "GroupMemoryBarrierWithGroupSync();",
    "slang": "GroupMemoryBarrierWithGroupSync();",
    "metal": "threadgroup_barrier(mem_flags::mem_threadgroup);",
    "cuda": "__syncthreads();",
    "hip": "__syncthreads();",
}


def _program(body, declarations="shared float a0;", structs=""):
    return f"""
shader UniformLoad {{
    {structs}
    compute {{
        layout(local_size_x = 4, local_size_y = 1, local_size_z = 1) in;
        {declarations}
        buffer float* outp;
        void main() {{
            uint lane = gl_LocalInvocationID.x;
            {body}
        }}
    }}
}}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "uniform_load.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _helper_body(code, name):
    """The text of helper ``name``'s definition up to its ``return``: the occurrence a ``{`` follows before any ``;``."""
    for match in re.finditer(rf"\b{name}\(", code):
        rest = code[match.end() :]
        brace, semicolon = rest.find("{"), rest.find(";")
        if brace != -1 and (semicolon == -1 or brace < semicolon):
            return rest[: rest.index("return")]
    raise AssertionError(f"no definition of {name}")


def test_workgroup_uniform_load_is_a_crossgl_operation():
    assert FUNCTION in BUILTIN_FUNCTION_NAMES


@pytest.mark.parametrize("backend", ["cgl", "wgsl"])
def test_crossgl_and_wgsl_keep_the_call(tmp_path, backend):
    code = _translate(
        tmp_path, _program("outp[lane] = workgroupUniformLoad(a0);"), backend
    )

    assert PREFIX not in code
    expected = "workgroupUniformLoad(&a0)" if backend == "wgsl" else FUNCTION + "(a0)"
    assert expected in code


@pytest.mark.parametrize("backend", sorted(_BARRIERS))
def test_other_targets_read_between_two_workgroup_barriers(tmp_path, backend):
    code = _translate(
        tmp_path, _program("outp[lane] = workgroupUniformLoad(a0);"), backend
    )

    assert FUNCTION not in code
    helper = PREFIX + "0_a0"
    assert re.search(rf"\b{helper}\(", code)
    body = _helper_body(code, helper)
    barrier = _BARRIERS[backend]
    first = body.index(barrier)
    read = body.index(PREFIX + "value", first)
    assert body.index(barrier, read) > read


def test_spirv_reads_between_two_workgroup_memory_barriers(tmp_path):
    path = tmp_path / "uniform_load.cgl"
    path.write_text(
        _program("outp[lane] = workgroupUniformLoad(a0);"), encoding="utf-8"
    )
    # The formatted module names its constants (%uint_2, %uint_264).
    code = crosstl.translate(str(path), backend="vulkan")

    # Execution and memory scope Workgroup (2); semantics AcquireRelease | WorkgroupMemory (0x108).
    barriers = re.findall(r"OpControlBarrier %uint_2 %uint_2 %uint_264", code)
    assert len(barriers) == 2
    assert FUNCTION not in code


_PATHS = """
cells[lane].v = float(lane);
cells[lane].k[1] = lane;
float first = workgroupUniformLoad(cells[2].v);
uint k = workgroupUniformLoad(cells[int(lane % 2u) + 1].k[1]);
float again = workgroupUniformLoad(cells[3].v);
outp[lane] = first + float(k) + again;
"""
_CELL = "struct Cell { float v; uint k[2]; };"


@pytest.mark.parametrize("backend", ["opengl", "directx", "metal"])
def test_indices_become_helper_arguments_and_equal_paths_share_a_helper(
    tmp_path, backend
):
    code = _translate(
        tmp_path, _program(_PATHS, "shared Cell cells[4];", _CELL), backend
    )

    helpers = sorted(set(re.findall(rf"\b{PREFIX}\d+_cells\b", code)))
    # cells[i].v serves both of its calls; cells[i].k[j] has a helper of its own.
    assert helpers == [PREFIX + "0_cells", PREFIX + "1_cells"]
    assert re.search(rf"{PREFIX}1_cells\(\(?int\(\(?lane % 2u\)?\)\)? \+ 1\)?, 1", code)
    assert re.search(rf"{PREFIX}0_cells\(3[,)]", code)


@pytest.mark.parametrize("backend", ["metal", "directx", "opengl"])
def test_a_variable_updated_atomically_is_read_by_a_plain_load(tmp_path, backend):
    # The barriers order the load after the atomic updates, so it needs no atomic operation of its own.
    body = """
if (lane == 0u) { counter = 0u; }
workgroupBarrier();
atomicAdd(counter, 1u);
outp[lane] = float(workgroupUniformLoad(counter));
"""
    code = _translate(tmp_path, _program(body, "shared uint counter;"), backend)

    helper = _helper_body(code, PREFIX + "0_counter")
    assert re.search(rf"{PREFIX}value = counter;", helper)
    assert "atomic" not in helper.lower() and "Interlocked" not in helper


def test_a_pointer_operand_reads_the_same_memory(tmp_path):
    code = _translate(
        tmp_path, _program("outp[lane] = workgroupUniformLoad(&a0);"), "opengl"
    )

    assert re.search(rf"= a0;", _helper_body(code, PREFIX + "0_a0"))


def test_a_program_function_of_that_name_is_called_as_written(tmp_path):
    source = _program("outp[lane] = workgroupUniformLoad(a0);").replace(
        "void main()",
        "float workgroupUniformLoad(float x) { return x + 1.0; }\n        void main()",
    )

    code = _translate(tmp_path, source, "opengl")

    assert PREFIX not in code
    assert "workgroupUniformLoad(a0)" in code


@pytest.mark.parametrize(
    ("body", "declarations", "message"),
    [
        (
            "float local = 1.0; outp[lane] = workgroupUniformLoad(local);",
            "shared float a0;",
            "not a shared variable",
        ),
        (
            "outp[lane] = workgroupUniformLoad(a0 + 1.0);",
            "shared float a0;",
            "must be a shared variable",
        ),
        (
            "outp[lane] = workgroupUniformLoad(v.x);",
            "shared vec4 v;",
            "cannot be determined",
        ),
        (
            "outp[lane] = workgroupUniformLoad(a0, a0);",
            "shared float a0;",
            "takes 1 operand",
        ),
        (
            "outp[lane] = workgroupUniformLoad(buf[0]);",
            "buffer float* buf;",
            "not a shared variable",
        ),
    ],
)
@pytest.mark.parametrize("backend", ["opengl", "metal"])
def test_unsupported_operands_are_refused(
    tmp_path, body, declarations, message, backend
):
    with pytest.raises(WorkgroupUniformLoadError, match=message):
        _translate(tmp_path, _program(body, declarations), backend)


def test_a_shadowing_local_is_not_the_shared_variable(tmp_path):
    body = "float a0 = 2.0; outp[lane] = workgroupUniformLoad(a0);"

    with pytest.raises(WorkgroupUniformLoadError, match="not a shared variable"):
        _translate(tmp_path, _program(body), "directx")


def test_the_helper_prefix_is_reserved(tmp_path):
    source = _program("outp[lane] = workgroupUniformLoad(a0);").replace(
        "void main()",
        f"float {PREFIX}0_a0() {{ return 0.0; }}\n        void main()",
    )

    with pytest.raises(WorkgroupUniformLoadError, match="reserved"):
        _translate(tmp_path, source, "opengl")
