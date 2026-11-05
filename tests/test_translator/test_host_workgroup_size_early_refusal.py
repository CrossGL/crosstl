"""A program with a kernel whose size only the host chooses is refused before any target code is generated.

Targets that fix the workgroup size in the shader refuse such a kernel when they reach its size. The translator checks
every entry before the semantics passes and code generation, so a library of many kernels is refused without
rewriting the whole program and generating the kernels declared in front of the one it cannot size.
"""

import pytest

import crosstl
import crosstl._crosstl as translator_module
from crosstl.translator.host_workgroup_size import (
    STATIC_WORKGROUP_SIZE_TARGETS,
    HostDefinedWorkgroupSizeError,
    refuse_host_defined_workgroup_sizes,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

HEADER = "#include <metal_stdlib>\nusing namespace metal;\n"

# ``scale`` runs one invocation per workgroup exactly; ``rotate`` shares threadgroup memory.
SCALE = """
kernel void scale(device float* data [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
    data[gid] = data[gid] * 2.0;
}
"""
ROTATE = """
kernel void rotate(device float* data [[buffer(0)]],
                   uint gid [[thread_position_in_grid]],
                   uint3 lid [[thread_position_in_threadgroup]]) {
    threadgroup float tile[256];
    tile[lid.x] = data[gid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    data[gid] = tile[(lid.x + 1) % 256];
}
"""
KERNELS = HEADER + SCALE + ROTATE

STAGE_BLOCK = """
shader main {
    struct FragmentOutput {
        vec4 color;
    }

    fragment {
        FragmentOutput main() {
            FragmentOutput output;
            output.color = vec4(1.0);
            return output;
        }
    }

    compute {
        @ host_workgroup_size
        @ stage_entry
        void rotate(RWStructuredBuffer<float> data @buffer(0), uint gid @gl_GlobalInvocationID) {
            threadgroup float[64] tile;
            tile[gid % 64u] = buffer_load(data, gid);
            workgroupBarrier();
            buffer_store(data, gid, tile[(gid + 1u) % 64u]);
        }
    }
}
"""


def _write(tmp_path, name, source):
    path = tmp_path / name
    path.write_text(source)
    return str(path)


@pytest.fixture
def no_code_generation(monkeypatch):
    def generate(*_args, **_kwargs):
        raise AssertionError("target code was generated")

    monkeypatch.setattr(translator_module, "_generate_target_code", generate)


@pytest.fixture
def no_semantics_passes(monkeypatch):
    def run(*_args, **_kwargs):
        raise AssertionError("the semantics passes ran")

    monkeypatch.setattr(translator_module, "apply_rule_carrying", run)


@pytest.mark.parametrize("backend", sorted(STATIC_WORKGROUP_SIZE_TARGETS))
def test_metal_kernel_library_is_refused_before_code_generation(
    tmp_path, no_code_generation, no_semantics_passes, backend
):
    with pytest.raises(HostDefinedWorkgroupSizeError) as raised:
        crosstl.translate(_write(tmp_path, "kernels.metal", KERNELS), backend=backend)
    assert raised.value.entry == "rotate"
    assert "threadgroup variable tile" in str(raised.value)


@pytest.mark.parametrize("backend", sorted(STATIC_WORKGROUP_SIZE_TARGETS))
def test_crossgl_stage_block_is_refused_before_code_generation(
    tmp_path, no_code_generation, no_semantics_passes, backend
):
    with pytest.raises(HostDefinedWorkgroupSizeError) as raised:
        crosstl.translate(_write(tmp_path, "rotate.cgl", STAGE_BLOCK), backend=backend)
    assert raised.value.entry == "rotate"


@pytest.mark.parametrize("backend", sorted(STATIC_WORKGROUP_SIZE_TARGETS))
def test_generated_refusal_is_the_one_the_early_check_gives(
    tmp_path, monkeypatch, backend
):
    """Without the early check, the generator reaches the same entry and gives the same diagnostic."""
    ast = Parser(Lexer(STAGE_BLOCK).get_tokens()).parse()
    with pytest.raises(HostDefinedWorkgroupSizeError) as early:
        refuse_host_defined_workgroup_sizes(ast, backend)
    monkeypatch.setattr(
        translator_module, "refuse_host_defined_workgroup_sizes", lambda *_: None
    )
    with pytest.raises(HostDefinedWorkgroupSizeError) as generated:
        crosstl.translate(_write(tmp_path, "rotate.cgl", STAGE_BLOCK), backend=backend)
    assert str(generated.value) == str(early.value)


@pytest.mark.parametrize("backend", sorted(STATIC_WORKGROUP_SIZE_TARGETS))
def test_single_entry_is_left_to_its_generator(tmp_path, monkeypatch, backend):
    """With nothing in front of the entry, the generator reports it, after the semantics passes."""
    single = HEADER + ROTATE
    compute_only = (
        STAGE_BLOCK[: STAGE_BLOCK.index("    fragment {")]
        + STAGE_BLOCK[STAGE_BLOCK.index("    compute {") :]
    )
    ast = Parser(Lexer(compute_only).get_tokens()).parse()
    assert len(ast.stages) == 1
    assert refuse_host_defined_workgroup_sizes(ast, backend) is None
    passes = []
    run = translator_module.apply_rule_carrying
    monkeypatch.setattr(
        translator_module,
        "apply_rule_carrying",
        lambda *args: passes.append(args[1:]) or run(*args),
    )
    with pytest.raises(HostDefinedWorkgroupSizeError) as raised:
        crosstl.translate(_write(tmp_path, "rotate.metal", single), backend=backend)
    assert raised.value.entry == "rotate"
    assert passes


@pytest.mark.parametrize("backend", sorted(STATIC_WORKGROUP_SIZE_TARGETS))
def test_host_size_option_still_sizes_the_kernel(tmp_path, backend):
    code = crosstl.translate(
        _write(tmp_path, "rotate.metal", HEADER + ROTATE),
        backend=backend,
        source_options={"threadgroup_size": (64, 1, 1)},
    )
    assert "64" in code


@pytest.mark.parametrize("backend", ["metal", "cuda"])
def test_targets_the_host_sizes_are_not_refused(tmp_path, backend):
    assert crosstl.translate(
        _write(tmp_path, "rotate.cgl", STAGE_BLOCK), backend=backend
    )


def test_selected_entry_is_left_to_its_generator(tmp_path):
    """An entry-scoped translation of the kernel that needs no size is not refused for the other kernel."""
    code = crosstl.translate(
        _write(tmp_path, "kernels.metal", KERNELS),
        backend="directx",
        entry_point="scale",
    )
    assert "[numthreads(1, 1, 1)]" in code
    assert "rotate" not in code
