"""Implicit WGSL resources use the lowest free slots in each bind group."""

import re
import shutil
import subprocess

import pytest

from crosstl.translator.codegen.wgsl_codegen import WGSLCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser


def generate(declarations, generator=None):
    source = f"""shader Bindings {{
        {declarations}
        compute {{
            layout(local_size_x = 1) in;
            void main() {{ return; }}
        }}
    }}"""
    return (generator or WGSLCodeGen()).generate(
        Parser(Lexer(source).get_tokens()).parse()
    )


def bindings(code):
    found = re.findall(
        r"@group\((\d+)\) @binding\((\d+)\)\s+var(?:<[^>]+>)?\s+(\w+)", code
    )
    assert len({(group, slot) for group, slot, _ in found}) == len(found)
    return {name: (int(group), int(slot)) for group, slot, name in found}


@pytest.mark.parametrize("group", [0, 2])
@pytest.mark.parametrize("slot", [0, 1, 3])
def test_explicit_cbuffer_does_not_advance_implicit_storage_bindings(group, slot):
    code = generate(f"""
        cbuffer Params @set({group}) @binding({slot}) {{ float scale; }};
        buffer float* fin;
        buffer float* outp;
    """)
    found = bindings(code)
    available = [n for n in range(4) if group != 0 or n != slot]
    assert found == {
        "_Params": (group, slot),
        "fin": (0, available[0]),
        "outp": (0, available[1]),
    }


@pytest.mark.parametrize(
    "explicit",
    [
        "uniform vec4 fixed @binding(8);",
        "sampler fixed @binding(8);",
        "sampler2D fixed @binding(8);",
        "buffer float fixed[4] @binding(8);",
        "image2D fixed @binding(8) @rgba8 @writeonly;",
    ],
)
def test_explicit_resource_and_synthetic_sampler_leave_lower_slots_free(explicit):
    found = bindings(generate(explicit + "\nbuffer float* fin;\nbuffer float* outp;"))
    assert found["fixed"] == (0, 8)
    assert found["fin"] == (0, 0)
    assert found["outp"] == (0, 1)
    if "fixed_sampler" in found:
        assert found["fixed_sampler"] == (0, 9)


def test_implicit_resources_fill_holes_around_later_explicit_reservations():
    found = bindings(generate("""
        uniform vec4 high @binding(8);
        buffer float* first;
        sampler occupied @binding(0);
        uniform vec4 second;
        buffer float explicitValues[4] @binding(2);
        buffer float* third;
    """))
    assert found == {
        "high": (0, 8),
        "first": (0, 1),
        "occupied": (0, 0),
        "second": (0, 3),
        "explicitValues": (0, 2),
        "third": (0, 4),
    }


@pytest.mark.parametrize("explicit", [False, True])
def test_texture_array_expansion_and_default_group_allocate_independently(explicit):
    attribute = "@binding(3)" if explicit else ""
    found = bindings(
        generate(f"sampler2D textures[2] @set(2) {attribute}; buffer float* values;")
    )
    start = 3 if explicit else 0
    assert found == {
        "textures_0": (2, start),
        "textures_0_sampler": (2, start + 1),
        "textures_1": (2, start + 2),
        "textures_1_sampler": (2, start + 3),
        "values": (0, 0),
    }


def test_implicit_binding_cursors_reset_when_reusing_the_generator():
    generator = WGSLCodeGen()
    generate("buffer float* first; buffer float* second;", generator)
    assert bindings(generate("buffer float* next;", generator)) == {"next": (0, 0)}


def test_uniform_buffer_probe_keeps_storage_slots_and_validates_with_naga(tmp_path):
    from tools.coverage import probes
    from tools.coverage.model import Feature

    feature = Feature("wgsl", "resource", "uniform-buffer", None)
    source = probes.build(feature, "crossgl").source
    code = WGSLCodeGen().generate(Parser(Lexer(source).get_tokens()).parse())
    found = bindings(code)
    assert found["fin"] == (0, 0)
    assert found["outp"] == (0, 1)
    assert found["_Params"] == (0, 3)
    naga = shutil.which("naga")
    if naga is None:
        pytest.skip("naga is not installed")
    path = tmp_path / "probe.wgsl"
    path.write_text(code)
    result = subprocess.run(
        [naga, str(path)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
