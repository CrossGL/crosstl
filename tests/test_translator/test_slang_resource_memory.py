"""Memory qualifiers on Slang resource declarations (crosstl/translator/codegen/slang_resource_memory.py)."""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.slang_codegen import SlangCodeGen
from crosstl.translator.codegen.slang_refusals import SlangLoweringRefused
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

GLSL_STORAGE = """#version 450
layout(local_size_x = 1) in;
layout(r32ui, binding = 0) uniform volatile uimage2D img0;
layout(r32ui, binding = 1) uniform restrict uimage2D img1;
layout(r32ui, binding = 2) uniform writeonly uimage2D img2;
layout(r32ui, binding = 3) uniform coherent volatile restrict uimage2D img3;
layout(std430, binding = 4) coherent volatile restrict buffer A { uint a[]; };
layout(std430, binding = 5) volatile writeonly buffer B { uint b[]; };
layout(std430, binding = 6) coherent volatile buffer C { uint count; float data[]; } c;
layout(std430, binding = 7) restrict readonly buffer D { uint dcount; float ddata[]; } d;
void main() {
    imageStore(img0, ivec2(0), uvec4(imageLoad(img1, ivec2(0)).x));
    imageStore(img2, ivec2(0), imageLoad(img3, ivec2(0)));
    a[0] = c.count;
    b[0] = a[1];
    c.data[0] = d.ddata[0] + float(d.dcount);
}
"""


def generate(code):
    return SlangCodeGen().generate(Parser(Lexer(code).get_tokens()).parse())


def refusal_reasons(code):
    with pytest.raises(SlangLoweringRefused) as refused:
        generate(code)
    return refused.value.reasons


def declaration(code, name):
    """The module-scope declaration of ``name`` in Slang ``code``."""
    pattern = re.compile(
        rf"^\S.*\b{re.escape(name)}(?:\[\d+\])? : register\(\w+\);$", re.M
    )
    matches = pattern.findall(code)
    assert len(matches) == 1, (name, matches)
    return matches[0]


def translate_file(tmp_path, name, source):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="slang")


def test_glsl_storage_images_and_blocks_keep_memory_qualifiers(tmp_path):
    code = translate_file(tmp_path, "storage.comp", GLSL_STORAGE)

    expected = {
        "img0": '[format("r32ui")] volatile RWTexture2D<uint> img0',
        "img1": '[format("r32ui")] restrict RWTexture2D<uint> img1',
        "img2": '[format("r32ui")] writeonly RWTexture2D<uint> img2',
        "img3": (
            '[format("r32ui")] globallycoherent volatile restrict RWTexture2D<uint> img3'
        ),
        # Storage blocks lowered to byte-address buffers lost every qualifier, coherence included.
        "a_2": "globallycoherent volatile restrict RWByteAddressBuffer a_2",
        "b_2": "volatile writeonly RWByteAddressBuffer b_2",
        "c": "globallycoherent volatile RWByteAddressBuffer c",
        "d": "restrict ByteAddressBuffer d",
    }
    for name, text in expected.items():
        assert text in declaration(code, name)


def test_crossgl_resources_keep_memory_modifiers():
    code = generate("""
    shader Resources {
        uimage2D readImage @r32ui @readonly @volatile;
        buffer uint* storage @volatile @restrict;
        buffer uint* readStorage @readonly @coherent;
        RWStructuredBuffer<uint> written @writeonly @volatile;
        StructuredBuffer<uint> shared @volatile;
        RWByteAddressBuffer raw @restrict @coherent(device);
        RWStructuredBuffer<uint> array[2] @volatile;
        AppendStructuredBuffer<uint> queue @writeonly @volatile;
        compute {
            void main() {
                uint x = imageLoad(readImage, ivec2(0)).x;
                storage[0] = readStorage[0] + shared[0] + x;
                written[0] = 1u;
                raw.Store(0, 1u);
                array[1][0] = 2u;
                queue.Append(3u);
            }
        }
    }
    """)

    expected = {
        "readImage": '[format("r32ui")] volatile readonly RWTexture2D<uint> readImage',
        "storage": "volatile restrict RWStructuredBuffer<uint> storage",
        # A read-only type implies readonly; coherence and volatility still apply to its reads.
        "readStorage": "globallycoherent StructuredBuffer<uint> readStorage",
        "written": "volatile writeonly RWStructuredBuffer<uint> written",
        "shared": "volatile StructuredBuffer<uint> shared",
        "raw": "globallycoherent restrict RWByteAddressBuffer raw",
        "array": "volatile RWStructuredBuffer<uint> array[2]",
        # An append buffer implies its access.
        "queue": "volatile AppendStructuredBuffer<uint> queue",
    }
    for name, text in expected.items():
        assert text in declaration(code, name)


def test_resource_passed_to_function_leaves_off_access_promises():
    code = generate("""
    shader Helpers {
        uimage2D written @r32ui @writeonly @restrict;
        uimage2D kept @r32ui @writeonly;
        void store(uimage2D image @r32ui, uint value) {
            imageStore(image, ivec2(0, 0), uvec4(value));
        }
        compute {
            void main() {
                store(written, 1u);
                imageStore(kept, ivec2(1, 1), uvec4(2u));
            }
        }
    }
    """)

    # slangc rejects a writeonly argument for a parameter without the qualifier (E30048); restrict is accepted.
    assert '[format("r32ui")] restrict RWTexture2D<uint> written' in declaration(
        code, "written"
    )
    assert '[format("r32ui")] writeonly RWTexture2D<uint> kept' in declaration(
        code, "kept"
    )
    assert "void store(RWTexture2D<uint> image, uint value)" in code


@pytest.mark.parametrize(
    "qualifier, word", [("@coherent", "coherent"), ("@volatile", "volatile")]
)
def test_coherent_or_volatile_resource_passed_to_function_is_refused(qualifier, word):
    reasons = refusal_reasons(f"""
    shader Helpers {{
        RWStructuredBuffer<uint> values {qualifier};
        void store(RWStructuredBuffer<uint> target, uint value) {{
            target[0] = value;
        }}
        compute {{
            void main() {{
                store(values, 1u);
            }}
        }}
    }}
    """)

    assert reasons == (
        f"resource 'values' is {word} and is passed to a function: slangc rejects an argument whose "
        "memory qualifiers the parameter lacks (E30048), and allows no coherent or volatile parameter "
        "(E31201)",
    )


@pytest.mark.parametrize(
    "declared",
    [
        "coherent(system) RWStructuredBuffer<uint> values;",
        "RWStructuredBuffer<uint> values @coherent(system);",
    ],
)
def test_system_scope_coherence_is_refused(declared):
    reasons = refusal_reasons(f"""
    shader SystemScope {{
        {declared}
        compute {{
            void main() {{
                values[0] = 1u;
            }}
        }}
    }}
    """)

    assert reasons == (
        "resource 'values' is coherent at system scope: Slang's globallycoherent makes a resource coherent "
        "across the device only, and Vulkan has no wider coherence",
    )


def test_unknown_coherence_scope_is_refused():
    reasons = refusal_reasons("""
    shader UnknownScope {
        RWStructuredBuffer<uint> values @coherent(subgroup);
        compute {
            void main() {
                values[0] = 1u;
            }
        }
    }
    """)

    assert reasons == (
        "resource 'values' declares coherent(subgroup), a coherence scope Slang has no modifier for",
    )


def test_metal_scoped_coherence_and_volatile_parameters_are_refused(tmp_path):
    source = """#include <metal_stdlib>
using namespace metal;
kernel void k(device coherent(device) uint* shared [[buffer(0)]],
              volatile device uint* flags [[buffer(1)]],
              device coherent(system) uint* host [[buffer(2)]],
              device uint* plain [[buffer(3)]],
              uint id [[thread_position_in_grid]]) {
    shared[id] = flags[id] + host[id] + plain[id];
}
"""

    with pytest.raises(SlangLoweringRefused) as refused:
        translate_file(tmp_path, "kernel.metal", source)

    assert refused.value.reasons == (
        "resource parameter 'shared' is coherent: slangc allows no coherent or volatile parameter (E31201), "
        "so its accesses would lose that behaviour",
        "resource parameter 'flags' is volatile: slangc allows no coherent or volatile parameter (E31201), "
        "so its accesses would lose that behaviour",
        "resource parameter 'host' is coherent at system scope: Slang's globallycoherent makes a resource "
        "coherent across the device only, and Vulkan has no wider coherence",
    )


def test_restrict_and_access_parameters_compile_without_modifiers():
    code = generate("""
    shader Parameters {
        void update(RWStructuredBuffer<uint> values @restrict, uimage2D image @r32ui @writeonly) {
            values[0] = 1u;
            imageStore(image, ivec2(0, 0), uvec4(2u));
        }
    }
    """)

    assert (
        "void update(RWStructuredBuffer<uint> values, RWTexture2D<uint> image)" in code
    )


@pytest.mark.skipif(shutil.which("slangc") is None, reason="slangc is not installed")
def test_slangc_decorates_glsl_storage_memory_qualifiers(tmp_path):
    code = translate_file(tmp_path, "storage.comp", GLSL_STORAGE)
    source = tmp_path / "storage.slang"
    source.write_text(code, encoding="utf-8")
    output = tmp_path / "storage.spvasm"
    result = subprocess.run(
        [
            shutil.which("slangc"),
            str(source),
            "-target",
            "spirv-asm",
            "-entry",
            "main",
            "-stage",
            "compute",
            "-o",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assembly = output.read_text(encoding="utf-8")
    decorations = set(
        re.findall(
            r"OpDecorate %(\w+) (Coherent|Volatile|Restrict|NonReadable|NonWritable)$",
            assembly,
            re.M,
        )
    )

    assert {
        ("img0", "Volatile"),
        ("img1", "Restrict"),
        ("img2", "NonReadable"),
        ("img3", "Coherent"),
        ("img3", "Volatile"),
        ("img3", "Restrict"),
        ("a_2", "Coherent"),
        ("a_2", "Volatile"),
        ("a_2", "Restrict"),
        ("b_2", "Volatile"),
        ("b_2", "NonReadable"),
        ("c", "Coherent"),
        ("c", "Volatile"),
        ("d", "Restrict"),
        ("d", "NonWritable"),
    } <= decorations
