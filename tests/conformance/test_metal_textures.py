"""The textures and samplers the Metal harness binds (:mod:`tests.conformance.metal_textures`).

The parsing and slot checks run everywhere. The device checks need an Apple GPU and opt in with
CROSSTL_CONFORMANCE_NATIVE=1, as :mod:`tests.conformance.test_native_runtimes` does: a translation that samples the
textures the native probe samples is supported, and one that reads another component, another slot or another
sampler state is wrong.
"""

import os

import pytest

from . import metal_textures
from .interface import InterfaceChanged, interface
from .runtimes import RUNTIMES, provision

_KERNEL = """#include <metal_stdlib>
using namespace metal;

kernel void probe(device float* fin [[buffer(0)]], device float* outp [[buffer(1)]], uint3 id [[thread_position_in_grid]], {params}) {{
    uint i = id.x;
    {body}
}}
"""
_SAMPLED = "texture2d<float> tex [[texture(0)]], sampler smp [[sampler(0)]]"
_SAMPLE = "outp[i] = tex.sample(smp, float2(fin[i], 0.5), level(0.0)).x;"


def _kernel(params=_SAMPLED, body=_SAMPLE):
    return _KERNEL.format(params=params, body=body)


@pytest.mark.parametrize(
    "parameter, expected",
    [
        (
            "texture2d<float> tex [[texture(0)]]",
            {
                "kind": "texture",
                "slot": 0,
                "count": 1,
                "type": "texture2d",
                "element": "float",
                "access": "sample",
            },
        ),
        (
            "texture2d<float, access::read_write> img [[texture(3)]]",
            {
                "kind": "texture",
                "slot": 3,
                "count": 1,
                "type": "texture2d",
                "element": "float",
                "access": "read_write",
            },
        ),
        (
            "array<texture2d<float>, 2> tex [[texture(1)]]",
            {
                "kind": "texture",
                "slot": 1,
                "count": 2,
                "type": "texture2d",
                "element": "float",
                "access": "sample",
            },
        ),
        (
            "texturecube_array<half> cubes [[texture(0)]]",
            {
                "kind": "texture",
                "slot": 0,
                "count": 1,
                "type": "texturecube_array",
                "element": "half",
                "access": "sample",
            },
        ),
        (
            "texture_buffer<uint, access::read> texels [[texture(2)]]",
            {
                "kind": "texture",
                "slot": 2,
                "count": 1,
                "type": "texture_buffer",
                "element": "uint",
                "access": "read",
            },
        ),
        ("sampler smp [[sampler(0)]]", {"kind": "sampler", "slot": 0, "count": 1}),
        (
            "array<sampler, 2> texSampler [[sampler(4)]]",
            {"kind": "sampler", "slot": 4, "count": 2},
        ),
    ],
)
def test_declaration_reads_slot_count_type_and_access(parameter, expected):
    assert metal_textures.declaration(parameter) == expected


@pytest.mark.parametrize(
    "parameter",
    [
        "depth2d<float> shadow [[texture(0)]]",
        "texture2d_ms<float> samples [[texture(0)]]",
        "texture2d<bfloat> tex [[texture(0)]]",
        "texture2d<float, access::sample_compare> tex [[texture(0)]]",
        "device float* values [[buffer(0)]]",
    ],
)
def test_declaration_refuses_textures_the_harness_cannot_create(parameter):
    with pytest.raises(InterfaceChanged):
        metal_textures.declaration(parameter)


def test_resources_refuse_two_arguments_on_one_slot():
    records = [
        metal_textures.declaration("array<texture2d<float>, 2> a [[texture(0)]]"),
        metal_textures.declaration("texture2d<float> b [[texture(1)]]"),
    ]
    with pytest.raises(InterfaceChanged, match="share a slot"):
        metal_textures.resources(records)
    # A texture and a sampler on the same index use separate argument tables.
    assert metal_textures.resources(
        [records[1], metal_textures.declaration("sampler s [[sampler(1)]]")]
    )


def test_texels_depend_on_slot_and_slice_and_fill_every_texel():
    record = metal_textures.declaration("texturecube_array<float> t [[texture(0)]]")
    slices = metal_textures.texels(record, 0)
    # Two layers of six faces, 4 x 4 texels of four 32-bit components each.
    assert len(slices) == 12
    assert {len(data) for data in slices} == {4 * 4 * 4 * 4}
    assert len(set(slices)) == len(slices)
    assert metal_textures.texels(record, 0) == slices
    assert metal_textures.texels(record, 1) != slices


def test_interface_lists_textures_and_samplers_beside_the_buffers(tmp_path):
    path = tmp_path / "probe.metal"
    path.write_text(_kernel())
    found = interface(path, "metal", 1)
    assert found["bindings"] == (0, 1)
    assert [record["kind"] for record in found["resources"]] == ["sampler", "texture"]
    path.write_text(
        _kernel(
            params="constant float& scale [[buffer(2)]]",
            body="outp[i] = fin[i] * scale;",
        )
    )
    assert "resources" not in interface(path, "metal", 1, constants=True)


_SLOT = "struct Slot {\n    texture2d<float> tex;\n};\n"
_PAIRED = (
    "struct Pair {\n    texture2d<float> texture;\n    sampler sampler_state;\n};\n"
)
_ENCODED = "device const Slot* slots [[buffer(2)]], sampler smp [[sampler(0)]]"
_ENCODED_READ = (
    "uint k = uint(fin[0]) & 1u;\n"
    "    outp[i] = slots[k].tex.sample(smp, float2(fin[i], 0.5), level(0.0)).x;"
)
_PAIRED_PARAMS = "const device Pair* tex [[buffer(2)]]"
_PAIRED_READ = (
    "int k = int(fin[0]) & 1;\n"
    "    outp[i] = tex[k].texture.sample(tex[k].sampler_state, float2(fin[i], 0.5), level(0.0)).x;"
)


def _encoded(structs=_SLOT, params=_ENCODED, body=_ENCODED_READ):
    return _kernel(params=params, body=body).replace(
        "using namespace metal;\n", "using namespace metal;\n" + structs, 1
    )


def test_interface_reads_an_argument_buffer_of_textures_and_samplers(tmp_path):
    path = tmp_path / "probe.metal"
    path.write_text(_encoded())
    found = interface(path, "metal", 1)
    assert found["bindings"] == (0, 1)
    encoded = next(r for r in found["resources"] if r["kind"] == "argument_buffer")
    assert encoded["slot"] == 2 and encoded["count"] == 4
    assert [member["kind"] for member in encoded["members"]] == ["texture"]
    path.write_text(_encoded(_PAIRED, _PAIRED_PARAMS, _PAIRED_READ))
    [encoded] = interface(path, "metal", 1)["resources"]
    assert [member["kind"] for member in encoded["members"]] == ["texture", "sampler"]


def test_a_buffer_of_a_structure_with_data_members_stays_a_storage_buffer(tmp_path):
    path = tmp_path / "probe.metal"
    path.write_text(
        _encoded(
            "struct Slot {\n    texture2d<float> tex;\n    float scale;\n};\n",
            "device const Slot* slots [[buffer(2)]]",
            "outp[i] = slots[0].scale;",
        )
    )
    with pytest.raises(InterfaceChanged, match="buffer slots"):
        interface(path, "metal", 1)


def test_an_argument_buffer_member_with_its_own_index_is_refused(tmp_path):
    path = tmp_path / "probe.metal"
    path.write_text(
        _encoded("struct Slot {\n    texture2d<float> tex [[id(3)]];\n};\n")
    )
    with pytest.raises(InterfaceChanged, match="sets its own index"):
        interface(path, "metal", 1)


def test_argument_buffer_textures_hold_the_texels_of_consecutive_slots():
    texture = {"kind": "texture", "count": 1, "type": "texture2d"}
    sampler = {"kind": "sampler", "count": 1}
    record = {"kind": "argument_buffer", "slot": 2, "count": 2}
    layout = metal_textures.argument_buffer_layout(
        {**record, "members": [texture, sampler]}
    )
    assert [(kind, slot) for kind, _, slot in layout] == [
        ("texture", 0),
        ("sampler", -1),
        ("texture", 1),
        ("sampler", -1),
    ]
    two = {**texture, "count": 2}
    layout = metal_textures.argument_buffer_layout({**record, "members": [two]})
    assert [slot for _, _, slot in layout] == [0, 1, 2, 3]


_NATIVE = os.environ.get("CROSSTL_CONFORMANCE_NATIVE") == "1"


@pytest.mark.skipif(
    not _NATIVE,
    reason="set CROSSTL_CONFORMANCE_NATIVE=1 to execute on native runtimes",
)
@pytest.mark.parametrize(
    "translated, expected",
    [
        (_kernel(), "supported"),
        (_kernel(body=_SAMPLE.replace(").x;", ").y;")), "wrong"),
        (
            _kernel(
                params="texture2d<float> tex [[texture(1)]], sampler smp [[sampler(0)]]"
            ),
            "wrong",
        ),
        (
            _kernel(
                params="texture2d<float> tex [[texture(0)]]",
                body="constexpr sampler smp(mag_filter::linear, min_filter::linear);\n    "
                + _SAMPLE,
            ),
            "wrong",
        ),
    ],
    ids=["same", "component", "slot", "inline-sampler"],
)
def test_coverage_compares_texture_reads_on_the_apple_gpu(translated, expected):
    from tools.coverage import execution
    from tools.coverage.model import Feature, Result

    runtime = RUNTIMES["metal-apple"]
    provided = provision(runtime)
    if not provided.available:
        pytest.skip(provided.reason)
    feature = Feature("metal", "construct", "sampled-texture", None)
    [result], used = execution.run(
        [Result(feature, "output", "supported", sources=(_kernel(), translated))],
        runtimes=runtime.id,
    )
    assert used == [runtime.id]
    assert result.outcome == expected, result


@pytest.mark.skipif(
    not _NATIVE,
    reason="set CROSSTL_CONFORMANCE_NATIVE=1 to execute on native runtimes",
)
@pytest.mark.parametrize(
    "translated, expected",
    [
        (_encoded(), "supported"),
        (_encoded(_PAIRED, _PAIRED_PARAMS, _PAIRED_READ), "supported"),
        (_encoded(body=_ENCODED_READ.replace("slots[k]", "slots[k ^ 1u]")), "wrong"),
        (
            _kernel(
                params="array<texture2d<float>, 2> tex [[texture(0)]], sampler smp [[sampler(0)]]",
                body=_ENCODED_READ.replace("slots[k].tex", "tex[k]"),
            ),
            "supported",
        ),
    ],
    ids=["same", "texture-and-sampler-members", "other-element", "texture-array"],
)
def test_coverage_compares_argument_buffer_reads_on_the_apple_gpu(translated, expected):
    from tools.coverage import execution
    from tools.coverage.model import Feature, Result

    runtime = RUNTIMES["metal-apple"]
    provided = provision(runtime)
    if not provided.available:
        pytest.skip(provided.reason)
    feature = Feature("metal", "construct", "unsized-array-of-textures", None)
    [result], used = execution.run(
        [Result(feature, "output", "supported", sources=(_encoded(), translated))],
        runtimes=runtime.id,
    )
    assert used == [runtime.id]
    assert result.outcome == expected, result
