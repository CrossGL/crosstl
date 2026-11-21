"""The Metal bare type-name memo reuses a spelling only while the state it read holds, and repeats its effects."""

import pytest

from crosstl.translator import parse
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.codegen.metal_type_name_memo import memoized_bare_metal_type


def _uncached(generator, name):
    return generator.map_type(name, memoized=False)


def test_memo_returns_the_uncached_spelling():
    generator = MetalCodeGen()
    reference = MetalCodeGen()
    for name in ("vec3", "float16_t", "int", "uint16_t4", "bfloat16_t2", "Payload"):
        for _ in range(2):
            assert generator.map_type(name) == _uncached(reference, name)


def test_memo_hits_do_not_rerun_the_mapping(monkeypatch):
    generator = MetalCodeGen()
    generator.map_type("vec4")
    calls = []
    original = MetalCodeGen.map_type

    def counting(self, vtype, *, memoized=True):
        if not memoized:
            calls.append(vtype)
        return original(self, vtype, memoized=memoized)

    monkeypatch.setattr(MetalCodeGen, "map_type", counting)
    for _ in range(5):
        assert generator.map_type("vec4") == "float4"
    assert calls == []


def _alias(generator):
    generator.metal_type_aliases = {"Payload": "vec2"}


def _type_table(generator):
    generator.type_mapping["Payload"] = "float3"


def _enum(generator):
    generator.enum_type_names = {"Payload"}


def _enum_struct(generator):
    generator.enum_struct_type_names = {"Payload"}


@pytest.mark.parametrize(
    "declare,expected",
    [
        (_alias, "float2"),
        (_type_table, "float3"),
        (_enum, "int"),
        (_enum_struct, "Payload"),
    ],
)
def test_memo_follows_declarations_made_after_the_first_lookup(declare, expected):
    generator = MetalCodeGen()
    assert generator.map_type("Payload") == "Payload"
    declare(generator)
    assert generator.map_type("Payload") == expected


def test_memo_follows_the_names_a_mapping_maps_in_turn():
    # ``float16_t2`` maps through its sized spelling; a changed table entry of
    # any name the mapping read invalidates the entry.
    generator = MetalCodeGen()
    first = generator.map_type("float16_t2")
    entry = generator._metal_bare_type_memo["float16_t2"]
    read = [name for name, _ in entry[2]]
    assert read[0] == "float16_t2"
    for name in read:
        generator.type_mapping[name] = "Changed"
        assert generator.map_type("float16_t2") == _uncached(generator, "float16_t2")
        generator.type_mapping = MetalCodeGen().type_mapping
        assert generator.map_type("float16_t2") == first


def test_memo_replays_emulated_matrix_shapes_on_every_hit():
    generator = MetalCodeGen()
    name = "int2x3"
    spelling = _uncached(MetalCodeGen(), name)
    reference = MetalCodeGen()
    _uncached(reference, name)
    shapes = set(reference.metal_emulated_matrix_shapes)
    assert shapes, "the emulated shape this test replays"
    assert generator.map_type(name) == spelling
    assert generator.metal_emulated_matrix_shapes == shapes
    generator.metal_emulated_matrix_shapes = set()
    assert generator.map_type(name) == spelling
    assert generator.metal_emulated_matrix_shapes == shapes


def test_memo_records_foreign_type_names_in_each_rendering():
    generator = MetalCodeGen()
    reference = MetalCodeGen()
    reference.written_foreign_type_names = set()
    _uncached(reference, "HitObject")
    foreign = set(reference.written_foreign_type_names)
    assert foreign, "the foreign built-in type this test replays"
    for _ in range(2):
        generator.written_foreign_type_names = set()
        generator.map_type("HitObject")
        generator.map_type("HitObject")
        assert generator.written_foreign_type_names == foreign
    generator.written_foreign_type_names = None
    generator.map_type("HitObject")
    assert generator.written_foreign_type_names is None


def test_memo_is_bypassed_for_declared_type_names_of_a_new_program():
    generator = MetalCodeGen()
    generator.written_foreign_type_names = set()
    generator.map_type("HitObject")
    assert generator.written_foreign_type_names
    generator.program_type_names = frozenset({"HitObject"})
    generator.written_foreign_type_names = set()
    generator.map_type("HitObject")
    assert generator.written_foreign_type_names == set()


@pytest.mark.parametrize("name", ["RayQuery", "RayDesc", "visible_function_table"])
def test_names_whose_mappings_set_runtime_requirements_are_never_memoized(name):
    generator = MetalCodeGen()
    generator.map_type(name)
    assert generator._metal_bare_type_memo[name] is not None
    assert not isinstance(generator._metal_bare_type_memo[name], tuple)


def test_ray_description_requirement_is_raised_on_every_lookup():
    generator = MetalCodeGen()
    assert generator.map_type("RayDesc") == "CglRayDesc"
    generator.required_metal_ray_desc_runtime = False
    assert generator.map_type("RayDesc") == "CglRayDesc"
    assert generator.required_metal_ray_desc_runtime


def test_a_mapping_that_records_a_refusal_is_not_memoized(monkeypatch):
    generator = MetalCodeGen()
    original = MetalCodeGen.map_type

    def refusing(self, vtype, *, memoized=True):
        if not memoized and vtype == "Payload":
            self.refuse_lowering("Payload has no Metal spelling")
        return original(self, vtype, memoized=memoized)

    monkeypatch.setattr(MetalCodeGen, "map_type", refusing)
    for _ in range(2):
        generator.refusals.reasons = []
        generator.map_type("Payload")
        assert generator.refusals.reasons == ["Payload has no Metal spelling"]
    assert "Payload" not in generator._metal_bare_type_memo or not isinstance(
        generator._metal_bare_type_memo["Payload"], tuple
    )


def test_a_mapping_that_changes_generator_state_is_not_memoized(monkeypatch):
    generator = MetalCodeGen()
    original = MetalCodeGen.map_type

    def recording(self, vtype, *, memoized=True):
        if not memoized and vtype == "Payload":
            self.payload_mappings = getattr(self, "payload_mappings", 0) + 1
        return original(self, vtype, memoized=memoized)

    monkeypatch.setattr(MetalCodeGen, "map_type", recording)
    for _ in range(3):
        generator.map_type("Payload")
    assert generator.payload_mappings == 3


def test_an_outer_mapping_that_reads_an_unmemoized_mapping_is_not_memoized(
    monkeypatch,
):
    # A mapping that maps a declared name (or any type node) reads state the
    # entry would not record, so neither it nor its callers are stored.
    generator = MetalCodeGen()
    generator.metal_type_aliases = {"Inner": "vec2"}
    original = MetalCodeGen.map_type

    def outer(self, vtype, *, memoized=True):
        if not memoized and vtype == "Outer":
            return self.map_type("Inner")
        return original(self, vtype, memoized=memoized)

    monkeypatch.setattr(MetalCodeGen, "map_type", outer)
    assert generator.map_type("Outer") == "float2"
    generator.metal_type_aliases = {"Inner": "vec3"}
    assert generator.map_type("Outer") == "float3"


def test_a_mapping_that_raises_is_not_memoized(monkeypatch):
    generator = MetalCodeGen()
    original = MetalCodeGen.map_type
    attempts = []

    def failing(self, vtype, *, memoized=True):
        if not memoized and vtype == "Payload":
            attempts.append(vtype)
            raise ValueError("no spelling")
        return original(self, vtype, memoized=memoized)

    monkeypatch.setattr(MetalCodeGen, "map_type", failing)
    for _ in range(2):
        with pytest.raises(ValueError):
            generator.map_type("Payload")
    assert attempts == ["Payload", "Payload"]
    assert "_metal_bare_type_misses" not in generator.__dict__


def test_rendering_twice_with_one_generator_matches_fresh_generators():
    source = """
    shader Memo {
        struct Light { vec3 color; float range; };
        compute {
            layout(std430, binding = 0) buffer Out { vec4 values[]; } outBuffer;
            void main() {
                Light light;
                light.color = vec3(1.0, 0.5, 0.25);
                light.range = 2.0;
                vec4 value = vec4(light.color * light.range, 1.0);
                ivec2 cell = ivec2(3, 4);
                outBuffer.values[0] = value + vec4(vec2(cell), 0.0, 0.0);
            }
        }
    }
    """
    shared = MetalCodeGen()
    first = shared.generate(parse(source))
    second = shared.generate(parse(source))
    assert first == second == MetalCodeGen().generate(parse(source))
    assert shared._metal_bare_type_memo


def test_memo_entry_point_is_the_bare_name_path():
    generator = MetalCodeGen()
    assert memoized_bare_metal_type(generator, "vec2") == "float2"
    assert generator.map_type("vec2") == "float2"
