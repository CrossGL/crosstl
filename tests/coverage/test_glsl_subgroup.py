"""The GLSL subgroup extensions' functions in the GLSL list, and the non-uniform instructions they give the SPIR-V
list."""

import json

from tools.coverage import probes
from tools.coverage.features import glsl_subgroup, spirv
from tools.coverage.model import Type

# Excerpts of GL_KHR_shader_subgroup.txt, keeping its layout: the mapping to SPIR-V and the clusterSize rule in the
# overview, then the Syntax blocks of section 8.18. One block is indented by three spaces and one names its extension
# on two lines, as in the specification.
SUBGROUP = """\
Overview

      subgroupElect() -> OpGroupNonUniformElect(
        /*Execution*/Subgroup)

      subgroupBroadcast(value, id) -> OpGroupNonUniformBroadcast(
        /*Execution*/Subgroup,
        /*Value*/value,
        /*Id*/id)

      subgroupAdd(value) -> OpGroupNonUniformIAdd | OpGroupNonUniformFAdd(
        /*Execution*/Subgroup,
        /*Operation*/Reduce,
        /*Value*/value)

      subgroupAnd(value) -> OpGroupNonUniformBitwiseAnd | OpGroupNonUniformLogicalAnd(
        /*Execution*/Subgroup,
        /*Operation*/Reduce,
        /*Value*/value)

      subgroupExclusiveAdd(value) -> OpGroupNonUniformIAdd | OpGroupNonUniformFAdd(
        /*Execution*/Subgroup,
        /*Operation*/ExclusiveScan,
        /*Value*/value)

        The <clusterSize> as used in the subgroupClustered<op>() operations must
        be:

        - An integral constant expression.
        - At least 1.

Additions to Chapter 8 of the OpenGL Shading Language Specification
(Built-in Functions)

    Syntax:

        void subgroupBarrier(void);

    Only usable if the extension GL_KHR_shader_subgroup_basic is enabled.

    Syntax:

        bool subgroupElect(void);

    Only usable if the extension GL_KHR_shader_subgroup_basic is enabled.

    Syntax:

        genType  subgroupBroadcast(genType value,  uint id);
        genIType subgroupBroadcast(genIType value, uint id);

    Only usable if the extension GL_KHR_shader_subgroup_ballot is enabled.

    The function subgroupBroadcast() returns the <value> from the invocation
    whose <gl_SubgroupInvocationID> is equal to <id>. <id> must be an integral
    constant expression when targeting SPIR-V 1.4 and below.

    Syntax:

        genType  subgroupShuffleUp(genType value,  uint delta);

    Only usable if the extension GL_KHR_shader_subgroup_shuffle_relative is
    enabled.

    Syntax:

        genType  subgroupShuffle(genType value,  uint id);

    Only usable if the extension GL_KHR_shader_subgroup_shuffle is enabled.

    Syntax:

        genType  subgroupAdd(genType value);
        genIType subgroupAdd(genIType value);

    Only usable if the extension GL_KHR_shader_subgroup_arithmetic is enabled.

    Syntax:

        genIType subgroupAnd(genIType value);
        genUType subgroupAnd(genUType value);
        genBType subgroupAnd(genBType value);

    Only usable if the extension GL_KHR_shader_subgroup_arithmetic is enabled.

   Syntax:

        genType  subgroupExclusiveAdd(genType value);

    Only usable if the extension GL_KHR_shader_subgroup_arithmetic is enabled.

    Syntax:

        genType  subgroupClusteredAdd(genType value,  uint clusterSize);

    Only usable if the extension GL_KHR_shader_subgroup_clustered is enabled.
"""
GRAMMAR = json.dumps(
    {
        "instructions": [
            {"opname": name, "class": "Non-Uniform"}
            for name in (
                "OpGroupNonUniformElect",
                "OpGroupNonUniformBroadcast",
                "OpGroupNonUniformIAdd",
                "OpGroupNonUniformFAdd",
                "OpGroupNonUniformBitwiseAnd",
                "OpGroupNonUniformLogicalAnd",
                "OpGroupNonUniformQuadAllKHR",
            )
        ]
    }
)
VEC3 = Type("float", (3,))


def test_every_syntax_block_is_a_function_enabled_by_its_extension():
    functions = glsl_subgroup.functions(SUBGROUP)
    assert sorted(functions) == [
        "subgroupAdd",
        "subgroupAnd",
        "subgroupBarrier",
        "subgroupBroadcast",
        "subgroupClusteredAdd",
        "subgroupElect",
        "subgroupExclusiveAdd",
        "subgroupShuffle",
        "subgroupShuffleUp",
    ]
    assert (
        functions["subgroupShuffleUp"][0] == "GL_KHR_shader_subgroup_shuffle_relative"
    )
    assert functions["subgroupExclusiveAdd"][0] == "GL_KHR_shader_subgroup_arithmetic"
    # genType is a float vector, as GLSL 4.60's genFType is.
    assert functions["subgroupAdd"][1][0].result == VEC3
    assert functions["subgroupAdd"][1][1].result == Type("int", (3,))


def test_arguments_the_specification_requires_constant_are_passed_as_constants():
    functions = glsl_subgroup.functions(SUBGROUP)

    def directions(name):
        return [p.direction for p in functions[name][1][0].params]

    assert directions("subgroupBroadcast") == ["in", "const"]
    assert directions("subgroupClusteredAdd") == ["in", "const"]
    # subgroupShuffle's id may vary between invocations.
    assert directions("subgroupShuffle") == ["in", "in"]


def test_each_instruction_comes_from_the_first_function_listed_for_it():
    assert glsl_subgroup.instructions(SUBGROUP) == {
        "OpGroupNonUniformElect": ("subgroupElect", ""),
        "OpGroupNonUniformBroadcast": ("subgroupBroadcast", ""),
        "OpGroupNonUniformIAdd": ("subgroupAdd", "int"),
        "OpGroupNonUniformFAdd": ("subgroupAdd", "float"),
        "OpGroupNonUniformBitwiseAnd": ("subgroupAnd", "uint"),
        "OpGroupNonUniformLogicalAnd": ("subgroupAnd", "bool"),
    }


def test_glsl_features_name_their_extension_and_crossgl_spells_them_alike():
    features = {f.name: f for f in glsl_subgroup.parse(SUBGROUP)}
    add = features["subgroupAdd"]
    assert (add.group, add.extension, add.crossgl) == (
        "GL_KHR_shader_subgroup_arithmetic",
        "GL_KHR_shader_subgroup_arithmetic",
        "subgroupAdd",
    )
    native = probes.build(add, "glsl").source
    assert "#extension GL_KHR_shader_subgroup_arithmetic : require" in native
    assert native.index("#version 460") < native.index("#extension")
    assert "#extension" not in probes.build(add, "crossgl").source


def test_non_uniform_instructions_are_probed_with_the_function_that_compiles_to_them(
    monkeypatch,
):
    monkeypatch.setattr(spirv, "CORE", {})
    features = {f.name: f for f in spirv.core(GRAMMAR, "", SUBGROUP)}
    fadd, iadd = features["OpGroupNonUniformFAdd"], features["OpGroupNonUniformIAdd"]
    assert (fadd.spelling, fadd.signature.params[0].type) == ("subgroupAdd", VEC3)
    assert iadd.signature.params[0].type == Type("int", (3,))
    assert features["OpGroupNonUniformLogicalAnd"].signature.params[0].type == Type(
        "bool", (3,)
    )
    assert fadd.group == "core-non-uniform" and spirv.is_core(fadd)
    # The GLSL probe glslang compiles for the instruction enables the function's extension.
    assert fadd.extension == "GL_KHR_shader_subgroup_arithmetic"
    # A non-uniform instruction no function of the extension compiles to is listed, without a probe.
    quad = features["OpGroupNonUniformQuadAllKHR"]
    assert quad.signature is None
    assert quad.note == "no function of a GLSL subgroup extension compiles to it"


# Excerpts of GL_KHR_shader_subgroup_rotate.txt, GLSL_EXT_shader_quad_control.txt and
# GL_NV_shader_subgroup_partitioned.txt, keeping their layout. The rotate specification puts each instruction on the
# line after its function, the quad votes need two extensions, and the partitioned specification names
# OpGroupNonUniformPartitionNV, which the grammar lists as an alias of OpGroupNonUniformPartitionEXT.
ROTATE = """\
    Mapping to SPIR-V
    -----------------

      subgroupClusteredRotate(value, delta, clusterSize) ->
        Opcode: OpGroupNonUniformRotateKHR(
                    /*Execution*/Subgroup,
                    /*Value*/value,
                    /*Delta*/delta,
                    /*ClusterSize*/clusterSize)
        Capabilities: GroupNonUniformRotateKHR

Additions to Chapter 8 of the OpenGL Shading Language Specification
(Built-in Functions)

    Syntax:

        genType  subgroupClusteredRotate(genType value,  uint delta, uint clusterSize);
        genIType subgroupClusteredRotate(genIType value, uint delta, uint clusterSize);

        If GL_EXT_shader_subgroup_extended_types_int8 is enabled:

        genI8Type subgroupClusteredRotate(genI8Type value, uint delta, uint clusterSize);

    Only usable if the extension GL_KHR_shader_subgroup_rotate is enabled.
"""
QUAD = """\
      subgroupQuadAll(value) -> OpGroupNonUniformQuadAllKHR(
        /*Predicate*/value)

Additions to Chapter 8 of the OpenGL Shading Language Specification
(Built-in Functions)

    Syntax:

        bool subgroupQuadAll(bool value);

    Only usable if the extensions GL_KHR_shader_subgroup_vote and
    GL_EXT_shader_quad_control are enabled.
"""
PARTITIONED = """\
    When using GroupOperationPartitioned*, the <ballot> parameter is the last
    operand to OpGroupNonUniform<op>, similar to ClusterSize.

        subgroupPartitionNV -> OpGroupNonUniformPartitionNV

Additions to Chapter 8 of the OpenGL Shading Language Specification
(Built-in Functions)

    Syntax:

        uvec4 subgroupPartitionNV(genType value);
        uvec4 subgroupPartitionNV(genIType value);

    Only usable if the extension GL_NV_shader_subgroup_partitioned is enabled.
"""
EXTENSIONS = (SUBGROUP, ROTATE, QUAD, PARTITIONED)


def test_the_extensions_map_their_functions_in_their_own_layouts():
    assert glsl_subgroup.instructions(*EXTENSIONS) == glsl_subgroup.instructions(
        SUBGROUP
    ) | {
        "OpGroupNonUniformRotateKHR": ("subgroupClusteredRotate", ""),
        "OpGroupNonUniformQuadAllKHR": ("subgroupQuadAll", ""),
        "OpGroupNonUniformPartitionNV": ("subgroupPartitionNV", ""),
    }


def test_a_function_needing_two_extensions_enables_both_and_belongs_to_the_last():
    quad = {f.name: f for f in glsl_subgroup.parse(*EXTENSIONS)}["subgroupQuadAll"]
    assert quad.group == "GL_EXT_shader_quad_control"
    assert quad.extension == "GL_KHR_shader_subgroup_vote GL_EXT_shader_quad_control"
    native = probes.build(quad, "glsl").source
    assert (
        "#extension GL_KHR_shader_subgroup_vote : require\n"
        "#extension GL_EXT_shader_quad_control : require\n"
    ) in native


def test_the_cluster_size_rule_of_gl_khr_shader_subgroup_binds_the_clustered_rotation():
    rotate = glsl_subgroup.functions(*EXTENSIONS)["subgroupClusteredRotate"]
    assert rotate[0] == "GL_KHR_shader_subgroup_rotate"
    assert [p.direction for p in rotate[1][0].params] == ["in", "in", "const"]
    # Without GL_KHR_shader_subgroup's overview the rotation's specification states no such rule.
    alone = glsl_subgroup.functions(ROTATE)["subgroupClusteredRotate"]
    assert [p.direction for p in alone[1][0].params] == ["in", "in", "in"]


def test_an_instruction_named_by_an_alias_takes_the_grammar_name(monkeypatch):
    monkeypatch.setattr(spirv, "CORE", {})
    grammar = json.dumps(
        {
            "instructions": [
                {
                    "opname": "OpGroupNonUniformPartitionEXT",
                    "class": "Non-Uniform",
                    "aliases": ["OpGroupNonUniformPartitionNV"],
                },
                {"opname": "OpGroupNonUniformQuadAllKHR", "class": "Non-Uniform"},
                {"opname": "OpGroupNonUniformRotateKHR", "class": "Group"},
            ]
        }
    )
    features = {f.name: f for f in spirv.core(grammar, "", ROTATE, QUAD, PARTITIONED)}
    assert sorted(features) == [
        "OpGroupNonUniformPartitionEXT",
        "OpGroupNonUniformQuadAllKHR",
        "OpGroupNonUniformRotateKHR",
    ]
    partition = features["OpGroupNonUniformPartitionEXT"]
    assert (partition.spelling, partition.extension) == (
        "subgroupPartitionNV",
        "GL_NV_shader_subgroup_partitioned",
    )
    assert features["OpGroupNonUniformRotateKHR"].group == "core-group"
