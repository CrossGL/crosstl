from collections import Counter

import pytest

from tools.coverage import semantic, validators
from tools.coverage.model import Feature, Param, Signature, Type

VEC2 = Type("float", (2,))

ASSEMBLY = """
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
          %2 = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
               OpMemoryModel Logical GLSL450
          %3 = OpTypeVoid
          %4 = OpFunction %3 None %5
          %6 = OpLabel
          %7 = OpLoad %float %8
          %9 = OpExtInst %float %1 Fma %7 %7 %7
         %10 = OpExtInst %void %2 DebugLine %11
               OpStore %12 %9
               OpReturn
               OpFunctionEnd
"""


def glsl(name, result, *params):
    signature = Signature(result, tuple(Param(p) for p in params))
    return Feature("glsl", "function", name, signature, crossgl=name)


def test_instructions_reads_function_bodies_and_names_extended_instructions():
    found = semantic.instructions(ASSEMBLY)
    assert found["GLSL.std.450:Fma"] == 1
    assert "OpCapability" not in found
    assert "OpTypeVoid" not in found
    assert not any(op.startswith("NonSemantic") for op in found)


def test_identify_subtracts_the_baseline_and_the_plumbing():
    baseline = Counter({"OpLoad": 3, "OpStore": 1, "OpIAdd": 2})
    probe = Counter({"OpLoad": 3, "OpStore": 2, "OpIAdd": 2, "OpSelect": 1})
    probe["GLSL.std.450:Fma"] = 1
    assert semantic.identify(probe, baseline) == "GLSL.std.450:Fma"
    # A call that lowers to plumbing alone, such as a bit cast, is named by that plumbing.
    bitcast = Counter({"OpLoad": 3, "OpStore": 2, "OpIAdd": 2, "OpBitcast": 1})
    assert semantic.identify(bitcast, baseline) == "OpBitcast"
    assert semantic.identify(baseline, baseline) == ""


def test_lower_names_spirv_features_directly_and_honours_curated_entries(monkeypatch):
    fma = Feature("spirv", "function", "Fma", None)
    assert semantic.lower(fma) == ("GLSL.std.450:Fma", "")
    metal = Feature("metal", "function", "fma", None)
    assert semantic.lower(metal) == ("", "no SPIR-V compiler for metal")
    monkeypatch.setitem(semantic.CURATED, metal.key, "GLSL.std.450:Fma")
    assert semantic.lower(metal) == ("GLSL.std.450:Fma", "")


@pytest.mark.skipif(
    not validators.can_lower("glsl"), reason="glslangValidator or spirv-dis is missing"
)
@pytest.mark.parametrize(
    ("feature", "expected"),
    [
        (glsl("fma", VEC2, VEC2, VEC2, VEC2), "GLSL.std.450:Fma"),
        (glsl("floatBitsToInt", Type("int", (2,)), VEC2), "OpBitcast"),
        (glsl("dot", Type("float"), VEC2, VEC2), "OpDot"),
    ],
    ids=lambda value: value.name if isinstance(value, Feature) else value,
)
def test_glsl_features_lower_to_their_instruction(feature, expected):
    assert semantic.lower(feature) == (expected, "")


BARRIERS = """
         %20 = OpConstant %uint 2
         %21 = OpConstant %uint 264
          %4 = OpFunction %3 None %5
               OpControlBarrier %20 %20 %21
         %30 = OpGroupNonUniformIAdd %uint %20 ExclusiveScan %31
         %32 = OpSubgroupAllKHR %bool %33
               OpFunctionEnd
"""


def test_instructions_keep_the_operands_that_change_the_operation():
    found = semantic.instructions(BARRIERS)
    assert found["OpControlBarrier(2,2,264)"] == 1
    assert found["OpGroupNonUniformIAdd.ExclusiveScan"] == 1
    assert found["OpGroupNonUniformAll"] == 1


def slang(name, result, *params):
    signature = Signature(result, tuple(Param(p) for p in params))
    return Feature("slang", "function", name, signature, crossgl=name)


@pytest.mark.skipif(
    not validators.can_lower("slang"), reason="slangc or spirv-dis is missing"
)
@pytest.mark.parametrize(
    ("feature", "expected"),
    [
        # Three arguments: the baseline must keep the reads slangc would otherwise drop as unused.
        (slang("lerp", VEC2, VEC2, VEC2, VEC2), "GLSL.std.450:FMix"),
        (slang("asuint", Type("uint", (2,)), VEC2), "OpBitcast"),
        (slang("dot", Type("float"), VEC2, VEC2), "OpDot"),
    ],
    ids=lambda value: value.name if isinstance(value, Feature) else value,
)
def test_slang_features_lower_to_their_instruction(feature, expected):
    assert semantic.lower(feature) == (expected, "")
