from __future__ import annotations

import textwrap
from dataclasses import dataclass

import pytest

from crosstl.backend.SPIRV import VulkanLexer, VulkanParser
from crosstl.backend.SPIRV.VulkanCrossGLCodeGen import VulkanToCrossGLConverter
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser

VULKAN_SAMPLES_REPO = "https://github.com/KhronosGroup/Vulkan-Samples"
VULKAN_SAMPLES_COMMIT = "ab1e93d4a5dadf4c804fb6abbbe0b27dfa912b5a"


@dataclass(frozen=True)
class ExternalFixture:
    name: str
    repo: str
    commit: str
    path: str
    code: str
    contains: tuple[str, ...]

    @property
    def source_url(self):
        return f"{self.repo}/blob/{self.commit}/{self.path}"


EXTERNAL_FIXTURES = [
    ExternalFixture(
        name="vulkan-samples-timeline-semaphore-image-write",
        repo=VULKAN_SAMPLES_REPO,
        commit=VULKAN_SAMPLES_COMMIT,
        path="shaders/timeline_semaphore/glsl/game_of_life_init.comp.spv",
        code=textwrap.dedent("""
            ; Reduced from KhronosGroup/Vulkan-Samples@ab1e93d4a5dadf4c804fb6abbbe0b27dfa912b5a
            ; shaders/timeline_semaphore/glsl/game_of_life_init.comp.spv.
            OpCapability Shader
            OpMemoryModel Logical GLSL450
            OpEntryPoint GLCompute %main "main"
            OpExecutionMode %main LocalSize 8 8 1
            OpName %image "Image"
            OpDecorate %image DescriptorSet 0
            OpDecorate %image Binding 0
            OpDecorate %image NonReadable
            %void = OpTypeVoid
            %fn = OpTypeFunction %void
            %float = OpTypeFloat 32
            %int = OpTypeInt 32 1
            %v2int = OpTypeVector %int 2
            %v4float = OpTypeVector %float 4
            %image_type = OpTypeImage %float 2D 0 0 0 2 Rgba8
            %ptr_image = OpTypePointer UniformConstant %image_type
            %zero_i = OpConstant %int 0
            %one_f = OpConstant %float 1.0
            %zero_f = OpConstant %float 0.0
            %coord = OpConstantComposite %v2int %zero_i %zero_i
            %texel = OpConstantComposite %v4float %one_f %one_f %one_f %zero_f
            %image = OpVariable %ptr_image UniformConstant
            %main = OpFunction %void None %fn
            %label = OpLabel
            OpImageWrite %image %coord %texel
            OpReturn
            OpFunctionEnd
        """).strip(),
        contains=(
            "RWTexture2D Image @set(0) @binding(0) @rgba8 @writeonly;",
            "layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;",
            "imageStore(Image, int2(0, 0), float4(1.0, 1.0, 1.0, 0.0));",
        ),
    )
]


def parse_vulkan(code):
    tokens = VulkanLexer(code).tokenize()
    return VulkanParser(tokens).parse()


def generate_crossgl(code):
    ast = parse_vulkan(code)
    return VulkanToCrossGLConverter().generate(ast)


def parse_crossgl(code):
    tokens = CrossGLLexer(code).get_tokens()
    return CrossGLParser(tokens).parse()


def test_external_vulkan_fixture_metadata_records_repositories_and_commits():
    assert all(
        fixture.repo.startswith("https://github.com/") for fixture in EXTERNAL_FIXTURES
    )
    assert all(len(fixture.commit) == 40 for fixture in EXTERNAL_FIXTURES)
    assert all(fixture.path.endswith(".spv") for fixture in EXTERNAL_FIXTURES)


@pytest.mark.parametrize("fixture", EXTERNAL_FIXTURES, ids=lambda fixture: fixture.name)
def test_parse_external_vulkan_fixture(fixture):
    ast = parse_vulkan(fixture.code)

    assert ast is not None
    assert ast.functions
    assert fixture.source_url.startswith(fixture.repo)


@pytest.mark.parametrize("fixture", EXTERNAL_FIXTURES, ids=lambda fixture: fixture.name)
def test_codegen_external_vulkan_fixture_to_parseable_crossgl(fixture):
    crossgl = generate_crossgl(fixture.code)

    for expected in fixture.contains:
        assert expected in crossgl
    assert parse_crossgl(crossgl) is not None
