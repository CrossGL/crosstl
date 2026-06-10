from __future__ import annotations

import textwrap
from dataclasses import dataclass

import pytest

from crosstl.backend.Mojo.MojoCrossGLCodeGen import MojoToCrossGLConverter
from crosstl.backend.Mojo.MojoLexer import MojoLexer
from crosstl.backend.Mojo.MojoParser import MojoParser

MODULAR_REPO = "https://github.com/modular/modular"
MODULAR_GPU_REDUCTION_COMMIT = "daa47bb846cc213723a54c51844ea4e923eb5e13"


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
        name="modular-gpu-reduction-function-type-parameters",
        repo=MODULAR_REPO,
        commit=MODULAR_GPU_REDUCTION_COMMIT,
        path="mojo/stdlib/std/algorithm/backend/gpu/reduction.mojo",
        code=textwrap.dedent("""
            def reduce_adapter(
                input_fn: def[dtype: DType, width: Int, rank: Int](
                    IndexList[rank]
                ) capturing[_] -> SIMD[dtype, width],
                output_fn: def[dtype: DType, width: SIMDSize, rank: Int](
                    IndexList[rank], StaticTuple[SIMD[dtype, width], num_reductions]
                ) capturing[_] -> None,
            ):
                pass
        """).strip(),
        contains=(
            "void reduce_adapter(",
            "def[dtype:DType, width:Int, rank:Int](IndexList[rank])",
            "capturing[_] -> SIMD[dtype, width] input_fn",
            "capturing[_] -> None output_fn",
        ),
    )
]


def parse_mojo(code):
    tokens = MojoLexer(code).tokenize()
    return MojoParser(tokens).parse()


def generate_crossgl(code):
    ast = parse_mojo(code)
    return MojoToCrossGLConverter().generate(ast)


def test_external_mojo_fixture_metadata_records_repositories_and_commits():
    assert all(
        fixture.repo.startswith("https://github.com/") for fixture in EXTERNAL_FIXTURES
    )
    assert all(len(fixture.commit) == 40 for fixture in EXTERNAL_FIXTURES)
    assert all(fixture.path.endswith(".mojo") for fixture in EXTERNAL_FIXTURES)


@pytest.mark.parametrize("fixture", EXTERNAL_FIXTURES, ids=lambda fixture: fixture.name)
def test_parse_external_mojo_fixture(fixture):
    ast = parse_mojo(fixture.code)

    assert ast is not None
    assert ast.functions
    assert fixture.source_url.startswith(fixture.repo)


@pytest.mark.parametrize("fixture", EXTERNAL_FIXTURES, ids=lambda fixture: fixture.name)
def test_codegen_external_mojo_fixture_preserves_function_type_parameters(fixture):
    crossgl = generate_crossgl(fixture.code)

    for expected in fixture.contains:
        assert expected in crossgl
