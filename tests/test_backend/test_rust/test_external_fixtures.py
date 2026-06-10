from __future__ import annotations

import textwrap
from dataclasses import dataclass

import pytest

from crosstl.backend.Rust.RustCrossGLCodeGen import RustToCrossGLConverter
from crosstl.backend.Rust.RustLexer import RustLexer
from crosstl.backend.Rust.RustParser import RustParser
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser

RUST_GPU_REPO = "https://github.com/Rust-GPU/rust-gpu"
RUST_GPU_SUBGROUP_COMMIT = "36e3348cdc2f824afec64b3b5af5d369d98a4c0d"


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
        name="rust-gpu-subgroup-local-transform-impl",
        repo=RUST_GPU_REPO,
        commit=RUST_GPU_SUBGROUP_COMMIT,
        path="crates/spirv-std/src/arch/subgroup.rs",
        code=textwrap.dedent("""
            pub fn subgroup_all_equal<T: ScalarComposite>(value: T) -> bool {
                struct Transform(bool);

                impl ScalarOrVectorTransform for Transform {
                    fn transform<T: ScalarOrVector>(&mut self, value: T) -> T {
                        value
                    }
                }

                let mut transform = Transform(true);
                value.transform(&mut transform);
                transform.0
            }
        """).strip(),
        contains=(
            "bool subgroup_all_equal(T value)",
            "let mut transform = Transform(true);",
            "value.transform(transform);",
            "return transform.0;",
        ),
    )
]


def parse_rust(code):
    tokens = RustLexer(code).tokenize()
    return RustParser(tokens).parse()


def generate_crossgl(code):
    ast = parse_rust(code)
    return RustToCrossGLConverter().generate(ast)


def parse_crossgl(code):
    tokens = CrossGLLexer(code).get_tokens()
    return CrossGLParser(tokens).parse()


def test_external_rust_fixture_metadata_records_repositories_and_commits():
    assert all(
        fixture.repo.startswith("https://github.com/") for fixture in EXTERNAL_FIXTURES
    )
    assert all(len(fixture.commit) == 40 for fixture in EXTERNAL_FIXTURES)
    assert all(fixture.path.endswith(".rs") for fixture in EXTERNAL_FIXTURES)


@pytest.mark.parametrize("fixture", EXTERNAL_FIXTURES, ids=lambda fixture: fixture.name)
def test_parse_external_rust_fixture(fixture):
    ast = parse_rust(fixture.code)

    assert ast is not None
    assert ast.functions
    assert fixture.source_url.startswith(fixture.repo)


@pytest.mark.parametrize("fixture", EXTERNAL_FIXTURES, ids=lambda fixture: fixture.name)
def test_codegen_external_rust_fixture_to_parseable_crossgl(fixture):
    crossgl = generate_crossgl(fixture.code)

    for expected in fixture.contains:
        assert expected in crossgl
    assert parse_crossgl(crossgl) is not None
