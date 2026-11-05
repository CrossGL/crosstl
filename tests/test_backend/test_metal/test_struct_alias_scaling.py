"""Struct-scoped alias canonicalization resolves only the aliases a type spells.

A module with many template instantiations has hundreds of concrete structs with ``using`` aliases, and the importer
canonicalizes thousands of short type strings against all of them. Resolving every alias of every struct for each
string, and compiling a pattern for each, cost minutes per module; the result depends only on the aliases the
string names.
"""

from crosstl.backend.Metal import pattern_cache
from crosstl.backend.Metal.preprocessor import MetalPreprocessor

STRUCT_COUNT = 60


def _module():
    blocks = [f"""
        struct Block{index} {{
          using value_t = float{index % 4 + 1};
          typedef int index_t;
          using Self = Block{index};
        }};
        """ for index in range(STRUCT_COUNT)]
    return "\n".join(blocks)


def _definitions():
    preprocessor = MetalPreprocessor()
    return preprocessor, preprocessor._find_concrete_struct_definitions(_module())


def _counting(monkeypatch, preprocessor):
    calls = []
    resolve = preprocessor._canonicalize_struct_scoped_type

    def counting(type_text, struct, *args, **kwargs):
        calls.append((type_text, struct.name))
        return resolve(type_text, struct, *args, **kwargs)

    monkeypatch.setattr(preprocessor, "_canonicalize_struct_scoped_type", counting)
    return calls


def test_qualified_aliases_resolve_only_the_aliases_the_text_names(monkeypatch):
    preprocessor, structs = _definitions()
    calls = _counting(monkeypatch, preprocessor)

    resolved = preprocessor._canonicalize_qualified_struct_type_aliases(
        "device Block7::value_t* data, Block3::index_t i", structs
    )

    assert resolved == "device float4* data, int i"
    # Only the two aliases the text spells, once each per owner that declares them.
    assert {name for name, _owner in calls} <= {"value_t", "index_t"}
    assert len(calls) <= 2 * STRUCT_COUNT


def test_text_without_struct_aliases_resolves_nothing(monkeypatch):
    preprocessor, structs = _definitions()
    calls = _counting(monkeypatch, preprocessor)

    for text in ("float", "uint3", "metal::array<int, 4>", "thread half4&"):
        assert (
            preprocessor._canonicalize_qualified_struct_type_aliases(text, structs)
            == text
        )

    assert calls == []


CHAINS = """
struct Inner { using value_t = half2; };
struct Outer { typedef Inner::value_t inner_t; using Self = Outer; };
struct Wrap { using element_t = Outer::inner_t; };
"""


def test_qualified_alias_chains_resolve_through_later_rewrites():
    # ``Outer::Self::inner_t`` names ``inner_t`` only once ``Outer::Self`` has been rewritten to ``Outer``.
    preprocessor = MetalPreprocessor()
    structs = preprocessor._find_concrete_struct_definitions(CHAINS)

    def resolve(text):
        return preprocessor._canonicalize_qualified_struct_type_aliases(text, structs)

    assert resolve("Outer::Self::inner_t") == "half2"
    assert resolve("Wrap::element_t") == "half2"
    assert (
        resolve("device Wrap::element_t* p, Inner::value_t v")
        == "device half2* p, half2 v"
    )


def test_pattern_cache_compiles_each_pattern_once():
    source = r"(?<![A-Za-z0-9_:])value_t(?=::)"
    assert pattern_cache.pattern(source) is pattern_cache.pattern(source)
    assert pattern_cache.PATTERN_CACHE_SIZE > 512
