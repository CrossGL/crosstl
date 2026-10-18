"""The Metal standard library derivation, on a synthetic compiler AST so it runs without the Metal toolchain."""

from .inventory import load
from .inventory import msl as msl_inventory
from .inventory import msl_stdlib

AST = """\
TranslationUnitDecl 0x1 <<invalid sloc>> <invalid sloc>
|-NamespaceDecl 0x2 <{root}/metal_math:10:1, line:90:1> line:10:11 metal
| |-FunctionDecl 0x3 <line:12:1, col:30> col:18 used abs 'float (float)' inline
| |-FunctionDecl 0x4 <line:13:1, col:30> col:18 __internal 'float (float)'
| |-NamespaceDecl 0x5 <line:20:1, line:30:1> line:20:11 fast
| | `-FunctionTemplateDecl 0x6 <line:21:1, col:9> col:9 sin
| |-NamespaceDecl 0x7 <line:31:1, line:40:1> line:31:11 __detail
| | `-FunctionDecl 0x8 <line:32:1, col:9> col:9 helper 'void ()'
| |-ClassTemplateDecl 0x9 <{root}/__bits/metal_texture2d:1:1, line:50:1> line:2:8 _texture2d_sample
| | `-CXXRecordDecl 0xa <line:2:1, line:50:1> line:2:8 struct _texture2d_sample definition
| |   |-CXXRecordDecl 0xb <col:1, col:8> col:8 implicit struct _texture2d_sample
| |   |-CXXMethodDecl 0xc <line:3:3, col:20> col:20 sample 'vec<T, 4> (sampler, float2) const'
| |   |-AccessSpecDecl 0xd <line:4:1, col:8> col:1 private
| |   `-CXXMethodDecl 0xe <line:5:3, col:20> col:20 hidden_helper 'void ()'
| |-ClassTemplateDecl 0xf <line:60:1, line:61:8> col:8 texture2d
| | `-CXXRecordDecl 0x10 prev 0x9 <line:61:1, line:80:1> line:61:8 class texture2d definition
| |   |-DefinitionData standard_layout
| |   | `-DefaultConstructor exists
| |   |-public 'metal::_texture2d_sample<T, metal::access::sample>'
| |   |-CXXMethodDecl 0x11 <line:62:3, col:20> col:20 private_before_access 'void ()'
| |   |-AccessSpecDecl 0x12 <line:63:1, col:7> col:1 public
| |   |-CXXConstructorDecl 0x13 <line:64:3, col:20> col:3 texture2d 'void ()'
| |   |-FunctionTemplateDecl 0x14 <line:65:3, col:20> col:3 texture2d
| |   |-FunctionTemplateDecl 0x15 <line:66:3, col:20> col:3 operator T
| |   `-CXXMethodDecl 0x16 <line:67:3, col:20> col:20 get_width 'uint () const'
| `-EnumDecl 0x17 <{root}/metal_types:5:1, line:9:1> line:5:12 class access 'int'
|   |-EnumConstantDecl 0x18 <line:6:3> col:3 sample 'metal::access'
|   `-EnumConstantDecl 0x19 <line:7:3> col:3 read_write 'metal::access'
`-FunctionDecl 0x1a <{root}/metal_new:1:1, col:5> col:21 operator new 'void *(size_t, void *)' inline
"""


def test_public_declarations_include_inherited_members_and_skip_internals(tmp_path):
    root = tmp_path.resolve()
    found = msl_stdlib.declarations(AST.format(root=root).splitlines(), root)
    assert {(d.kind, d.qualified, d.header) for d in found} == {
        ("function", "metal::abs", "metal_math"),
        ("function", "metal::fast::sin", "metal_math"),
        ("class", "metal::texture2d", "__bits/metal_texture2d"),
        ("member-function", "metal::texture2d::sample", "__bits/metal_texture2d"),
        ("member-function", "metal::texture2d::get_width", "__bits/metal_texture2d"),
        ("enum", "metal::access", "metal_types"),
        ("enumerator", "metal::access::sample", "metal_types"),
        ("enumerator", "metal::access::read_write", "metal_types"),
    }


def test_node_names_skip_flags_ranges_and_links():
    assert msl_stdlib.parse_node(
        " 0x7b prev 0x7a <line:1:1, line:2:1> line:1:8 implicit referenced struct texture2d"
    ) == (True, "texture2d")
    assert msl_stdlib.parse_node(
        " 0x7b <<invalid sloc>> <invalid sloc> implicit float2 'float2'"
    ) == (True, "float2")
    assert msl_stdlib.parse_node(" 0x7b <col:1, col:9> col:9 operator== 'bool ()'") == (
        False,
        None,
    )


def test_tree_digest_depends_on_every_header():
    first = msl_stdlib.tree_digest({"metal_math": "a", "metal_types": "b"})
    assert first == msl_stdlib.tree_digest({"metal_types": "b", "metal_math": "a"})
    assert first != msl_stdlib.tree_digest({"metal_math": "a", "metal_types": "c"})


def test_library_citations_prefer_class_sections_then_headers():
    cite = msl_inventory.library_citation
    assert cite(
        "member-function", ("texture2d",), "sample", ("__bits/metal_texture2d",)
    ) == (
        "section",
        "2D Texture",
    )
    assert cite("class", ("raytracing",), "intersector", ("metal_raytracing",)) == (
        "section",
        "Intersector Type",
    )
    assert cite(
        "enumerator", ("address",), "repeat", ("__bits/metal_texture_common",)
    ) == (
        "section",
        "Samplers",
    )
    assert cite("function", (), "fma", ("metal_math",)) == ("section", "Math Functions")
    assert cite("alias", (), "float2", ("metal_types",)) == (
        "section",
        "Vector Data Types",
    )
    assert cite("class", (), "is_same", ("metal_type_traits",)) == (
        "document",
        "metal_type_traits",
    )


def test_committed_metal_inventory_carries_the_derived_library():
    items = {item["id"]: item for item in load("metal").items}
    assert items["metal.method.texture2d.sample"]["requires"] == {"msl": "1.0"}
    assert items["metal.function.simd_shuffle"]["requires"] == {"msl": "2.0"}
    assert items["metal.function.fast.sin"]["cite"] == "msl-4.1 §6.6"
    assert items["metal.class.texture2d"]["category"] == "resource"
    assert sum(item["kind"] == "library-function" for item in items.values()) > 350
    assert sum(item["kind"] == "member-function" for item in items.values()) > 450
