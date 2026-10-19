"""The Metal and HLSL dialects of the shared preprocessor, against the compilers they were recorded from.

Behaviour tests run everywhere. The differential tests run each case through the real compiler (``xcrun metal``,
``dxc``) and through the shared preprocessor: the token sequences must match, and a source the compiler rejects
must be refused with a typed diagnostic. The predefined-macro tests compare the whole macro table with the
compiler's own and run only against the toolchain version the data was recorded from
(``tools/preprocessor_macros.py``), since another version may predefine different values.
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from crosstl.translator.diagnostics import TranslationDiagnostic
from crosstl.translator.preprocessor import (
    ConfigurationError,
    Dialect,
    HeaderMacros,
    MemoryFileSystem,
    hlsl_dialect,
    msl_dialect,
    preprocess,
)
from crosstl.translator.preprocessor.dialects import dialect as resolve_dialect
from crosstl.translator.preprocessor.engine import (
    HEADER_CODE,
    HEADER_CONFIGURATION_CODE,
)
from crosstl.translator.preprocessor.lexer import lex_text, spell
from crosstl.translator.preprocessor.predefined import (
    CONFIGURATION_CODE,
    metal_default_standard,
    metal_os_versions,
    metal_standards,
    recorded_toolchain,
)

from .cases import CPP_CASES, CPP_ERRORS
from .dialect_cases import HLSL_CASES, MSL_CASES

ROOT = Path(__file__).resolve().parents[3]


def _metal_command():
    if sys.platform != "darwin" or shutil.which("xcrun") is None:
        return None
    found = subprocess.run(
        ["xcrun", "--find", "metal"], capture_output=True, text=True, check=False
    )
    return ["xcrun", "metal"] if found.returncode == 0 else None


METAL = _metal_command()
DXC = shutil.which("dxc")


def metal_version():
    lines = subprocess.run(
        [*METAL, "--version"], capture_output=True, text=True, check=False
    ).stdout.splitlines()
    return lines[0].strip() if lines else ""


def dxc_version():
    result = subprocess.run(
        [DXC, "--version"], capture_output=True, text=True, check=False
    )
    lines = (result.stdout + result.stderr).strip().splitlines()
    return lines[0].split(": ", 1)[-1].strip() if lines else ""


def tokens(text, chosen):
    return [token.text for token in lex_text(text, mode=chosen.lex_mode) if token.text]


def ours(source, chosen, path):
    """The shared preprocessor's output without the compiler-header includes it keeps for the front end (the
    compiler's own output imports them as modules, which the comparison drops too)."""
    try:
        result = preprocess(source, chosen, path=str(path))
    except TranslationDiagnostic:
        return False, None
    kept = {f"#include <{header}>" for header in result.retained_includes}
    return True, "\n".join(
        line for line in result.text.splitlines() if line not in kept
    )


def run_metal(std, path):
    result = subprocess.run(
        [
            *METAL,
            "-E",
            "-P",
            "-x",
            "metal",
            f"-std={std}",
            "-fno-color-diagnostics",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    lines = [
        line
        for line in result.stdout.splitlines()
        if not line.startswith("#pragma clang module import")
    ]
    if result.returncode != 0:
        return False, result.stderr.strip()
    return True, "\n".join(lines)


def run_dxc(profile, extra, path):
    out = path.with_suffix(".i")
    result = subprocess.run(
        [DXC, "-T", profile, *extra, "-P", "-Fi", str(out), str(path)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    # dxc -P exits 0 after a preprocessing error; the diagnostic is the signal.
    failed = (
        result.returncode != 0
        or "error" in result.stdout + result.stderr
        or not out.is_file()
    )
    if failed:
        return False, None
    lines = [
        line for line in out.read_text().splitlines() if not line.startswith("#line")
    ]
    return True, "\n".join(lines)


def hlsl_from_options(profile, extra):
    options = list(extra)
    kwargs = {}
    if "-HV" in options:
        kwargs["hlsl_version"] = options[options.index("-HV") + 1]
    if "-enable-16bit-types" in options:
        kwargs["enable_16bit_types"] = True
    for option in options:
        if option.startswith("-fspv-target-env="):
            kwargs["spirv"] = option.split("=", 1)[1]
    if "-spirv" in options and "spirv" not in kwargs:
        kwargs["spirv"] = "vulkan1.0"
    return hlsl_dialect(profile, **kwargs)


def assert_same(reference, mine, chosen):
    accepted, text = reference
    if not accepted:
        assert not mine[
            0
        ], f"the compiler rejects the source ({text}) but the shared preprocessor gave:\n{mine[1]}"
        return
    assert mine[0], "the shared preprocessor refused a source the compiler accepts"
    assert tokens(mine[1], chosen) == tokens(text, chosen)


# Behaviour -----------------------------------------------------------------------------------------------
def test_dialect_names_resolve_to_the_recorded_defaults():
    assert resolve_dialect("msl") == msl_dialect()
    assert resolve_dialect("hlsl") == hlsl_dialect()
    assert msl_dialect().toolchain.startswith(recorded_toolchain("msl"))
    with pytest.raises(KeyError):
        resolve_dialect("wgsl")


@pytest.mark.parametrize(
    "build",
    [
        lambda: msl_dialect("metal9.9"),
        lambda: msl_dialect("macos-metal2.0", platform="ios"),
        lambda: msl_dialect("metal3.0", platform="tvos"),
        lambda: msl_dialect("metal3.0", os_version="13.4"),
        lambda: msl_dialect("metal3.0", options=["fast-everything"]),
        lambda: hlsl_dialect("cs_6_10"),
        lambda: hlsl_dialect("ms_6_4"),
        lambda: hlsl_dialect("cs_6_0", hlsl_version="2015"),
        lambda: hlsl_dialect("cs_6_1", enable_16bit_types=True),
        lambda: hlsl_dialect("cs_6_2", hlsl_version="2017", enable_16bit_types=True),
        lambda: hlsl_dialect("cs_6_0", spirv="vulkan9.0"),
    ],
)
def test_unrecorded_configurations_are_refused(build):
    with pytest.raises(ConfigurationError) as raised:
        build()
    assert raised.value.code == CONFIGURATION_CODE


def test_every_recorded_metal_configuration_builds():
    for platform in ("macos", "ios"):
        for std in metal_standards(platform):
            chosen = msl_dialect(std, platform=platform)
            assert chosen.int_width == 128
            assert "metal_stdlib" in chosen.headers
    for version in ("10.13", "14.0", "26.0"):
        assert msl_dialect("macos-metal2.0", os_version=version).predefined_definitions


def test_metal_standard_selects_the_language_and_lexer():
    def value(std, name):
        return preprocess(name, msl_dialect(std)).text.strip()

    assert value("metal3.1", "__METAL_VERSION__") == "310"
    assert value("metal3.1", "__cplusplus") == "201402L"
    assert value("metal4.0", "__cplusplus") == "201703L"
    assert value("macos-metal1.1", "__cplusplus") == "201103L"
    assert not msl_dialect("macos-metal1.1").lex_mode.digit_separators
    assert msl_dialect("metal3.1").lex_mode.digit_separators
    assert msl_dialect("metal4.0").lex_mode.utf8_characters
    assert not msl_dialect("metal4.1").lex_mode.raw_strings


def test_deployment_target_and_options_change_predefined_macros():
    def value(chosen, name):
        return preprocess(name, chosen).text.strip()

    newest = msl_dialect("metal3.1")
    oldest = msl_dialect("metal3.1", os_version="14.0")
    assert value(newest, "__AIR_VERSION__") != value(oldest, "__AIR_VERSION__")
    assert value(oldest, "__ENVIRONMENT_OS_VERSION_MIN_REQUIRED__") == "140000"
    assert (
        value(
            msl_dialect("metal3.0", platform="ios", os_version="16.0"),
            "__ENVIRONMENT_OS_VERSION_MIN_REQUIRED__",
        )
        == "160000"
    )
    assert value(newest, "__FAST_MATH__") == "1"
    assert (
        value(msl_dialect("metal3.1", options=["math-mode=safe"]), "__FAST_MATH__")
        == "__FAST_MATH__"
    )
    assert (
        value(msl_dialect("metal3.1", options=["O0"]), "__OPTIMIZE__ __NO_INLINE__")
        == "__OPTIMIZE__ 1"
    )


def test_metal_header_defines_its_macros_once():
    text = preprocess(
        "#include <metal_stdlib>\n#undef assert\n#include <metal_stdlib>\n#include <metal_assert>\nassert(1)\n",
        "msl",
    ).text
    assert text.splitlines()[-1] == "assert(1)"
    result = preprocess("#include <metal_stdlib>\nassert(1)\n", "msl")
    assert result.text.splitlines()[-1] == "((void) 0)"
    assert result.includes == []
    assert "#include <metal_stdlib>" in result.text


def test_a_changed_header_condition_is_refused():
    with pytest.raises(TranslationDiagnostic) as raised:
        preprocess("#define NDEBUG\n#include <metal_stdlib>\n", "msl")
    assert raised.value.code == HEADER_CONFIGURATION_CODE
    with pytest.raises(TranslationDiagnostic) as raised:
        preprocess("#include <metal_stdlib>\n", "msl", defines={"NDEBUG": None})
    assert raised.value.code == HEADER_CONFIGURATION_CODE
    # A command-line change to a macro the implicitly included <metal_types> tests is refused up front.
    with pytest.raises(TranslationDiagnostic) as raised:
        preprocess("x\n", "msl", defines={"__METAL_VERSION__": "200"})
    assert raised.value.code == HEADER_CONFIGURATION_CODE
    # A source macro the headers do not test is fine.
    assert (
        preprocess(
            "#define N 4\n#include <metal_stdlib>\nN\n", "msl"
        ).text.splitlines()[-1]
        == "4"
    )


def test_a_header_an_option_changes_is_refused():
    chosen = msl_dialect("metal3.1", options=["fp32-functions=precise"])
    with pytest.raises(TranslationDiagnostic) as raised:
        preprocess("#include <metal_stdlib>\n", chosen)
    assert raised.value.code == HEADER_CODE
    assert (
        preprocess("#include <metal_atomic>\nok\n", chosen).text.splitlines()[-1]
        == "ok"
    )


def test_an_unavailable_compiler_header_is_absent_and_refused():
    chosen = Dialect(
        "custom",
        retained_includes=frozenset({"future.h"}),
        headers={
            "future.h": HeaderMacros(
                (), frozenset(), frozenset(), "<future.h> needs -std=next", False
            )
        },
    )
    assert (
        preprocess(
            "#if __has_include(<future.h>)\nyes\n#else\nno\n#endif\n", chosen
        ).text.strip()
        == "no"
    )
    with pytest.raises(TranslationDiagnostic) as raised:
        preprocess("#include <future.h>\n", chosen)
    assert raised.value.code == HEADER_CODE


def test_dxc_builtin_headers_are_refused_unless_on_the_include_path():
    with pytest.raises(TranslationDiagnostic) as raised:
        preprocess("#include <dx/linalg.h>\n", "hlsl")
    assert raised.value.code == HEADER_CODE
    files = MemoryFileSystem(
        {"/inc/dx/linalg.h": "#define LINALG 1\n", "/src/a.hlsl": ""}
    )
    result = preprocess(
        "#include <dx/linalg.h>\nLINALG\n",
        "hlsl",
        path="/src/a.hlsl",
        include_paths=["/inc"],
        file_system=files,
    )
    assert result.text.splitlines()[-1] == "1"


def test_hlsl_profile_and_options_set_predefined_macros():
    def value(chosen, names):
        return preprocess(names, chosen).text.strip()

    assert (
        value(
            hlsl_dialect("ps_6_6"),
            "__SHADER_TARGET_STAGE __SHADER_TARGET_MAJOR __SHADER_TARGET_MINOR",
        )
        == "0 6 6"
    )
    assert (
        value(hlsl_dialect("cs_6_0", hlsl_version="202x"), "__HLSL_VERSION") == "2029"
    )
    assert (
        value(hlsl_dialect("cs_6_2", enable_16bit_types=True), "__HLSL_ENABLE_16_BIT")
        == "1"
    )
    assert (
        value(hlsl_dialect("cs_6_2"), "__HLSL_ENABLE_16_BIT") == "__HLSL_ENABLE_16_BIT"
    )
    assert (
        value(
            hlsl_dialect("cs_6_0", spirv="vulkan1.3"),
            "__spirv__ __SPIRV_MAJOR_VERSION__ __SPIRV_MINOR_VERSION__",
        )
        == "1 1 6"
    )
    assert (
        value(hlsl_dialect("cs_6_0"), "__cplusplus __spirv__")
        == "__cplusplus __spirv__"
    )


def test_hlsl_is_clang_3_7():
    chosen = hlsl_dialect()
    text = preprocess(
        "#define F(a,...) f(a __VA_OPT__(,) __VA_ARGS__)\nF(1)\n#define and 1\nand\n",
        chosen,
    ).text
    assert tokens(text, chosen) == [
        "f",
        "(",
        "1",
        "__VA_OPT__",
        "(",
        ",",
        ")",
        ")",
        "1",
    ]
    assert (
        preprocess(
            "#define A\n#if 0\n#elifdef A\nx\n#else\ny\n#endif\n", chosen
        ).text.strip()
        == "y"
    )
    for source in (
        "#elifdef A\n",
        "#if 1 and 1\n#endif\n",
        "#if 1'0\n#endif\n",
        '#ident "x"\n',
    ):
        with pytest.raises(TranslationDiagnostic):
            preprocess(source, chosen)


def test_output_does_not_depend_on_the_hash_seed():
    script = (
        "from crosstl.translator.preprocessor import preprocess, msl_dialect\n"
        "r = preprocess('#include <metal_stdlib>\\nassert(1) M_PI_F\\n', msl_dialect('metal3.2', platform='ios'))\n"
        "print(r.text); print(sorted(r.macros)[:50]); print(r.retained_includes)\n"
    )
    outputs = set()
    for seed in ("0", "1", "4242"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        outputs.add(
            subprocess.run(
                [sys.executable, "-c", script],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
    assert len(outputs) == 1


# Differential: Metal -------------------------------------------------------------------------------------
needs_metal = pytest.mark.skipif(
    METAL is None, reason="the Metal compiler is not installed"
)
needs_dxc = pytest.mark.skipif(DXC is None, reason="dxc is not installed")


@functools.cache
def metal_reference_standard():
    """The recorded default ``-std`` when the installed compiler accepts it, else the newest recorded standard it
    does accept. A compiler older than the recorded one (a CI image, say) rejects the newest ``-std`` outright,
    which would read as the compiler rejecting every case."""
    standards = metal_standards("macos")
    preferred = metal_default_standard()
    for std in (preferred, *reversed(standards)):
        probe = subprocess.run(
            [*METAL, "-E", "-P", "-x", "metal", f"-std={std}", "-"],
            input="",
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if probe.returncode == 0:
            return std
    return None


@needs_metal
@pytest.mark.parametrize("name", sorted(CPP_CASES) + sorted(CPP_ERRORS))
def test_cpp_case_matches_metal(name, tmp_path):
    source = CPP_CASES[name] if name in CPP_CASES else CPP_ERRORS[name][0]
    path = tmp_path / f"{name}.metal"
    path.write_text(source)
    std = metal_reference_standard()
    if std is None:
        pytest.skip(f"{metal_version()} accepts none of the recorded -std values")
    chosen = msl_dialect(std)
    assert_same(run_metal(std, path), ours(source, chosen, path), chosen)


@needs_metal
@pytest.mark.parametrize("name", sorted(MSL_CASES))
def test_msl_case_matches_metal(name, tmp_path):
    std, source = MSL_CASES[name]
    path = tmp_path / f"{name}.metal"
    path.write_text(source)
    chosen = msl_dialect(std)
    reference = run_metal(std, path)
    if name.startswith("error_"):
        assert not reference[0], "an error case must be rejected by the Metal compiler"
    elif "AIR_VERSION" not in source:
        assert reference[0], "a valid case must be accepted by the Metal compiler"
    if "__AIR_VERSION__" in source and metal_version() != recorded_toolchain("msl"):
        pytest.skip("__AIR_VERSION__ depends on the Metal toolchain version")
    assert_same(reference, ours(source, chosen, path), chosen)


def render(macro):
    head = macro.name
    if macro.params is not None:
        params = list(macro.params)
        if macro.variadic:
            params[-1] = "..." if params[-1] == "__VA_ARGS__" else params[-1] + "..."
        head += "(" + ",".join(params) + ")"
    return head + " " + spell(macro.body)


METAL_TABLES = [
    ("metal4.1", "macos", None, (), ""),
    ("metal3.0", "ios", "16.0", (), ""),
    ("macos-metal2.0", "macos", "10.13", ("O0",), ""),
    ("metal3.2", "macos", "15.0", ("math-mode=safe", "enable-logging"), ""),
    ("metal3.1", "macos", None, (), "#include <metal_stdlib>\n"),
    ("metal4.0", "ios", None, (), "#include <simd/simd.h>\n#include <metal_stdlib>\n"),
]
METAL_FLAGS = {
    "O0": "-O0",
    "math-mode=safe": "-fmetal-math-mode=safe",
    "enable-logging": "-fmetal-enable-logging",
}


@needs_metal
@pytest.mark.parametrize("std,platform,os_version,options,source", METAL_TABLES)
def test_macro_table_matches_metal(std, platform, os_version, options, source):
    if metal_version() != recorded_toolchain("msl"):
        pytest.skip(
            f"the data was recorded from {recorded_toolchain('msl')}, not {metal_version()}"
        )
    chosen = msl_dialect(std, platform=platform, os_version=os_version, options=options)
    target = f"air64-apple-{platform}{os_version or metal_os_versions(platform)[-1]}"
    result = subprocess.run(
        [
            *METAL,
            "-x",
            "metal",
            "-fno-modules",
            "-dM",
            "-E",
            f"-std={std}",
            "-target",
            target,
        ]
        + [METAL_FLAGS[option] for option in options]
        + ["-"],
        input=source,
        capture_output=True,
        text=True,
        check=True,
    )
    expected = {}
    for line in result.stdout.splitlines():
        definition = line[len("#define ") :]
        expected[lex_text(definition, mode=chosen.lex_mode)[0].text] = tokens(
            definition, chosen
        )
    table = preprocess(source, chosen).macros
    actual = {
        name: tokens(render(macro), chosen)
        for name, macro in table.items()
        if macro.builtin is None
    }
    assert sorted(set(actual) ^ set(expected)) == []
    assert {name for name in actual if actual[name] != expected[name]} == set()


# Differential: DXC ---------------------------------------------------------------------------------------
@needs_dxc
@pytest.mark.parametrize("name", sorted(CPP_CASES) + sorted(CPP_ERRORS))
def test_cpp_case_matches_dxc(name, tmp_path):
    source = CPP_CASES[name] if name in CPP_CASES else CPP_ERRORS[name][0]
    path = tmp_path / f"{name}.hlsl"
    path.write_text(source)
    chosen = hlsl_dialect()
    assert_same(run_dxc("cs_6_0", (), path), ours(source, chosen, path), chosen)


@needs_dxc
@pytest.mark.parametrize("name", sorted(HLSL_CASES))
def test_hlsl_case_matches_dxc(name, tmp_path):
    profile, extra, source = HLSL_CASES[name]
    path = tmp_path / f"{name}.hlsl"
    path.write_text(source)
    chosen = hlsl_from_options(profile, extra)
    reference = run_dxc(profile, extra, path)
    assert reference[0] == (
        not name.startswith("error_")
    ), "the case's expectation does not hold for dxc"
    if "__DXC_VERSION" in source and dxc_version() != recorded_toolchain("hlsl"):
        pytest.skip("the DXC version macros depend on the DXC build")
    assert_same(reference, ours(source, chosen, path), chosen)


HLSL_TABLES = [
    ("cs_6_0", ()),
    ("ps_6_6", ("-HV", "2018", "-enable-16bit-types")),
    ("lib_6_8", ("-HV", "202x")),
    ("as_6_5", ("-spirv", "-fspv-target-env=vulkan1.1spirv1.4")),
    ("vs_5_0", ("-HV", "2016")),
]


@needs_dxc
@pytest.mark.parametrize("profile,extra", HLSL_TABLES)
def test_macro_table_matches_dxc(profile, extra, tmp_path):
    chosen = hlsl_from_options(profile, extra)
    table = preprocess("", chosen).macros
    names = sorted(name for name, macro in table.items() if macro.builtin is None)
    probe = "".join(
        f"#ifdef {name}\n@@ {index} @@ {name} @@\n#else\n@@ {index} @@ undefined @@\n#endif\n"
        for index, name in enumerate(names)
    )
    path = tmp_path / "table.hlsl"
    path.write_text(probe)
    accepted, text = run_dxc(profile, extra, path)
    assert accepted
    mine = preprocess(probe, chosen).text
    skip = set()
    if dxc_version() != recorded_toolchain("hlsl"):
        skip = {
            "__DXC_VERSION_MAJOR",
            "__DXC_VERSION_MINOR",
            "__DXC_VERSION_RELEASE",
            "__DXC_VERSION_COMMITS",
            "__VERSION__",
        }
    for index, name in enumerate(names):
        if name in skip:
            continue
        marker = f"@@ {index} @@"
        reference_line = next(
            line for line in text.splitlines() if line.startswith(marker)
        )
        mine_line = next(line for line in mine.splitlines() if line.startswith(marker))
        assert tokens(mine_line, chosen) == tokens(reference_line, chosen), name
