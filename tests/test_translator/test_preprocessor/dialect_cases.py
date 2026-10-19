"""Cases for the Metal and HLSL dialects: where their compilers' preprocessors differ from Clang's C++20 one.

``MSL_CASES`` map a name to ``(std, source)`` and are compared against ``metal -E -P -std=<std>``;
``HLSL_CASES`` map a name to ``(profile, extra dxc options, source)`` and are compared against ``dxc -P``. A case
the reference compiler rejects must be refused with a typed diagnostic; every other case must give the same
tokens. The shared C++ cases (``cases.CPP_CASES`` and ``cases.CPP_ERRORS``) run against both compilers as well.
"""

MSL_CASES = {
    "metal_version": (
        "metal3.1",
        "__METAL_VERSION__ __cplusplus __INTMAX_WIDTH__ __AIR_VERSION__\n",
    ),
    "metal4_cplusplus": ("metal4.0", "__METAL_VERSION__ __cplusplus\n"),
    "metal1_cplusplus": ("macos-metal1.1", "__METAL_VERSION__ __cplusplus\n"),
    "have_macros": (
        "metal3.1",
        "#if defined(__HAVE_BFLOAT__) && __HAVE_SIMDGROUP_FUTURE__ + 0 == 0\nbfloat\n#endif\n"
        "#ifdef __HAVE_TENSOR__\ntensor\n#endif\n",
    ),
    "intmax_is_128_bits": (
        "metal3.1",
        "#if 0xFFFFFFFFFFFFFFFF + 1 != 0\nwide\n#endif\n"
        "#if 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF > 0 && (1 << 100) != 0\nwider\n#endif\n"
        "#if -1 > 0u\nunsigned\n#endif\n",
    ),
    "operator_names": (
        "metal3.1",
        "#if 1 and not 0 or (2 bitand 3) xor 1\nyes\n#endif\n",
    ),
    "digit_separators_cxx14": ("metal3.1", "#if 1'000 == 1000\nseparated\n#endif\n"),
    "utf8_character_cxx17": ("metal4.0", "#if u8'a' == 97\nutf8\n#endif\n"),
    "no_raw_strings": ("metal4.0", '#define S(x) #x\nS(R"(a\\nb)") S(u8R"(c)")\n'),
    "digraphs": ("metal3.1", "%:define DG 5\nDG <::> <% %>\n"),
    "file_name": ("metal3.1", "__FILE_NAME__\n"),
    "ident": ("metal3.1", '#ident "version 1"\n#sccs "two"\nafter\n'),
    "va_opt_before_cxx20": (
        "metal3.1",
        "#define F(a,...) f(a __VA_OPT__(,) __VA_ARGS__)\nF(1) F(1,2)\n",
    ),
    "elifdef_before_cxx23": (
        "metal3.1",
        "#define A\n#if 0\n#elifdef A\nx\n#else\ny\n#endif\n",
    ),
    "header_macros_after_include": (
        "metal3.1",
        "assert(x) M_PI_F\n#include <metal_stdlib>\nassert(x) M_PI_F METAL_FUNC\n",
    ),
    "header_guard_after_include": (
        "metal3.1",
        "#include <metal_math>\n#undef M_PI_F\n#include <metal_stdlib>\n#ifdef M_PI_F\nredefined\n#endif\n",
    ),
    "has_include_metal_header": (
        "metal3.0",
        "#if __has_include(<metal_stdlib>) && __has_include(<simd/simd.h>) && !__has_include(<no_such_header>)\n"
        "found\n#endif\n",
    ),
    "user_macro_after_include": (
        "metal3.1",
        "#include <metal_stdlib>\n#define assert(x) mine(x)\nassert(1)\n",
    ),
    "error_operator_name_macro": ("metal3.1", "#define and 1\n"),
    "error_digit_separator_cxx11": ("macos-metal1.1", "#if 1'0 == 10\n#endif\n"),
    "error_utf8_character_cxx14": ("metal3.1", "#if u8'a' == 97\n#endif\n"),
    "error_has_embed": ("metal3.1", '#if __has_embed("x")\n#endif\n'),
    "error_ident_without_string": ("metal3.1", "#ident x\n"),
}

HLSL_CASES = {
    "profile": (
        "ps_6_6",
        (),
        "__SHADER_TARGET_STAGE __SHADER_TARGET_MAJOR __SHADER_TARGET_MINOR\n",
    ),
    "library": ("lib_6_3", (), "__SHADER_TARGET_STAGE __SHADER_STAGE_LIBRARY\n"),
    "mesh": (
        "ms_6_5",
        (),
        "#if __SHADER_TARGET_STAGE == __SHADER_STAGE_MESH\nmesh\n#endif\n",
    ),
    "shader_model_5": ("vs_5_1", (), "__SHADER_TARGET_MAJOR __SHADER_TARGET_MINOR\n"),
    "hlsl_version": ("cs_6_0", ("-HV", "2018"), "__HLSL_VERSION\n"),
    "hlsl_202x": ("cs_6_0", ("-HV", "202x"), "__HLSL_VERSION\n"),
    "sixteen_bit": (
        "cs_6_2",
        ("-enable-16bit-types",),
        "#ifdef __HLSL_ENABLE_16_BIT\nhalf __HLSL_ENABLE_16_BIT\n#endif\n",
    ),
    "no_sixteen_bit": (
        "cs_6_2",
        (),
        "#ifdef __HLSL_ENABLE_16_BIT\nhalf\n#else\nmin16\n#endif\n",
    ),
    "spirv": (
        "cs_6_0",
        ("-spirv", "-fspv-target-env=vulkan1.2"),
        "__spirv__ __SPIRV_MAJOR_VERSION__ __SPIRV_MINOR_VERSION__\n",
    ),
    "not_spirv": ("cs_6_0", (), "#ifdef __spirv__\nspirv\n#else\ndxil\n#endif\n"),
    "compiler_version": (
        "cs_6_0",
        (),
        "__hlsl_dx_compiler __DXC_VERSION_MAJOR __DXC_VERSION_MINOR __clang_major__\n",
    ),
    "no_cplusplus": (
        "cs_6_0",
        (),
        "#if defined(__cplusplus) || defined(__STDC__)\nc\n#else\nhlsl\n#endif\n",
    ),
    "intmax_is_64_bits": (
        "cs_6_0",
        (),
        "#if 0xFFFFFFFFFFFFFFFF + 1 == 0\nwraps\n#endif\n",
    ),
    "true_is_one": ("cs_6_0", (), "#if true && !false\nbool\n#endif\n"),
    "character_literal": ("cs_6_0", (), "#if 'a' == 97 && L'b' == 98\nchar\n#endif\n"),
    "va_opt_is_an_identifier": (
        "cs_6_0",
        (),
        "#define F(a,...) f(a __VA_OPT__(,) __VA_ARGS__)\nF(1) F(1,2)\n",
    ),
    "va_opt_can_be_defined": ("cs_6_0", (), "#define __VA_OPT__ 7\n__VA_OPT__\n"),
    "elifdef_skipped": (
        "cs_6_0",
        (),
        "#define A\n#if 0\n#elifdef A\nx\n#else\ny\n#endif\n",
    ),
    "operator_name_is_an_identifier": ("cs_6_0", (), "#define and 1\nand or\n"),
    "no_digraphs": ("cs_6_0", (), "%:define DG 5\nDG <::> <% %>\n"),
    "no_raw_strings": ("cs_6_0", (), '#define S(x) #x\nS(R"(a\\nb)")\n'),
    "dollar_identifier": ("cs_6_0", (), "#define A$B 3\nA$B\n"),
    "no_file_name_macro": ("cs_6_0", (), "__FILE_NAME__\n"),
    "gnu_comma": ("cs_6_0", (), "#define G(a,...) g(a, ## __VA_ARGS__)\nG(1) G(1,2)\n"),
    "pragma_once_operator": ("cs_6_0", (), '_Pragma("once") X\n'),
    "error_elifdef_active": ("cs_6_0", (), "#elifdef A\n"),
    "error_operator_name_in_if": ("cs_6_0", (), "#if 1 and 1\n#endif\n"),
    "error_digit_separator": ("cs_6_0", (), "#if 1'000 == 1000\n#endif\n"),
    "error_utf8_character": ("cs_6_0", (), "#if u8'a' == 97\n#endif\n"),
    "error_ident": ("cs_6_0", (), '#ident "x"\n'),
    "error_digraph_paste": ("cs_6_0", (), "#define CAT(a,b) a##b\nCAT(<,:)\n"),
}
