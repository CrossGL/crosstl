"""Preprocessor cases shared by the expected-output and the differential tests.

Each case is self-contained source text. ``CPP_CASES`` are valid C++20 and compare against Clang's preprocessor
(``clang -E -P -x c++ -std=c++20``), the preprocessor of DXC and of the Metal compiler. ``GLSL_CASES`` compare
against ``glslangValidator -E``. ``CPP_ERRORS`` and ``GLSL_ERRORS`` must be refused, with the diagnostic code
given, by both the reference and the shared preprocessor.
"""

CPP_CASES = {
    # C11 6.10.3.5 EXAMPLE 3.
    "standard_example_3": (
        r"""
#define x 3
#define f(a) f(x * (a))
#undef x
#define x 2
#define g f
#define z z[0]
#define h g(~
#define m(a) a(w)
#define w 0,1
#define t(a) a
#define p() int
#define q(x) x
#define r(x,y) x ## y
#define str(x) # x
f(y+1) + f(f(z)) % t(t(g)(0) + t)(1);
g(x+(3,4)-w) | h 5) & m
(f)^m(m);
p() i[q()] = { q(1), r(2,3), r(4,), r(,5), r(,) };
char c[2][6] = { str(hello), str() };
"""
    ),
    # C11 6.10.3.5 EXAMPLE 4.
    "standard_example_4": (
        r"""
#define str(s) # s
#define xstr(s) str(s)
#define debug(s, t) printf("x" # s "= %d, x" # t "= %s", \
 x ## s, x ## t)
#define INCFILE(n) vers ## n
#define glue(a, b) a ## b
#define xglue(a, b) glue(a, b)
#define HIGHLOW "hello"
#define LOW LOW ", world"
debug(1, 2);
fputs(str(strncmp("abc\0d", "abc", '\4') // this goes away
 == 0) str(: @\n), s);
xstr(INCFILE(2).h)
glue(HIGH, LOW);
xglue(HIGH, LOW)
"""
    ),
    # C11 6.10.3.5 EXAMPLE 5: placemarkers.
    "standard_example_5": (
        r"""
#define t(x,y,z) x ## y ## z
int j[] = { t(1,2,3), t(,4,5), t(6,,7), t(8,9,),
 t(10,,), t(,11,), t(,,12), t(,,) };
"""
    ),
    # C11 6.10.3.5 EXAMPLE 7: variable arguments.
    "standard_example_7": (
        r"""
#define debug(...) fprintf(stderr, __VA_ARGS__)
#define showlist(...) puts(#__VA_ARGS__)
#define report(test, ...) ((test)?puts(#test):\
 printf(__VA_ARGS__))
debug("Flag");
debug("X = %d\n", x);
showlist(The first, second, and third items.);
report(x>y, "x is %d but y is %d", x, y);
"""
    ),
    # C11 6.10.3.3 EXAMPLE: '##' produced by '# ## #' is not an operator.
    "hash_hash": (
        r"""
#define hash_hash # ## #
#define mkstr(a) # a
#define in_between(a) mkstr(a)
#define join(c, d) in_between(c hash_hash d)
char p[] = join(x, y);
"""
    ),
    # C++20 [cpp.subst] examples for __VA_OPT__.
    "va_opt": (
        r"""
#define F(...) f(0 __VA_OPT__(,) __VA_ARGS__)
#define G(X, ...) f(0, X __VA_OPT__(,) __VA_ARGS__)
#define SDEF(sname, ...) S sname __VA_OPT__(= { __VA_ARGS__ })
#define EMP
F(a,b,c)
F()
F(EMP)
G(a,b,c)
G(a,)
G(a)
SDEF(foo);
SDEF(bar, 1, 2);
#define H2(X, Y, ...) __VA_OPT__(X ## Y,) __VA_ARGS__
H2(a, b, c, d)
#define H3(X, ...) #__VA_OPT__(X##X X##X)
H3(, 0)
#define H4(X, ...) __VA_OPT__(a X ## X) ## b
H4(, 1)
#define H5A(...) __VA_OPT__()/**/__VA_OPT__()
#define H5B(X) a ## X ## b
#define H5C(X) H5B(X)
H5C(H5A())
"""
    ),
    "gnu_comma_elision": (
        r"""
#define e(fmt, ...) printf(fmt, ## __VA_ARGS__)
#define n(fmt, args...) printf(fmt, ## args)
e("a");
e("a", 1, 2);
n("b");
n("b", 3);
"""
    ),
    "nested_arguments": (
        r"""
#define M(a, b) [a|b]
#define ID(x) x
#define TWO 1, 2
M((1, 2), (3, 4))
M(ID(1), ID((5, 6)))
M(f(1, 2), x)
ID(M)(7, 8)
ID(M(TWO, 3))
M(
  multi,
  line
)
"""
    ),
    "painted_blue": (
        r"""
#define foo foo
#define a b
#define b a
#define f(x) x + f(x)
#define g(x) h(x)
#define h(x) g(x)
foo a b f(1) g(2)
#define AA BB
#define BB AA
AA BB
#define NIL(x) x
#define G_0(arg) NIL(G_1)(arg)
#define G_1(arg) NIL(arg)
G_0(42)
"""
    ),
    "deferred_invocation": (
        r"""
#define EMPTY
#define DEFER(id) id EMPTY
#define EXPAND(...) __VA_ARGS__
#define A() 123
DEFER(A)()
EXPAND(DEFER(A)())
#define LPAREN (
#define FN(x) [x]
FN LPAREN 1)
#define CALL(m) m(4)
CALL(FN)
"""
    ),
    "rescan_with_following_tokens": (
        r"""
#define f(a) a*g
#define g(a) f(a)
f(2)(9)
#define q(x) x q
q(1)(2)(3)
#define obj fn
#define fn(x) <x>
obj(5)
"""
    ),
    "stringize_spacing": (
        r"""
#define S(x) #x
#define XS(x) S(x)
S(  a   +    b  )
S("x\n" '\'' "\\")
S(  )
S(a
b)
S(/* comment */ c /* another */ d)
XS(__LINE__)
S(L"wide" u8"utf")
"""
    ),
    "token_pasting": (
        r"""
#define cat(a, b) a ## b
#define xcat(a, b) cat(a, b)
cat(+, +) cat(-, =) cat(<<, =) cat(x, 1) cat(1, e) cat(1e, 5) cat(., 5)
cat(L, 'a') cat(u8, "s") cat(, x) cat(y, ) cat(,)
xcat(xcat(1, 2), 3)
#define VAR(n) var_ ## n ## _end
VAR(3) VAR()
cat(%:, %:)
"""
    ),
    "object_vs_function_like": (
        r"""
#define OBJ (x) x
#define FUN(x) x
OBJ FUN(3) OBJ(4)
#define FUN2(x) FUN
FUN2(1)(2)
"""
    ),
    "conditionals": (
        r"""
#define ONE 1
#define ZERO 0
#if ONE
one
#elif 1/0
never
#else
never
#endif
#if ZERO
#error not reached
#elif defined ONE && !defined(TWO)
two
#endif
#ifdef ONE
three
#endif
#ifndef TWO
four
#endif
#if 0
don't lex this ' as an error
#if garbage (
#endif
#else
five
#endif
#if defined(ONE) ? ONE : 1/0
six
#endif
#ifdef UNDEFINED
#elifdef ONE
seven
#elifndef ONE
#endif
"""
    ),
    "if_arithmetic": (
        r"""
#if -1 < 0u
no
#else
unsigned_compare
#endif
#if 18446744073709551615 == -1
max_unsigned
#endif
#if 0x7fffffffffffffff > 0 && 0x8000000000000000 > 0
hex_unsigned
#endif
#if (2 || 1 / 0) && !(0 && 1 / 0)
short_circuit
#endif
#if (1 ? -1 : 0u) > 0
conditional_unsigned
#endif
#if 'a' == 97 && '\377' < 0 && '\n' == 10 && 'ab' == 24930
characters
#endif
#if true && !false
cpp_bool
#endif
#if 7 / -2 == -3 && 7 % -2 == 1 && -7 / 2 == -3 && -7 % 2 == -1
truncation
#endif
#if (-1 >> 1) == -1 && (1 << 62) > 0 && ~0u == 18446744073709551615u
shifts
#endif
#if 1'000 == 1000 && 0b101 == 5 && 017 == 15 && 10ull == 10 && 5LL == 5
literals
#endif
#if UNDEFINED_NAME == 0 && (UNDEFINED_NAME + 3) == 3
undefined_is_zero
#endif
#if (1, 2) == 2
comma
#endif
"""
    ),
    "builtin_macros": (
        r"""
a __LINE__
#define L __LINE__
b L
#define M(x) x __LINE__
M(
1
)
c __COUNTER__ __COUNTER__ __INCLUDE_LEVEL__
#line 100
d __LINE__
#line 7 "renamed.c"
e __LINE__ __FILE__
"""
    ),
    "pragma_operator_once": (
        r"""_Pragma("once") X
#define ONCE _Pragma("once")
ONCE Y
_Pragma("omp parallel") Z
"""
    ),
    "pragmas": (
        r"""
#pragma pack_matrix(row_major)
#pragma STDC FP_CONTRACT OFF
_Pragma("clang diagnostic push")
#define DO_PRAGMA(x) _Pragma(#x)
DO_PRAGMA(message("hello"))
after
"""
    ),
    "digraphs_and_splices": (
        r"""
%:define DIGRAPH <: 1 :> <% %>
DIGRAPH
#define LONG 1 + \
2 + \
  3
LONG
#define C1 /* comment */ 5
#define C2 1 /* multi
line */ + 2
C1 C2
#
# /* null directive */
#define EMPTY
EMPTY # define NOT_A_DIRECTIVE 1
NOT_A_DIRECTIVE
"""
    ),
    "redefinition_and_undef": (
        r"""
#define X 1
#define X 1
#define Y(a) a
#define Y(a) a
#undef X
#define X 2
X Y(3)
#undef NEVER_DEFINED
"""
    ),
    "includes_absent": (
        r"""
#if __has_include("no_such_header.h") || __has_include(<no_such_header.h>)
found
#else
absent
#endif
#ifdef __has_include
has_include_defined
#endif
"""
    ),
    "avoid_accidental_pasting": (
        r"""
#define PLUS +
#define MINUS -
#define E
+PLUS -MINUS PLUS+ x E y 1 E.5 a/E/b < E<
#define NEG -1
-NEG
"""
    ),
    "line_in_multiline_invocations": (
        r"""#define M(x) x __LINE__
M(
__LINE__
)
#define N(x) x
N(
__LINE__
)
#define O() __LINE__
O(
)
#define P O
P(

)
"""
    ),
}

# Files of the include test: the main file includes through every search rule.
INCLUDE_TREE = {
    "main.cpp": (
        r"""#include "local.h"
#include <system.h>
#include "sub/nested.h"
#define HEADER <system.h>
#include HEADER
#include "once.h"
#include "once.h"
#include "guarded.h"
#include "guarded.h"
#if __has_include("local.h") && __has_include(<system.h>) && !__has_include("missing.h")
has_include_works
#endif
main_line __LINE__ __INCLUDE_LEVEL__
"""
    ),
    "local.h": "local_header __INCLUDE_LEVEL__\n",
    "inc/system.h": "system_header\n#include_next <system.h>\n",
    "inc2/system.h": "next_system_header\n",
    "sub/nested.h": '#include "sibling.h"\nnested_header\n',
    "sub/sibling.h": "sibling_header __LINE__\n",
    "once.h": "#pragma once\nonce_header\n",
    "guarded.h": "#ifndef GUARDED_H\n#define GUARDED_H\nguarded_header\n#endif\n",
}

CPP_ERRORS = {
    "invalid_paste": (
        "#define cat(a, b) a ## b\ncat(., .)\n",
        "project.translate.preprocessor-invalid-paste",
    ),
    "unterminated_invocation": (
        "#define f(x) x\nf(1, \n",
        "project.translate.preprocessor-macro-arguments-invalid",
    ),
    "too_many_arguments": (
        "#define f(x) x\nf(1, 2)\n",
        "project.translate.preprocessor-macro-arguments-invalid",
    ),
    "too_few_arguments": (
        "#define f(x, y) x\nf(1)\n",
        "project.translate.preprocessor-macro-arguments-invalid",
    ),
    "error_directive": (
        "#error stop here\n",
        "project.translate.preprocessor-error-directive",
    ),
    "unterminated_if": (
        "#if 1\nx\n",
        "project.translate.preprocessor-unbalanced-conditional",
    ),
    "else_after_else": (
        "#if 1\n#else\n#else\n#endif\n",
        "project.translate.preprocessor-unbalanced-conditional",
    ),
    "endif_without_if": (
        "#endif\n",
        "project.translate.preprocessor-unbalanced-conditional",
    ),
    "division_by_zero": (
        "#if 1 / 0\n#endif\n",
        "project.translate.preprocessor-if-division-by-zero",
    ),
    "missing_include": (
        '#include "no_such_header.h"\n',
        "project.translate.preprocessor-include-not-found",
    ),
    "stringize_non_parameter": (
        "#define s(x) #y\n",
        "project.translate.preprocessor-invalid-macro-definition",
    ),
    "paste_at_edge": (
        "#define p ## x\n",
        "project.translate.preprocessor-invalid-macro-definition",
    ),
    "define_defined": (
        "#define defined 1\n",
        "project.translate.preprocessor-invalid-macro-definition",
    ),
    "invalid_directive": (
        "#frobnicate\n",
        "project.translate.preprocessor-invalid-directive",
    ),
    "float_in_if": (
        "#if 1.0\n#endif\n",
        "project.translate.preprocessor-if-expression-invalid",
    ),
    "unterminated_comment": (
        "x /* never closed\n",
        "project.translate.preprocessor-invalid-token",
    ),
}

GLSL_CASES = {
    "glsl_version_and_profile": (
        r"""#version 450 core
#if __VERSION__ >= 450 && defined(GL_core_profile)
modern
#endif
#ifdef GL_ES
never
#endif
#define SCALE(x) ((x) * 2)
float f = SCALE(1.0);
"""
    ),
    "glsl_es": (
        r"""#version 300 es
#if GL_ES == 1 && __VERSION__ == 300
es
#endif
precision highp float;
"""
    ),
    "glsl_arithmetic": (
        r"""#version 450
#if 2147483647 + 1 < 0
wraps
#endif
#if (7 / -2) == -3 && (-7 % 2) == -1
truncates
#endif
#if (1 << 31) < 0
sign_bit
#endif
"""
    ),
    "glsl_extension_and_pragma": (
        r"""#version 460
#extension GL_EXT_shader_explicit_arithmetic_types : enable
#pragma optimize(off)
#define CAT(a, b) a ## b
int CAT(value, 1) = __LINE__;
"""
    ),
}

GLSL_ERRORS = {
    "glsl_reserved_name": (
        "#version 450\n#define GL_custom 1\n",
        "project.translate.preprocessor-invalid-macro-definition",
    ),
    "glsl_redefinition": (
        "#version 450\n#define A 1\n#define A 2\n",
        "project.translate.preprocessor-invalid-macro-definition",
    ),
    "glsl_division_by_zero": (
        "#version 450\n#if 1 / 0\n#endif\n",
        "project.translate.preprocessor-if-division-by-zero",
    ),
}
