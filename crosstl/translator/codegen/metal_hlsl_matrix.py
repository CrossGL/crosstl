"""Metal emulation of matrices that Metal has no native type for.

Metal only provides floating-point matrices; HLSL also has ``int``, ``uint`` and
``bool`` matrices, which CrossGL keeps under their HLSL names (``int2x3`` has
two rows and three columns, the integer counterpart of CrossGL ``mat3x2``). They become values
of ``crosstl_hlsl_matrix<T, Columns, Rows>``, which mirrors Metal's
``matrix<T, Columns, Rows>`` exactly: an array of column vectors, ``m[i]`` is
column ``i``, constructors fill whole columns, ``*`` with a vector or matrix
is the linear-algebra product and the other operators work element-wise. A
CrossGL ``int2x3`` therefore maps to ``crosstl_int3x2`` just as ``mat3x2`` maps
to ``float3x2``.
"""

import re

HLSL_EMULATED_MATRIX_PATTERN = re.compile(
    r"(?P<scalar>int|uint|bool)(?P<rows>[2-4])x(?P<columns>[2-4])"
)


def hlsl_emulated_matrix_shape(type_name):
    """``(scalar, rows, columns)`` of a CrossGL matrix Metal has to emulate."""
    match = HLSL_EMULATED_MATRIX_PATTERN.fullmatch(str(type_name).strip())
    if match is None:
        return None
    return (
        match.group("scalar"),
        int(match.group("rows")),
        int(match.group("columns")),
    )


def hlsl_emulated_matrix_type_name(shape):
    """The Metal alias, spelled like Metal matrices: columns before rows."""
    scalar, rows, columns = shape
    return f"crosstl_{scalar}{columns}x{rows}"


METAL_HLSL_MATRIX_PRELUDE = """\
template <typename T, int C, int R>
struct crosstl_hlsl_matrix {
    vec<T, R> columns[C];
    thread vec<T, R>& operator[](int i) thread { return columns[i]; }
    vec<T, R> operator[](int i) const thread { return columns[i]; }
    vec<T, R> operator[](int i) const constant { return columns[i]; }
    vec<T, R> operator[](int i) const device { return columns[i]; }
    vec<T, R> operator[](int i) const threadgroup { return columns[i]; }
};

template <typename T, int C, int R, typename S>
crosstl_hlsl_matrix<T, C, R> crosstl_hlsl_matrix_splat(S value) {
    crosstl_hlsl_matrix<T, C, R> result;
    for (int i = 0; i < C; ++i) result.columns[i] = vec<T, R>(T(value));
    return result;
}

template <typename T, typename U, int C, int R>
crosstl_hlsl_matrix<T, C, R> crosstl_hlsl_matrix_cast(crosstl_hlsl_matrix<U, C, R> value) {
    crosstl_hlsl_matrix<T, C, R> result;
    for (int i = 0; i < C; ++i) result.columns[i] = vec<T, R>(value.columns[i]);
    return result;
}

template <typename T, typename U, int C, int R>
crosstl_hlsl_matrix<T, C, R> crosstl_hlsl_matrix_cast(matrix<U, C, R> value) {
    crosstl_hlsl_matrix<T, C, R> result;
    for (int i = 0; i < C; ++i) result.columns[i] = vec<T, R>(value[i]);
    return result;
}

template <typename T, typename U, int C, int R>
matrix<T, C, R> crosstl_hlsl_matrix_native(crosstl_hlsl_matrix<U, C, R> value) {
    matrix<T, C, R> result;
    for (int i = 0; i < C; ++i) result[i] = vec<T, R>(value.columns[i]);
    return result;
}

#define CROSSTL_HLSL_MATRIX_ELEMENTWISE(OP) \\
template <typename T, typename U, int C, int R> \\
crosstl_hlsl_matrix<T, C, R> operator OP(crosstl_hlsl_matrix<T, C, R> a, crosstl_hlsl_matrix<U, C, R> b) { \\
    for (int i = 0; i < C; ++i) a.columns[i] = a.columns[i] OP vec<T, R>(b.columns[i]); \\
    return a; \\
}
#define CROSSTL_HLSL_MATRIX_SCALAR(OP) \\
template <typename T, int C, int R, typename S> \\
crosstl_hlsl_matrix<T, C, R> operator OP(crosstl_hlsl_matrix<T, C, R> a, S b) { \\
    for (int i = 0; i < C; ++i) a.columns[i] = a.columns[i] OP vec<T, R>(T(b)); \\
    return a; \\
} \\
template <typename T, int C, int R, typename S> \\
crosstl_hlsl_matrix<T, C, R> operator OP(S a, crosstl_hlsl_matrix<T, C, R> b) { \\
    for (int i = 0; i < C; ++i) b.columns[i] = vec<T, R>(T(a)) OP b.columns[i]; \\
    return b; \\
} \\
template <typename T, int C, int R, typename S> \\
thread crosstl_hlsl_matrix<T, C, R>& operator OP##=(thread crosstl_hlsl_matrix<T, C, R>& a, S b) { \\
    a = a OP b; \\
    return a; \\
}
#define CROSSTL_HLSL_MATRIX_BINARY(OP) \\
CROSSTL_HLSL_MATRIX_ELEMENTWISE(OP) \\
CROSSTL_HLSL_MATRIX_SCALAR(OP)
CROSSTL_HLSL_MATRIX_BINARY(+)
CROSSTL_HLSL_MATRIX_BINARY(-)
CROSSTL_HLSL_MATRIX_BINARY(/)
CROSSTL_HLSL_MATRIX_BINARY(%)
CROSSTL_HLSL_MATRIX_BINARY(&)
CROSSTL_HLSL_MATRIX_BINARY(|)
CROSSTL_HLSL_MATRIX_BINARY(^)
CROSSTL_HLSL_MATRIX_BINARY(<<)
CROSSTL_HLSL_MATRIX_BINARY(>>)
CROSSTL_HLSL_MATRIX_SCALAR(*)
#undef CROSSTL_HLSL_MATRIX_BINARY
#undef CROSSTL_HLSL_MATRIX_SCALAR
#undef CROSSTL_HLSL_MATRIX_ELEMENTWISE

// Linear-algebra products, as for Metal matrix<T, C, R>.
template <typename T, int C, int R>
vec<T, R> operator*(crosstl_hlsl_matrix<T, C, R> m, vec<T, C> v) {
    vec<T, R> result = vec<T, R>(T(0));
    for (int i = 0; i < C; ++i) result += m.columns[i] * v[i];
    return result;
}

template <typename T, int C, int R>
vec<T, C> operator*(vec<T, R> v, crosstl_hlsl_matrix<T, C, R> m) {
    vec<T, C> result;
    for (int j = 0; j < C; ++j) {
        T sum = T(0);
        for (int k = 0; k < R; ++k) sum += v[k] * m.columns[j][k];
        result[j] = sum;
    }
    return result;
}

template <typename T, int K, int C, int R>
crosstl_hlsl_matrix<T, C, R> operator*(crosstl_hlsl_matrix<T, K, R> a, crosstl_hlsl_matrix<T, C, K> b) {
    crosstl_hlsl_matrix<T, C, R> result;
    for (int j = 0; j < C; ++j) result.columns[j] = a * b.columns[j];
    return result;
}

#define CROSSTL_HLSL_MATRIX_COMPARE(OP) \\
template <typename T, typename U, int C, int R> \\
crosstl_hlsl_matrix<bool, C, R> operator OP(crosstl_hlsl_matrix<T, C, R> a, crosstl_hlsl_matrix<U, C, R> b) { \\
    crosstl_hlsl_matrix<bool, C, R> result; \\
    for (int i = 0; i < C; ++i) result.columns[i] = a.columns[i] OP vec<T, R>(b.columns[i]); \\
    return result; \\
} \\
template <typename T, int C, int R, typename S> \\
crosstl_hlsl_matrix<bool, C, R> operator OP(crosstl_hlsl_matrix<T, C, R> a, S b) { \\
    crosstl_hlsl_matrix<bool, C, R> result; \\
    for (int i = 0; i < C; ++i) result.columns[i] = a.columns[i] OP vec<T, R>(T(b)); \\
    return result; \\
} \\
template <typename T, int C, int R, typename S> \\
crosstl_hlsl_matrix<bool, C, R> operator OP(S a, crosstl_hlsl_matrix<T, C, R> b) { \\
    crosstl_hlsl_matrix<bool, C, R> result; \\
    for (int i = 0; i < C; ++i) result.columns[i] = vec<T, R>(T(a)) OP b.columns[i]; \\
    return result; \\
}
CROSSTL_HLSL_MATRIX_COMPARE(==)
CROSSTL_HLSL_MATRIX_COMPARE(!=)
CROSSTL_HLSL_MATRIX_COMPARE(<)
CROSSTL_HLSL_MATRIX_COMPARE(<=)
CROSSTL_HLSL_MATRIX_COMPARE(>)
CROSSTL_HLSL_MATRIX_COMPARE(>=)
#undef CROSSTL_HLSL_MATRIX_COMPARE

#define CROSSTL_HLSL_MATRIX_UNARY(OP) \\
template <typename T, int C, int R> \\
crosstl_hlsl_matrix<T, C, R> operator OP(crosstl_hlsl_matrix<T, C, R> a) { \\
    for (int i = 0; i < C; ++i) a.columns[i] = OP a.columns[i]; \\
    return a; \\
}
CROSSTL_HLSL_MATRIX_UNARY(-)
CROSSTL_HLSL_MATRIX_UNARY(+)
CROSSTL_HLSL_MATRIX_UNARY(~)
#undef CROSSTL_HLSL_MATRIX_UNARY

template <int C, int R>
crosstl_hlsl_matrix<bool, C, R> operator!(crosstl_hlsl_matrix<bool, C, R> a) {
    for (int i = 0; i < C; ++i) a.columns[i] = !a.columns[i];
    return a;
}

template <typename T, typename U, int C, int R>
crosstl_hlsl_matrix<bool, C, R> crosstl_hlsl_and(crosstl_hlsl_matrix<T, C, R> a, crosstl_hlsl_matrix<U, C, R> b) {
    crosstl_hlsl_matrix<bool, C, R> result;
    for (int i = 0; i < C; ++i) result.columns[i] = vec<bool, R>(a.columns[i]) && vec<bool, R>(b.columns[i]);
    return result;
}

template <typename T, typename U, int C, int R>
crosstl_hlsl_matrix<bool, C, R> crosstl_hlsl_or(crosstl_hlsl_matrix<T, C, R> a, crosstl_hlsl_matrix<U, C, R> b) {
    crosstl_hlsl_matrix<bool, C, R> result;
    for (int i = 0; i < C; ++i) result.columns[i] = vec<bool, R>(a.columns[i]) || vec<bool, R>(b.columns[i]);
    return result;
}

// Metal argument order: select(a, b, c) returns c ? b : a for each element.
template <typename T, int C, int R>
crosstl_hlsl_matrix<T, C, R> select(crosstl_hlsl_matrix<T, C, R> a, crosstl_hlsl_matrix<T, C, R> b, crosstl_hlsl_matrix<bool, C, R> c) {
    for (int i = 0; i < C; ++i) a.columns[i] = select(a.columns[i], b.columns[i], c.columns[i]);
    return a;
}

template <typename T, int C, int R>
bool all(crosstl_hlsl_matrix<T, C, R> a) {
    bool result = true;
    for (int i = 0; i < C; ++i) result = result && all(vec<bool, R>(a.columns[i]));
    return result;
}

template <typename T, int C, int R>
bool any(crosstl_hlsl_matrix<T, C, R> a) {
    bool result = false;
    for (int i = 0; i < C; ++i) result = result || any(vec<bool, R>(a.columns[i]));
    return result;
}

template <typename T, int C, int R>
crosstl_hlsl_matrix<T, R, C> transpose(crosstl_hlsl_matrix<T, C, R> a) {
    crosstl_hlsl_matrix<T, R, C> result;
    for (int i = 0; i < R; ++i)
        for (int j = 0; j < C; ++j) result.columns[i][j] = a.columns[j][i];
    return result;
}

"""


def metal_hlsl_matrix_prelude(shapes):
    """The emulation template followed by one alias for each used shape."""
    aliases = "".join(
        f"typedef crosstl_hlsl_matrix<{scalar}, {columns}, {rows}> "
        f"{hlsl_emulated_matrix_type_name((scalar, rows, columns))};\n"
        for scalar, rows, columns in sorted(shapes)
    )
    return METAL_HLSL_MATRIX_PRELUDE + aliases + "\n"
