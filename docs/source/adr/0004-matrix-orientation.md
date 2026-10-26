# ADR 0004: CrossGL matrices follow GLSL: `matCxR` has C columns and R rows

- Status: accepted
- Date: 2026-10-19
- Scope: `crosstl/translator/matrix_shape.py`, `crosstl/translator/parser.py` (`MatrixType`),
  `crosstl/translator/codegen/hlsl_matrix_orientation.py`, `crosstl/translator/codegen/metal_matrix_builtins.py`,
  the matrix paths of every emitter, and the GLSL importer

## Context

The languages spell and index matrices differently:

| Language | Two columns of three rows | `m[i]` | Constructor arguments fill | `m * v` | Reference |
|---|---|---|---|---|---|
| GLSL, GLSL ES | `mat2x3` | column `i` | columns | linear algebra | GLSL 4.60 §4.1.6, §5.4.2, §5.6 |
| MSL | `float2x3` | column `i` | columns | linear algebra | MSL 4.1 §2.3 |
| SPIR-V | `OpTypeMatrix %v3float 2` | column `i` | columns | `OpMatrixTimesVector` | SPIR-V 1.6 §2.2.2 |
| HLSL | `float3x2` | row `i` | rows | componentwise; `mul(m, v)` is linear algebra | HLSL matrix type, `mul` reference |

CrossGL had no stated convention. The parser stored the first number of `matAxB` as `rows`, the GLSL emitter then
printed `mat{cols}x{rows}`, so CrossGL `mat2x3` came out as GLSL `mat3x2`; the Metal emitter mapped `mat2x3` to
`float3x2`; the HLSL emitter mapped the type correctly but kept `m[i]`, the constructor arguments and `matN(s)`
unchanged, so a constant column became a row, a constructor filled rows, and `mat4(1.0)` became an all-ones matrix.
Values read from a GLSL storage block into HLSL were built with the columns as rows.

## Decision

CrossGL matrices have GLSL's meaning. `matCxR` (and `dmatCxR`, `f16matCxR`) has C columns of R-component vectors,
`MatrixType.cols` is C and `MatrixType.rows` is R, constructors fill columns, `matN(s)` puts `s` on the diagonal,
`m[i]` is column `i`, `m.length()` is the column count, `m * v`, `v * m` and `m * n` are linear-algebra products,
and `{...}` initializers list columns, a flat list filling the columns in order.

Per target:

- GLSL, GLSL ES and SPIR-V use the same spelling and meaning.
- MSL spells `matCxR` as `floatCxR` (`halfCxR`); everything else is the same.
- HLSL declares `matCxR` as `floatRxC`, the same matrix with R rows and C columns, so products stay `mul` with the
  operands in order and HLSL's default `column_major` packing stores the columns as std140 and std430 do.
  `hlsl_matrix_orientation.py` lowers the rest: a constant column `m[i]` becomes the matrix swizzle
  `m._m0i_m1i...` (also an lvalue), a dynamic column read `transpose(m)[i]`, a dynamic column write a setter helper,
  `m[i][j]` becomes `m[j][i]`, a constructor with several arguments `transpose(floatCxR(...))`, `matCxR(s)` a
  diagonal matrix, `matrixCompMult` HLSL's componentwise `*`, and `outerProduct` a `mul` of a column and a row.
  Storage-block matrices load as `transpose(floatCxR(column loads))` and store each column through its swizzle.

## Consequences

- CrossGL text that used `matAxB` with the old, row-first reading changes meaning; in-repo fixtures that relied on it
  were updated. There is no compatibility switch (the IR has no compatibility guarantee).
- The GLSL importer passes `matCxR` through unchanged, and no longer renames the ray-query transforms
  (`mat4x3`, GLSL_EXT_ray_query) to `mat3x4`.
- The HLSL importer (`crosstl/backend/DirectX/hlsl_matrix_import.py`) maps `floatRxC` to `matCxR`. A row read
  `m[i]` becomes `transpose(m)[i]`, `m[i][j]` becomes `m[j][i]`, a row write calls a setter helper, constructors,
  brace initializers and vector casts become `transpose(matRxC(...))`, `m._mRC` and `m._RC` become `m[C][R]`,
  `*` of two matrices becomes `matrixCompMult`, a matrix cast keeps the upper-left elements and a scalar cast fills
  every element. Row writes and multi-component swizzle writes whose operands have side effects are refused.
- With row-major packing (`matrix_packing="row_major"`, DXC `-Zpr`) the HLSL importer represents each HLSL matrix
  by its transpose, so that CrossGL's column-major storage holds the bytes of the HLSL rows: `floatRxC` becomes
  `matRxC`, `m[i]`, `m[i][j]`, `m._mRC`, constructors and initializers keep their order, and `mul(a, b)` becomes
  `b * a`.
- Integer and boolean matrices, which only HLSL has, keep HLSL's row-first spelling in CrossGL with the column
  indexing above: `int3x2` has three rows and two columns, `m[i]` is one of its two `ivec3` columns, and it is the
  integer counterpart of `mat2x3`. The GLSL and Metal emitters emulate them as structs of columns.
- MSL takes only whole columns or one scalar per element in a matrix constructor, so the Metal emitter spreads the
  components of a mixed argument list, such as `mat2(vec4)`, through a helper.
- `floatNxM` names a different matrix in MSL (N columns) and HLSL (N rows), so CrossGL refuses the non-square
  `floatNxM` and `doubleNxM` spellings (`crosstl/translator/semantics/matrix_spellings.py`) and the importers write
  `matCxR`. Square `floatNxN` and `doubleNxN` are aliases of `matN` and `dmatN`.
- Still open: Slang keeps its previous spelling until it gets the HLSL lowering; its non-square matrices are refused.
