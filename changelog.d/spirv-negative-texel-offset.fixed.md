## Fixed

- **SPIR-V codegen**: Negated integer literal texel offsets (e.g. `textureOffset(line, u, -1)`,
  `textureOffset(colorMap, uv, -ivec2(1, 0))`, `textureOffset(colorMap, uv, -(ivec2(3, -4)))`)
  now lower to a `ConstOffset` operand instead of being refused with
  `non-constant texel offset is valid only on gather operations`.  Previously the codegen emitted
  `OpSNegate` for the negated constant, which `is_constant_instruction` did not recognise as a
  constant, violating VUID-RuntimeSpirv-Offset-10213.  The fix extends
  `literal_integer_vector_constant` / `process_call_argument` in `SpirvCodegen` to fold scalar and
  negated-vector-constructor offset arguments into genuine `OpConstant` / `OpConstantComposite`
  instructions without touching `unary_operation`.  HLSL `Texture1D.Sample`,
  `Texture1D.SampleCmp` and `Texture1DArray.SampleCmpLevelZero` with negative integer offsets are
  also fixed.
