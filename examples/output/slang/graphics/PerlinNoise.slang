struct VertexInput
{
    VectorType(element_type=PrimitiveType(name=float, size_bits=None), size=3) position;
};

struct VertexOutput
{
    VectorType(element_type=PrimitiveType(name=float, size_bits=None), size=2) uv;
    VectorType(element_type=PrimitiveType(name=float, size_bits=None), size=4) position;
};

struct FragmentInput
{
    VectorType(element_type=PrimitiveType(name=float, size_bits=None), size=2) uv;
};

struct FragmentOutput
{
    VectorType(element_type=PrimitiveType(name=float, size_bits=None), size=4) color;
};

