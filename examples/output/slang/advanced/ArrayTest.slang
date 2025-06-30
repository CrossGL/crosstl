struct VertexInput
{
    VectorType(element_type=PrimitiveType(name=float, size_bits=None), size=3) position;
    VectorType(element_type=PrimitiveType(name=float, size_bits=None), size=2) texCoord;
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

struct TestBuffer
{
    ArrayType(element_type=PrimitiveType(name=float, size_bits=None), size=LiteralNode(value=4, literal_type=PrimitiveType(name=int, size_bits=None))) values;
    ArrayType(element_type=VectorType(element_type=PrimitiveType(name=float, size_bits=None), size=3), size=LiteralNode(value=2, literal_type=PrimitiveType(name=int, size_bits=None))) colors;
};

