"""HIP AST Node definitions"""

from ..common_ast import (
    ASTNode,
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    BreakNode,
    ContinueNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IfNode,
    MemberAccessNode,
    PreprocessorNode,
    ReturnNode,
    ShaderNode,
    StructNode,
    SyncNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)

# HIP-specific nodes (very similar to CUDA)


class KernelNode(FunctionNode):
    """Node representing a HIP kernel function (marked with __global__)"""

    def __init__(self, return_type, name, params, body, attributes=None):
        super().__init__(return_type, name, params, body, ["__global__"], attributes)

    def __repr__(self):
        return f"KernelNode(name={self.name}, params={self.params}, body={self.body})"


class KernelLaunchNode(ASTNode):
    """Node representing a kernel launch: kernel<<<blocks, threads>>>(args)"""

    def __init__(
        self, kernel_name, blocks, threads, shared_mem=None, stream=None, args=None
    ):
        self.kernel_name = kernel_name
        self.blocks = blocks
        self.threads = threads
        self.shared_mem = shared_mem
        self.stream = stream
        self.args = args or []

    def __repr__(self):
        return f"KernelLaunchNode(kernel_name={self.kernel_name}, blocks={self.blocks}, threads={self.threads}, args={self.args})"


class AtomicOperationNode(FunctionCallNode):
    """Node representing a HIP atomic operation"""

    def __init__(self, operation, args):
        super().__init__(operation, args)
        self.operation = operation  # atomicAdd, hipAtomicAdd, etc.

    def __repr__(self):
        return f"AtomicOperationNode(operation={self.operation}, args={self.args})"


class HipBuiltinNode(ASTNode):
    """Node representing HIP built-in variables (threadIdx, hipThreadIdx_x, etc.)"""

    def __init__(self, builtin_name, component=None):
        self.builtin_name = builtin_name
        self.component = component

    def __repr__(self):
        if self.component:
            return f"HipBuiltinNode(builtin_name={self.builtin_name}, component={self.component})"
        return f"HipBuiltinNode(builtin_name={self.builtin_name})"


class TextureAccessNode(ASTNode):
    """Node representing texture memory access"""

    def __init__(self, texture_name, coordinates):
        self.texture_name = texture_name
        self.coordinates = coordinates

    def __repr__(self):
        return f"TextureAccessNode(texture_name={self.texture_name}, coordinates={self.coordinates})"


class SharedMemoryNode(VariableNode):
    """Node representing shared memory variable declaration"""

    def __init__(self, vtype, name, size=None):
        super().__init__(vtype, name, qualifiers=["__shared__"])
        self.size = size

    def __repr__(self):
        return (
            f"SharedMemoryNode(vtype={self.vtype}, name={self.name}, size={self.size})"
        )


class ConstantMemoryNode(VariableNode):
    """Node representing constant memory variable declaration"""

    def __init__(self, vtype, name, value=None):
        super().__init__(vtype, name, value, qualifiers=["__constant__"])

    def __repr__(self):
        return f"ConstantMemoryNode(vtype={self.vtype}, name={self.name}, value={self.value})"


class HipErrorHandlingNode(ASTNode):
    """Node representing HIP error handling (hipError_t, hipGetErrorString, etc.)"""

    def __init__(self, error_type, error_expr):
        self.error_type = error_type
        self.error_expr = error_expr

    def __repr__(self):
        return f"HipErrorHandlingNode(error_type={self.error_type}, error_expr={self.error_expr})"


class HipDevicePropertyNode(ASTNode):
    """Node representing HIP device property access"""

    def __init__(self, property_name, device_id=None):
        self.property_name = property_name
        self.device_id = device_id

    def __repr__(self):
        return f"HipDevicePropertyNode(property_name={self.property_name}, device_id={self.device_id})"
