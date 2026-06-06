"""CUDA AST Node definitions"""

from ..common_ast import (
    ArrayAccessNode,
    AssignmentNode,
    ASTNode,
    BinaryOpNode,
    BreakNode,
    CaseNode,
    CastNode,
    ContinueNode,
    DeleteNode,
    DesignatedInitializerNode,
    DoWhileNode,
    EnumNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IfNode,
    InitializerListNode,
    MemberAccessNode,
    NewNode,
    PreprocessorNode,
    RangeForNode,
    ReturnNode,
    ShaderNode,
    StructNode,
    SwitchNode,
    SyncNode,
    TernaryOpNode,
    TypeAliasNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)

_COMMON_NODES = (
    ASTNode,
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    BreakNode,
    CaseNode,
    CastNode,
    ContinueNode,
    DesignatedInitializerNode,
    DeleteNode,
    DoWhileNode,
    EnumNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IfNode,
    InitializerListNode,
    MemberAccessNode,
    NewNode,
    PreprocessorNode,
    RangeForNode,
    ReturnNode,
    ShaderNode,
    StructNode,
    SwitchNode,
    SyncNode,
    TernaryOpNode,
    TypeAliasNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)

# CUDA-specific nodes


def _ensure_qualifier(qualifiers, required):
    qualifiers = list(qualifiers or [])
    if required not in qualifiers:
        qualifiers.append(required)
    return qualifiers


class KernelNode(FunctionNode):
    """Node representing a CUDA kernel function (marked with __global__)"""

    def __init__(
        self,
        return_type,
        name,
        params,
        body,
        attributes=None,
        qualifiers=None,
        linkage=None,
    ):
        super().__init__(
            return_type,
            name,
            params,
            body,
            qualifiers or ["__global__"],
            attributes,
        )
        self.linkage = linkage

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
    """Node representing a CUDA atomic operation"""

    def __init__(self, operation, args):
        super().__init__(operation, args)
        self.operation = operation

    def __repr__(self):
        return f"AtomicOperationNode(operation={self.operation}, args={self.args})"


class CudaBuiltinNode(ASTNode):
    """Node representing CUDA built-in variables (threadIdx, blockIdx, etc.)"""

    def __init__(self, builtin_name, component=None):
        self.builtin_name = builtin_name
        self.component = component

    def __repr__(self):
        if self.component:
            return f"CudaBuiltinNode(builtin_name={self.builtin_name}, component={self.component})"
        return f"CudaBuiltinNode(builtin_name={self.builtin_name})"


class CudaAsmOperandNode(ASTNode):
    """Node representing one inline PTX asm operand."""

    def __init__(self, constraint, expression=None, symbolic_name=None):
        self.constraint = constraint
        self.expression = expression
        self.symbolic_name = symbolic_name

    def __repr__(self):
        return (
            "CudaAsmOperandNode("
            f"constraint={self.constraint}, expression={self.expression}, "
            f"symbolic_name={self.symbolic_name})"
        )


class CudaAsmNode(ASTNode):
    """Node representing a CUDA inline PTX asm statement."""

    def __init__(
        self,
        template,
        outputs=None,
        inputs=None,
        clobbers=None,
        is_volatile=False,
    ):
        self.template = template
        self.outputs = outputs or []
        self.inputs = inputs or []
        self.clobbers = clobbers or []
        self.is_volatile = is_volatile

    def __repr__(self):
        return (
            "CudaAsmNode("
            f"template={self.template}, outputs={self.outputs}, "
            f"inputs={self.inputs}, clobbers={self.clobbers}, "
            f"is_volatile={self.is_volatile})"
        )


class TextureAccessNode(ASTNode):
    """Node representing texture memory access"""

    def __init__(self, texture_name, coordinates):
        self.texture_name = texture_name
        self.coordinates = coordinates

    def __repr__(self):
        return f"TextureAccessNode(texture_name={self.texture_name}, coordinates={self.coordinates})"


class SharedMemoryNode(VariableNode):
    """Node representing shared memory variable declaration"""

    def __init__(
        self,
        vtype,
        name,
        size=None,
        is_extern=False,
        is_dynamic=False,
        qualifiers=None,
    ):
        super().__init__(
            vtype,
            name,
            qualifiers=_ensure_qualifier(qualifiers, "__shared__"),
        )
        self.size = size
        self.is_extern_shared_memory = is_extern
        self.is_dynamic_shared_memory = is_dynamic

    def __repr__(self):
        return (
            f"SharedMemoryNode(vtype={self.vtype}, name={self.name}, size={self.size})"
        )


class ConstantMemoryNode(VariableNode):
    """Node representing constant memory variable declaration"""

    def __init__(self, vtype, name, value=None, qualifiers=None):
        super().__init__(
            vtype,
            name,
            value,
            qualifiers=_ensure_qualifier(qualifiers, "__constant__"),
        )

    def __repr__(self):
        return f"ConstantMemoryNode(vtype={self.vtype}, name={self.name}, value={self.value})"
