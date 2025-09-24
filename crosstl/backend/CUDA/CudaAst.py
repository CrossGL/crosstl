"""CUDA AST Node definitions"""

from ..base_ast import ASTNode, BaseArrayAccessNode, BaseAssignmentNode, BaseAtomicOperationNode, BaseBinaryOpNode, BaseBreakNode, BaseBuiltinVariableNode, BaseCaseNode, BaseContinueNode, BaseForNode, BaseFunctionCallNode, BaseFunctionNode, BaseIfNode, BaseKernelLaunchNode, BaseKernelNode, BaseMemberAccessNode, BasePreprocessorNode, BaseReturnNode, BaseShaderNode, BaseStructNode, BaseSwitchNode, BaseSyncNode, BaseTernaryOpNode, BaseTextureAccessNode, BaseUnaryOpNode, BaseVariableNode, BaseVectorConstructorNode, BaseWhileNode, NodeType


class CudaShaderNode(BaseShaderNode):
    """Root node representing a complete CUDA program"""

    def __init__(
        self,
        includes=None,
        functions=None,
        structs=None,
        global_variables=None,
        kernels=None,
        **kwargs,
    ):
        super().__init__(includes, structs, functions, global_variables, **kwargs)
        self.kernels = kernels or []

        for kernel in self.kernels:
            self.add_child(kernel)

    def __repr__(self):
        return f"CudaShaderNode(includes={len(self.includes)}, functions={len(self.functions)}, structs={len(self.structs)}, global_variables={len(self.global_variables)}, kernels={len(self.kernels)})"


class CudaFunctionNode(BaseFunctionNode):
    """Node representing a CUDA function declaration"""

    def __init__(
        self,
        return_type,
        name,
        params,
        body,
        qualifiers=None,
        attributes=None,
        **kwargs,
    ):
        super().__init__(
            return_type, name, params, body, qualifiers, attributes, **kwargs
        )

    def __repr__(self):
        return f"CudaFunctionNode(return_type={self.return_type}, name={self.name}, params={len(self.params)}, qualifiers={self.qualifiers})"


class CudaKernelNode(BaseKernelNode):
    """Node representing a CUDA kernel function (marked with __global__)"""

    def __init__(self, return_type, name, params, body, attributes=None, **kwargs):
        super().__init__(return_type, name, params, body, attributes, **kwargs)

    def __repr__(self):
        return f"CudaKernelNode(name={self.name}, params={len(self.params)})"


class CudaKernelLaunchNode(BaseKernelLaunchNode):
    """Node representing a CUDA kernel launch: kernel<<<blocks, threads>>>(args)"""

    def __repr__(self):
        return f"CudaKernelLaunchNode(kernel_name={self.kernel_name}, blocks={self.blocks}, threads={self.threads}, args={len(self.args)})"


class CudaStructNode(BaseStructNode):
    """Node representing a CUDA struct declaration"""

    def __repr__(self):
        return f"CudaStructNode(name={self.name}, members={len(self.members)})"


class CudaVariableNode(BaseVariableNode):
    """Node representing a CUDA variable declaration"""

    def __repr__(self):
        return f"CudaVariableNode(vtype={self.vtype}, name={self.name}, qualifiers={self.qualifiers})"


# Use base classes directly for common operations
CudaAssignmentNode = BaseAssignmentNode
CudaBinaryOpNode = BaseBinaryOpNode
CudaUnaryOpNode = BaseUnaryOpNode
CudaFunctionCallNode = BaseFunctionCallNode


class CudaAtomicOperationNode(BaseAtomicOperationNode):
    """Node representing a CUDA atomic operation"""

    def __repr__(self):
        return f"CudaAtomicOperationNode(operation={self.operation}, args={len(self.args)})"


class CudaSyncNode(BaseSyncNode):
    """Node representing CUDA synchronization operations"""

    def __repr__(self):
        return f"CudaSyncNode(sync_type={self.sync_type}, args={len(self.args)})"


# Use base classes for common access patterns
CudaMemberAccessNode = BaseMemberAccessNode
CudaArrayAccessNode = BaseArrayAccessNode


# Use base classes for control flow
CudaIfNode = BaseIfNode
CudaForNode = BaseForNode
CudaWhileNode = BaseWhileNode


class CudaDoWhileNode(ASTNode):
    """Node representing a CUDA do-while loop"""

    def __init__(self, body, condition, **kwargs):
        super().__init__(NodeType.WHILE, **kwargs)  # Reuse WHILE type
        self.body = body if isinstance(body, list) else [body] if body else []
        self.condition = condition

        for stmt in self.body:
            if stmt:
                self.add_child(stmt)
        if condition:
            self.add_child(condition)

    def __repr__(self):
        return f"CudaDoWhileNode(body={len(self.body)}, condition={self.condition})"


# Use base classes for common control flow
CudaSwitchNode = BaseSwitchNode
CudaCaseNode = BaseCaseNode
CudaReturnNode = BaseReturnNode
CudaBreakNode = BaseBreakNode
CudaContinueNode = BaseContinueNode


# Use base classes for common constructs
CudaVectorConstructorNode = BaseVectorConstructorNode
CudaTernaryOpNode = BaseTernaryOpNode


class CudaCastNode(ASTNode):
    """Node representing a CUDA type cast"""

    def __init__(self, target_type, expression, **kwargs):
        super().__init__(NodeType.CAST, **kwargs)
        self.target_type = target_type
        self.expression = expression
        self.add_child(expression)

    def __repr__(self):
        return f"CudaCastNode(target_type={self.target_type}, expression={self.expression})"


# Use base classes for common constructs
CudaPreprocessorNode = BasePreprocessorNode


class CudaBuiltinNode(BaseBuiltinVariableNode):
    """Node representing CUDA built-in variables (threadIdx, blockIdx, etc.)"""

    def __repr__(self):
        if self.component:
            return f"CudaBuiltinNode(builtin_name={self.builtin_name}, component={self.component})"
        return f"CudaBuiltinNode(builtin_name={self.builtin_name})"


class CudaTextureAccessNode(BaseTextureAccessNode):
    """Node representing CUDA texture memory access"""

    def __repr__(self):
        return f"CudaTextureAccessNode(texture_name={self.texture_name}, coordinates={self.coordinates})"


class CudaSharedMemoryNode(CudaVariableNode):
    """Node representing CUDA shared memory variable declaration"""

    def __init__(self, vtype, name, size=None, **kwargs):
        super().__init__(vtype, name, qualifiers=["__shared__"], **kwargs)
        self.size = size  # For dynamic shared memory

    def __repr__(self):
        return f"CudaSharedMemoryNode(vtype={self.vtype}, name={self.name}, size={self.size})"


class CudaConstantMemoryNode(CudaVariableNode):
    """Node representing CUDA constant memory variable declaration"""

    def __init__(self, vtype, name, value=None, **kwargs):
        super().__init__(vtype, name, value, qualifiers=["__constant__"], **kwargs)

    def __repr__(self):
        return f"CudaConstantMemoryNode(vtype={self.vtype}, name={self.name}, value={self.value})"


# Backward compatibility aliases
ShaderNode = CudaShaderNode
FunctionNode = CudaFunctionNode
KernelNode = CudaKernelNode
KernelLaunchNode = CudaKernelLaunchNode
StructNode = CudaStructNode
VariableNode = CudaVariableNode
AssignmentNode = CudaAssignmentNode
BinaryOpNode = CudaBinaryOpNode
UnaryOpNode = CudaUnaryOpNode
FunctionCallNode = CudaFunctionCallNode
AtomicOperationNode = CudaAtomicOperationNode
SyncNode = CudaSyncNode
MemberAccessNode = CudaMemberAccessNode
ArrayAccessNode = CudaArrayAccessNode
IfNode = CudaIfNode
ForNode = CudaForNode
WhileNode = CudaWhileNode
DoWhileNode = CudaDoWhileNode
SwitchNode = CudaSwitchNode
CaseNode = CudaCaseNode
ReturnNode = CudaReturnNode
BreakNode = CudaBreakNode
ContinueNode = CudaContinueNode
VectorConstructorNode = CudaVectorConstructorNode
TernaryOpNode = CudaTernaryOpNode
CastNode = CudaCastNode
PreprocessorNode = CudaPreprocessorNode
TextureAccessNode = CudaTextureAccessNode
SharedMemoryNode = CudaSharedMemoryNode
ConstantMemoryNode = CudaConstantMemoryNode
