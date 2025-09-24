"""CUDA AST Node definitions"""

from ..base_ast import *


class CudaShaderNode(BaseShaderNode):
    """Root node representing a complete CUDA program"""

    def __init__(
        self,
        includes=None,
        functions=None,
        structs=None,
        global_variables=None,
        kernels=None,
        **kwargs
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
        self, return_type, name, params, body, qualifiers=None, attributes=None, **kwargs
    ):
        super().__init__(return_type, name, params, body, qualifiers, attributes, **kwargs)

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


class AtomicOperationNode(FunctionCallNode):
    """Node representing a CUDA atomic operation"""

    def __init__(self, operation, args):
        super().__init__(operation, args)
        self.operation = operation  # atomicAdd, atomicSub, etc.

    def __repr__(self):
        return f"AtomicOperationNode(operation={self.operation}, args={self.args})"


class SyncNode(ASTNode):
    """Node representing synchronization operations"""

    def __init__(self, sync_type, args=None):
        self.sync_type = sync_type  # __syncthreads, __syncwarp
        self.args = args or []

    def __repr__(self):
        return f"SyncNode(sync_type={self.sync_type}, args={self.args})"


class MemberAccessNode(ASTNode):
    """Node representing member access (dot or arrow operator)"""

    def __init__(self, object, member, is_pointer=False):
        self.object = object
        self.member = member
        self.is_pointer = is_pointer  # True for ->, False for .

    def __repr__(self):
        op = "->" if self.is_pointer else "."
        return f"MemberAccessNode(object={self.object}, member={self.member}, operator={op})"


class ArrayAccessNode(ASTNode):
    """Node representing array access"""

    def __init__(self, array, index):
        self.array = array
        self.index = index

    def __repr__(self):
        return f"ArrayAccessNode(array={self.array}, index={self.index})"


class IfNode(ASTNode):
    """Node representing an if statement"""

    def __init__(self, condition, if_body, else_body=None):
        self.condition = condition
        self.if_body = if_body
        self.else_body = else_body

    def __repr__(self):
        return f"IfNode(condition={self.condition}, if_body={self.if_body}, else_body={self.else_body})"


class ForNode(ASTNode):
    """Node representing a for loop"""

    def __init__(self, init, condition, update, body):
        self.init = init
        self.condition = condition
        self.update = update
        self.body = body

    def __repr__(self):
        return f"ForNode(init={self.init}, condition={self.condition}, update={self.update}, body={self.body})"


class WhileNode(ASTNode):
    """Node representing a while loop"""

    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

    def __repr__(self):
        return f"WhileNode(condition={self.condition}, body={self.body})"


class DoWhileNode(ASTNode):
    """Node representing a do-while loop"""

    def __init__(self, body, condition):
        self.body = body
        self.condition = condition

    def __repr__(self):
        return f"DoWhileNode(body={self.body}, condition={self.condition})"


class SwitchNode(ASTNode):
    """Node representing a switch statement"""

    def __init__(self, expression, cases, default_case=None):
        self.expression = expression
        self.cases = cases
        self.default_case = default_case

    def __repr__(self):
        return f"SwitchNode(expression={self.expression}, cases={self.cases}, default_case={self.default_case})"


class CaseNode(ASTNode):
    """Node representing a case in a switch statement"""

    def __init__(self, value, body):
        self.value = value
        self.body = body

    def __repr__(self):
        return f"CaseNode(value={self.value}, body={self.body})"


class ReturnNode(ASTNode):
    """Node representing a return statement"""

    def __init__(self, value=None):
        self.value = value

    def __repr__(self):
        return f"ReturnNode(value={self.value})"


class BreakNode(ASTNode):
    """Node representing a break statement"""

    def __repr__(self):
        return "BreakNode()"


class ContinueNode(ASTNode):
    """Node representing a continue statement"""

    def __repr__(self):
        return "ContinueNode()"


class VectorConstructorNode(ASTNode):
    """Node representing CUDA vector constructor (make_float4, etc.)"""

    def __init__(self, vector_type, args):
        self.vector_type = vector_type
        self.args = args

    def __repr__(self):
        return (
            f"VectorConstructorNode(vector_type={self.vector_type}, args={self.args})"
        )


class TernaryOpNode(ASTNode):
    """Node representing a ternary conditional operator"""

    def __init__(self, condition, true_expr, false_expr):
        self.condition = condition
        self.true_expr = true_expr
        self.false_expr = false_expr

    def __repr__(self):
        return f"TernaryOpNode(condition={self.condition}, true_expr={self.true_expr}, false_expr={self.false_expr})"


class CastNode(ASTNode):
    """Node representing a type cast"""

    def __init__(self, target_type, expression):
        self.target_type = target_type
        self.expression = expression

    def __repr__(self):
        return f"CastNode(target_type={self.target_type}, expression={self.expression})"


class PreprocessorNode(ASTNode):
    """Node representing preprocessor directives"""

    def __init__(self, directive, content):
        self.directive = directive  # include, define, etc.
        self.content = content

    def __repr__(self):
        return f"PreprocessorNode(directive={self.directive}, content={self.content})"


class CudaBuiltinNode(ASTNode):
    """Node representing CUDA built-in variables (threadIdx, blockIdx, etc.)"""

    def __init__(self, builtin_name, component=None):
        self.builtin_name = builtin_name  # threadIdx, blockIdx, etc.
        self.component = component  # x, y, z component

    def __repr__(self):
        if self.component:
            return f"CudaBuiltinNode(builtin_name={self.builtin_name}, component={self.component})"
        return f"CudaBuiltinNode(builtin_name={self.builtin_name})"


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
        self.size = size  # For dynamic shared memory

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
