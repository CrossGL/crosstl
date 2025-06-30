"""HIP AST Node definitions"""


class ASTNode:
    """Base class for all AST nodes"""


class ShaderNode(ASTNode):
    """Root node representing a complete HIP program"""

    def __init__(
        self,
        includes=None,
        functions=None,
        structs=None,
        global_variables=None,
        kernels=None,
    ):
        self.includes = includes or []
        self.functions = functions or []
        self.structs = structs or []
        self.global_variables = global_variables or []
        self.kernels = kernels or []

    def __repr__(self):
        return f"ShaderNode(includes={self.includes}, functions={self.functions}, structs={self.structs}, global_variables={self.global_variables}, kernels={self.kernels})"


class FunctionNode(ASTNode):
    """Node representing a function declaration"""

    def __init__(
        self, return_type, name, params, body, qualifiers=None, attributes=None
    ):
        self.return_type = return_type
        self.name = name
        self.params = params
        self.body = body
        self.qualifiers = qualifiers or []  # __global__, __device__, __host__
        self.attributes = attributes or []

    def __repr__(self):
        return f"FunctionNode(return_type={self.return_type}, name={self.name}, params={self.params}, body={self.body}, qualifiers={self.qualifiers})"


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
        self.shared_mem = shared_mem  # Optional shared memory size
        self.stream = stream  # Optional stream
        self.args = args or []

    def __repr__(self):
        return f"KernelLaunchNode(kernel_name={self.kernel_name}, blocks={self.blocks}, threads={self.threads}, args={self.args})"


class StructNode(ASTNode):
    """Node representing a struct declaration"""

    def __init__(self, name, members, attributes=None):
        self.name = name
        self.members = members
        self.attributes = attributes or []

    def __repr__(self):
        return f"StructNode(name={self.name}, members={self.members})"


class VariableNode(ASTNode):
    """Node representing a variable declaration"""

    def __init__(self, vtype, name, value=None, qualifiers=None):
        self.vtype = vtype
        self.name = name
        self.value = value
        self.qualifiers = qualifiers or []  # __shared__, __constant__, etc.

    def __repr__(self):
        return f"VariableNode(vtype={self.vtype}, name={self.name}, value={self.value}, qualifiers={self.qualifiers})"


class AssignmentNode(ASTNode):
    """Node representing an assignment operation"""

    def __init__(self, left, right, operator="="):
        self.left = left
        self.right = right
        self.operator = operator

    def __repr__(self):
        return f"AssignmentNode(left={self.left}, operator={self.operator}, right={self.right})"


class BinaryOpNode(ASTNode):
    """Node representing a binary operation"""

    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

    def __repr__(self):
        return f"BinaryOpNode(left={self.left}, op={self.op}, right={self.right})"


class UnaryOpNode(ASTNode):
    """Node representing a unary operation"""

    def __init__(self, op, operand):
        self.op = op
        self.operand = operand

    def __repr__(self):
        return f"UnaryOpNode(op={self.op}, operand={self.operand})"


class FunctionCallNode(ASTNode):
    """Node representing a function call"""

    def __init__(self, name, args):
        self.name = name
        self.args = args

    def __repr__(self):
        return f"FunctionCallNode(name={self.name}, args={self.args})"


class AtomicOperationNode(FunctionCallNode):
    """Node representing a HIP atomic operation"""

    def __init__(self, operation, args):
        super().__init__(operation, args)
        self.operation = operation  # atomicAdd, hipAtomicAdd, etc.

    def __repr__(self):
        return f"AtomicOperationNode(operation={self.operation}, args={self.args})"


class SyncNode(ASTNode):
    """Node representing synchronization operations"""

    def __init__(self, sync_type, args=None):
        self.sync_type = sync_type  # __syncthreads, hipDeviceSynchronize
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
    """Node representing HIP vector constructor (make_float4, etc.)"""

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


class HipBuiltinNode(ASTNode):
    """Node representing HIP built-in variables (threadIdx, hipThreadIdx_x, etc.)"""

    def __init__(self, builtin_name, component=None):
        self.builtin_name = builtin_name  # threadIdx, hipThreadIdx_x, etc.
        self.component = component  # x, y, z component (for CUDA-style builtins)

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


class HipErrorHandlingNode(ASTNode):
    """Node representing HIP error handling"""

    def __init__(self, error_var, hip_call):
        self.error_var = error_var
        self.hip_call = hip_call

    def __repr__(self):
        return f"HipErrorHandlingNode(error_var={self.error_var}, hip_call={self.hip_call})"


class HipDevicePropertyNode(ASTNode):
    """Node representing HIP device property access"""

    def __init__(self, property_name, device_id=None):
        self.property_name = property_name
        self.device_id = device_id

    def __repr__(self):
        return f"HipDevicePropertyNode(property_name={self.property_name}, device_id={self.device_id})"
