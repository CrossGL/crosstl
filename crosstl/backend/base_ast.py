"""
Base AST Node definitions for all backends.
This module provides the foundation for all language-specific AST implementations,
reducing redundancy and ensuring consistency across backends.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Dict, Union
from enum import Enum


class NodeType(Enum):
    """Enumeration of all AST node types."""
    
    # Base nodes
    SHADER = "shader"
    FUNCTION = "function"
    STRUCT = "struct"
    VARIABLE = "variable"
    PARAMETER = "parameter"
    
    # Expression nodes
    BINARY_OP = "binary_op"
    UNARY_OP = "unary_op"
    ASSIGNMENT = "assignment"
    FUNCTION_CALL = "function_call"
    MEMBER_ACCESS = "member_access"
    ARRAY_ACCESS = "array_access"
    LITERAL = "literal"
    IDENTIFIER = "identifier"
    TERNARY_OP = "ternary_op"
    CAST = "cast"
    CONSTRUCTOR = "constructor"
    
    # Statement nodes
    IF = "if"
    FOR = "for"
    WHILE = "while"
    DO_WHILE = "do_while"
    SWITCH = "switch"
    CASE = "case"
    RETURN = "return"
    BREAK = "break"
    CONTINUE = "continue"
    BLOCK = "block"
    
    # Type nodes
    PRIMITIVE_TYPE = "primitive_type"
    VECTOR_TYPE = "vector_type"
    MATRIX_TYPE = "matrix_type"
    ARRAY_TYPE = "array_type"
    POINTER_TYPE = "pointer_type"
    STRUCT_TYPE = "struct_type"
    
    # Language-specific nodes
    KERNEL = "kernel"
    KERNEL_LAUNCH = "kernel_launch"
    ATOMIC_OP = "atomic_op"
    SYNC = "sync"
    TEXTURE_SAMPLE = "texture_sample"
    BUILTIN_VARIABLE = "builtin_variable"
    PREPROCESSOR = "preprocessor"
    ATTRIBUTE = "attribute"
    IMPORT = "import"
    EXPORT = "export"


class ASTNode(ABC):
    """Base class for all AST nodes with standardized interface."""
    
    def __init__(self, node_type: NodeType, **kwargs):
        self.node_type = node_type
        self.source_location = kwargs.get('source_location')
        self.annotations = kwargs.get('annotations', {})
        self.parent = None
        self._children = []
    
    def add_child(self, child: 'ASTNode'):
        """Add a child node."""
        if child is not None:
            child.parent = self
            self._children.append(child)
    
    def get_children(self) -> List['ASTNode']:
        """Get all child nodes."""
        return self._children.copy()
    
    def accept(self, visitor):
        """Accept a visitor for traversal."""
        return visitor.visit(self)
    
    def add_annotation(self, key: str, value: Any):
        """Add language-specific annotation."""
        self.annotations[key] = value
    
    def get_annotation(self, key: str, default=None):
        """Get annotation value."""
        return self.annotations.get(key, default)
    
    @abstractmethod
    def __repr__(self) -> str:
        """String representation of the node."""
        pass


class BaseShaderNode(ASTNode):
    """Base class for shader program nodes."""
    
    def __init__(self, 
                 includes: Optional[List['PreprocessorNode']] = None,
                 structs: Optional[List['StructNode']] = None,
                 functions: Optional[List['FunctionNode']] = None,
                 global_variables: Optional[List['VariableNode']] = None,
                 **kwargs):
        super().__init__(NodeType.SHADER, **kwargs)
        self.includes = includes or []
        self.structs = structs or []
        self.functions = functions or []
        self.global_variables = global_variables or []
        
        # Add all as children
        for child_list in [self.includes, self.structs, self.functions, self.global_variables]:
            for child in child_list:
                self.add_child(child)
    
    def __repr__(self) -> str:
        return f"BaseShaderNode(includes={len(self.includes)}, structs={len(self.structs)}, functions={len(self.functions)}, global_variables={len(self.global_variables)})"


class BaseStructNode(ASTNode):
    """Base class for struct declarations."""
    
    def __init__(self, 
                 name: str, 
                 members: List['VariableNode'],
                 attributes: Optional[List['AttributeNode']] = None,
                 **kwargs):
        super().__init__(NodeType.STRUCT, **kwargs)
        self.name = name
        self.members = members
        self.attributes = attributes or []
        
        for member in self.members:
            self.add_child(member)
        for attr in self.attributes:
            self.add_child(attr)
    
    def __repr__(self) -> str:
        return f"BaseStructNode(name={self.name}, members={len(self.members)})"


class BaseFunctionNode(ASTNode):
    """Base class for function declarations."""
    
    def __init__(self, 
                 return_type: str,
                 name: str,
                 params: List['ParameterNode'],
                 body: Optional[List['StatementNode']] = None,
                 qualifiers: Optional[List[str]] = None,
                 attributes: Optional[List['AttributeNode']] = None,
                 **kwargs):
        super().__init__(NodeType.FUNCTION, **kwargs)
        self.return_type = return_type
        self.name = name
        self.params = params
        self.body = body or []
        self.qualifiers = qualifiers or []
        self.attributes = attributes or []
        
        for param in self.params:
            self.add_child(param)
        for stmt in self.body:
            if stmt:
                self.add_child(stmt)
        for attr in self.attributes:
            self.add_child(attr)
    
    def __repr__(self) -> str:
        return f"BaseFunctionNode(return_type={self.return_type}, name={self.name}, params={len(self.params)})"


class BaseVariableNode(ASTNode):
    """Base class for variable declarations."""
    
    def __init__(self, 
                 vtype: str,
                 name: str,
                 value: Optional['ExpressionNode'] = None,
                 qualifiers: Optional[List[str]] = None,
                 attributes: Optional[List['AttributeNode']] = None,
                 **kwargs):
        super().__init__(NodeType.VARIABLE, **kwargs)
        self.vtype = vtype
        self.name = name
        self.value = value
        self.qualifiers = qualifiers or []
        self.attributes = attributes or []
        
        if self.value:
            self.add_child(self.value)
        for attr in self.attributes:
            self.add_child(attr)
    
    def __repr__(self) -> str:
        return f"BaseVariableNode(vtype={self.vtype}, name={self.name})"


class BaseParameterNode(ASTNode):
    """Base class for function parameters."""
    
    def __init__(self, 
                 vtype: str,
                 name: str,
                 attributes: Optional[List['AttributeNode']] = None,
                 **kwargs):
        super().__init__(NodeType.PARAMETER, **kwargs)
        self.vtype = vtype
        self.name = name
        self.attributes = attributes or []
        
        for attr in self.attributes:
            self.add_child(attr)
    
    def __repr__(self) -> str:
        return f"BaseParameterNode(vtype={self.vtype}, name={self.name})"


class BaseBinaryOpNode(ASTNode):
    """Base class for binary operations."""
    
    def __init__(self, 
                 left: 'ExpressionNode',
                 op: str,
                 right: 'ExpressionNode',
                 **kwargs):
        super().__init__(NodeType.BINARY_OP, **kwargs)
        self.left = left
        self.op = op
        self.right = right
        
        self.add_child(left)
        self.add_child(right)
    
    def __repr__(self) -> str:
        return f"BaseBinaryOpNode(left={self.left}, op={self.op}, right={self.right})"


class BaseUnaryOpNode(ASTNode):
    """Base class for unary operations."""
    
    def __init__(self, 
                 op: str,
                 operand: 'ExpressionNode',
                 is_postfix: bool = False,
                 **kwargs):
        super().__init__(NodeType.UNARY_OP, **kwargs)
        self.op = op
        self.operand = operand
        self.is_postfix = is_postfix
        
        self.add_child(operand)
    
    def __repr__(self) -> str:
        return f"BaseUnaryOpNode(op={self.op}, operand={self.operand}, is_postfix={self.is_postfix})"


class BaseAssignmentNode(ASTNode):
    """Base class for assignment operations."""
    
    def __init__(self, 
                 left: 'ExpressionNode',
                 right: 'ExpressionNode',
                 operator: str = "=",
                 **kwargs):
        super().__init__(NodeType.ASSIGNMENT, **kwargs)
        self.left = left
        self.right = right
        self.operator = operator
        
        self.add_child(left)
        self.add_child(right)
    
    def __repr__(self) -> str:
        return f"BaseAssignmentNode(left={self.left}, operator={self.operator}, right={self.right})"


class BaseFunctionCallNode(ASTNode):
    """Base class for function calls."""
    
    def __init__(self, 
                 name: Union[str, 'ExpressionNode'],
                 args: List['ExpressionNode'],
                 **kwargs):
        super().__init__(NodeType.FUNCTION_CALL, **kwargs)
        self.name = name
        self.args = args
        
        if isinstance(name, ASTNode):
            self.add_child(name)
        for arg in self.args:
            self.add_child(arg)
    
    def __repr__(self) -> str:
        return f"BaseFunctionCallNode(name={self.name}, args={len(self.args)})"


class BaseMemberAccessNode(ASTNode):
    """Base class for member access operations."""
    
    def __init__(self, 
                 object_expr: 'ExpressionNode',
                 member: str,
                 is_pointer: bool = False,
                 **kwargs):
        super().__init__(NodeType.MEMBER_ACCESS, **kwargs)
        self.object = object_expr  # Standardize to 'object'
        self.member = member
        self.is_pointer = is_pointer
        
        self.add_child(object_expr)
    
    def __repr__(self) -> str:
        op = "->" if self.is_pointer else "."
        return f"BaseMemberAccessNode(object={self.object}, member={self.member}, operator={op})"


class BaseArrayAccessNode(ASTNode):
    """Base class for array access operations."""
    
    def __init__(self, 
                 array: 'ExpressionNode',
                 index: 'ExpressionNode',
                 **kwargs):
        super().__init__(NodeType.ARRAY_ACCESS, **kwargs)
        self.array = array
        self.index = index
        
        self.add_child(array)
        self.add_child(index)
    
    def __repr__(self) -> str:
        return f"BaseArrayAccessNode(array={self.array}, index={self.index})"


class BaseIfNode(ASTNode):
    """Base class for if statements."""
    
    def __init__(self, 
                 condition: 'ExpressionNode',
                 if_body: Union['StatementNode', List['StatementNode']],
                 else_body: Optional[Union['StatementNode', List['StatementNode']]] = None,
                 **kwargs):
        super().__init__(NodeType.IF, **kwargs)
        self.condition = condition
        self.if_body = if_body if isinstance(if_body, list) else [if_body]
        self.else_body = else_body if else_body is None or isinstance(else_body, list) else [else_body]
        
        self.add_child(condition)
        for stmt in self.if_body:
            if stmt:
                self.add_child(stmt)
        if self.else_body:
            for stmt in self.else_body:
                if stmt:
                    self.add_child(stmt)
    
    def __repr__(self) -> str:
        return f"BaseIfNode(condition={self.condition}, if_body={len(self.if_body)}, else_body={len(self.else_body) if self.else_body else 0})"


class BaseForNode(ASTNode):
    """Base class for for loops."""
    
    def __init__(self, 
                 init: Optional['StatementNode'],
                 condition: Optional['ExpressionNode'],
                 update: Optional['ExpressionNode'],
                 body: Union['StatementNode', List['StatementNode']],
                 **kwargs):
        super().__init__(NodeType.FOR, **kwargs)
        self.init = init
        self.condition = condition
        self.update = update
        self.body = body if isinstance(body, list) else [body] if body else []
        
        if self.init:
            self.add_child(self.init)
        if self.condition:
            self.add_child(self.condition)
        if self.update:
            self.add_child(self.update)
        for stmt in self.body:
            if stmt:
                self.add_child(stmt)
    
    def __repr__(self) -> str:
        return f"BaseForNode(init={self.init}, condition={self.condition}, update={self.update}, body={len(self.body)})"


class BaseWhileNode(ASTNode):
    """Base class for while loops."""
    
    def __init__(self, 
                 condition: 'ExpressionNode',
                 body: Union['StatementNode', List['StatementNode']],
                 **kwargs):
        super().__init__(NodeType.WHILE, **kwargs)
        self.condition = condition
        self.body = body if isinstance(body, list) else [body] if body else []
        
        self.add_child(condition)
        for stmt in self.body:
            if stmt:
                self.add_child(stmt)
    
    def __repr__(self) -> str:
        return f"BaseWhileNode(condition={self.condition}, body={len(self.body)})"


class BaseReturnNode(ASTNode):
    """Base class for return statements."""
    
    def __init__(self, 
                 value: Optional['ExpressionNode'] = None,
                 **kwargs):
        super().__init__(NodeType.RETURN, **kwargs)
        self.value = value
        
        if self.value:
            self.add_child(self.value)
    
    def __repr__(self) -> str:
        return f"BaseReturnNode(value={self.value})"


class BaseBreakNode(ASTNode):
    """Base class for break statements."""
    
    def __init__(self, 
                 label: Optional[str] = None,
                 **kwargs):
        super().__init__(NodeType.BREAK, **kwargs)
        self.label = label
    
    def __repr__(self) -> str:
        return f"BaseBreakNode(label={self.label})"


class BaseContinueNode(ASTNode):
    """Base class for continue statements."""
    
    def __init__(self, 
                 label: Optional[str] = None,
                 **kwargs):
        super().__init__(NodeType.CONTINUE, **kwargs)
        self.label = label
    
    def __repr__(self) -> str:
        return f"BaseContinueNode(label={self.label})"


class BaseTernaryOpNode(ASTNode):
    """Base class for ternary conditional operations."""
    
    def __init__(self, 
                 condition: 'ExpressionNode',
                 true_expr: 'ExpressionNode',
                 false_expr: 'ExpressionNode',
                 **kwargs):
        super().__init__(NodeType.TERNARY_OP, **kwargs)
        self.condition = condition
        self.true_expr = true_expr
        self.false_expr = false_expr
        
        self.add_child(condition)
        self.add_child(true_expr)
        self.add_child(false_expr)
    
    def __repr__(self) -> str:
        return f"BaseTernaryOpNode(condition={self.condition}, true_expr={self.true_expr}, false_expr={self.false_expr})"


class BaseSwitchNode(ASTNode):
    """Base class for switch statements."""
    
    def __init__(self, 
                 expression: 'ExpressionNode',
                 cases: List['CaseNode'],
                 default_case: Optional['CaseNode'] = None,
                 **kwargs):
        super().__init__(NodeType.SWITCH, **kwargs)
        self.expression = expression
        self.cases = cases
        self.default_case = default_case
        
        self.add_child(expression)
        for case in self.cases:
            self.add_child(case)
        if self.default_case:
            self.add_child(self.default_case)
    
    def __repr__(self) -> str:
        return f"BaseSwitchNode(expression={self.expression}, cases={len(self.cases)})"


class BaseCaseNode(ASTNode):
    """Base class for switch cases."""
    
    def __init__(self, 
                 value: Optional['ExpressionNode'],  # None for default case
                 body: List['StatementNode'],
                 **kwargs):
        super().__init__(NodeType.CASE, **kwargs)
        self.value = value
        self.body = body
        
        if self.value:
            self.add_child(self.value)
        for stmt in self.body:
            if stmt:
                self.add_child(stmt)
    
    def __repr__(self) -> str:
        return f"BaseCaseNode(value={self.value}, body={len(self.body)})"


class BasePreprocessorNode(ASTNode):
    """Base class for preprocessor directives."""
    
    def __init__(self, 
                 directive: str,
                 content: str,
                 **kwargs):
        super().__init__(NodeType.PREPROCESSOR, **kwargs)
        self.directive = directive
        self.content = content
    
    def __repr__(self) -> str:
        return f"BasePreprocessorNode(directive={self.directive}, content={self.content})"


class BaseAttributeNode(ASTNode):
    """Base class for attributes/annotations."""
    
    def __init__(self, 
                 name: str,
                 args: Optional[List[str]] = None,
                 **kwargs):
        super().__init__(NodeType.ATTRIBUTE, **kwargs)
        self.name = name
        self.args = args or []
    
    def __repr__(self) -> str:
        return f"BaseAttributeNode(name={self.name}, args={self.args})"


class BaseVectorConstructorNode(ASTNode):
    """Base class for vector constructor expressions."""
    
    def __init__(self, 
                 vector_type: str,
                 args: List['ExpressionNode'],
                 **kwargs):
        super().__init__(NodeType.CONSTRUCTOR, **kwargs)
        self.vector_type = vector_type
        self.type_name = vector_type  # Alias for compatibility
        self.args = args
        self.arguments = args  # Alias for compatibility
        
        for arg in self.args:
            self.add_child(arg)
    
    def __repr__(self) -> str:
        return f"BaseVectorConstructorNode(vector_type={self.vector_type}, args={len(self.args)})"


# Language-specific extensions


class BaseKernelNode(BaseFunctionNode):
    """Base class for compute kernels (CUDA, HIP, etc.)."""
    
    def __init__(self, 
                 return_type: str,
                 name: str,
                 params: List['ParameterNode'],
                 body: Optional[List['StatementNode']] = None,
                 attributes: Optional[List['AttributeNode']] = None,
                 **kwargs):
        super().__init__(return_type, name, params, body, ["__global__"], attributes, **kwargs)
        self.node_type = NodeType.KERNEL
    
    def __repr__(self) -> str:
        return f"BaseKernelNode(name={self.name}, params={len(self.params)})"


class BaseKernelLaunchNode(ASTNode):
    """Base class for kernel launch operations."""
    
    def __init__(self, 
                 kernel_name: str,
                 blocks: 'ExpressionNode',
                 threads: 'ExpressionNode',
                 shared_mem: Optional['ExpressionNode'] = None,
                 stream: Optional['ExpressionNode'] = None,
                 args: Optional[List['ExpressionNode']] = None,
                 **kwargs):
        super().__init__(NodeType.KERNEL_LAUNCH, **kwargs)
        self.kernel_name = kernel_name
        self.blocks = blocks
        self.threads = threads
        self.shared_mem = shared_mem
        self.stream = stream
        self.args = args or []
        
        self.add_child(blocks)
        self.add_child(threads)
        if self.shared_mem:
            self.add_child(self.shared_mem)
        if self.stream:
            self.add_child(self.stream)
        for arg in self.args:
            self.add_child(arg)
    
    def __repr__(self) -> str:
        return f"BaseKernelLaunchNode(kernel_name={self.kernel_name}, blocks={self.blocks}, threads={self.threads})"


class BaseAtomicOperationNode(BaseFunctionCallNode):
    """Base class for atomic operations."""
    
    def __init__(self, 
                 operation: str,
                 args: List['ExpressionNode'],
                 **kwargs):
        super().__init__(operation, args, **kwargs)
        self.node_type = NodeType.ATOMIC_OP
        self.operation = operation
    
    def __repr__(self) -> str:
        return f"BaseAtomicOperationNode(operation={self.operation}, args={len(self.args)})"


class BaseSyncNode(ASTNode):
    """Base class for synchronization operations."""
    
    def __init__(self, 
                 sync_type: str,
                 args: Optional[List['ExpressionNode']] = None,
                 **kwargs):
        super().__init__(NodeType.SYNC, **kwargs)
        self.sync_type = sync_type
        self.args = args or []
        
        for arg in self.args:
            self.add_child(arg)
    
    def __repr__(self) -> str:
        return f"BaseSyncNode(sync_type={self.sync_type}, args={len(self.args)})"


class BaseBuiltinVariableNode(ASTNode):
    """Base class for built-in variables (threadIdx, gl_Position, etc.)."""
    
    def __init__(self, 
                 builtin_name: str,
                 component: Optional[str] = None,
                 **kwargs):
        super().__init__(NodeType.BUILTIN_VARIABLE, **kwargs)
        self.builtin_name = builtin_name
        self.component = component
    
    def __repr__(self) -> str:
        if self.component:
            return f"BaseBuiltinVariableNode(builtin_name={self.builtin_name}, component={self.component})"
        return f"BaseBuiltinVariableNode(builtin_name={self.builtin_name})"


class BaseTextureAccessNode(ASTNode):
    """Base class for texture access operations."""
    
    def __init__(self, 
                 texture_name: Union[str, 'ExpressionNode'],
                 coordinates: 'ExpressionNode',
                 sampler: Optional['ExpressionNode'] = None,
                 lod: Optional['ExpressionNode'] = None,
                 **kwargs):
        super().__init__(NodeType.TEXTURE_SAMPLE, **kwargs)
        self.texture_name = texture_name
        self.texture = texture_name  # Alias for compatibility
        self.coordinates = coordinates
        self.sampler = sampler
        self.lod = lod
        
        if isinstance(texture_name, ASTNode):
            self.add_child(texture_name)
        self.add_child(coordinates)
        if self.sampler:
            self.add_child(self.sampler)
        if self.lod:
            self.add_child(self.lod)
    
    def __repr__(self) -> str:
        return f"BaseTextureAccessNode(texture={self.texture_name}, coordinates={self.coordinates})"


# Type alias for backward compatibility
StatementNode = ASTNode
ExpressionNode = ASTNode
ParameterNode = BaseParameterNode
VariableNode = BaseVariableNode
StructNode = BaseStructNode
FunctionNode = BaseFunctionNode
ShaderNode = BaseShaderNode
BinaryOpNode = BaseBinaryOpNode
UnaryOpNode = BaseUnaryOpNode
AssignmentNode = BaseAssignmentNode
FunctionCallNode = BaseFunctionCallNode
MemberAccessNode = BaseMemberAccessNode
ArrayAccessNode = BaseArrayAccessNode
IfNode = BaseIfNode
ForNode = BaseForNode
WhileNode = BaseWhileNode
ReturnNode = BaseReturnNode
BreakNode = BaseBreakNode
ContinueNode = BaseContinueNode
TernaryOpNode = BaseTernaryOpNode
SwitchNode = BaseSwitchNode
CaseNode = BaseCaseNode
PreprocessorNode = BasePreprocessorNode
AttributeNode = BaseAttributeNode
VectorConstructorNode = BaseVectorConstructorNode
KernelNode = BaseKernelNode
KernelLaunchNode = BaseKernelLaunchNode
AtomicOperationNode = BaseAtomicOperationNode
SyncNode = BaseSyncNode
BuiltinVariableNode = BaseBuiltinVariableNode
TextureAccessNode = BaseTextureAccessNode
