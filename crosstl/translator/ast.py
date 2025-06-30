from typing import List, Optional, Any, Union, Dict
from enum import Enum


class ASTNode:
    """Base class for all AST nodes with common functionality."""

    def __init__(self, source_location=None, annotations=None):
        self.source_location = source_location  # For error reporting
        self.annotations = annotations or {}  # For backend-specific metadata
        self.parent = None  # Parent node reference

    def accept(self, visitor):
        """Visitor pattern support for AST traversal."""
        method_name = f"visit_{self.__class__.__name__}"
        method = getattr(visitor, method_name, visitor.generic_visit)
        return method(self)

    def add_annotation(self, key: str, value: Any):
        """Add backend-specific annotation."""
        self.annotations[key] = value

    def get_annotation(self, key: str, default=None):
        """Get backend-specific annotation."""
        return self.annotations.get(key, default)


# ============================================================================
# TYPE SYSTEM
# ============================================================================


class TypeNode(ASTNode):
    """Base class for all type representations."""


class PrimitiveType(TypeNode):
    """Primitive types (int, float, bool, etc.)."""

    def __init__(self, name: str, size_bits: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.name = name  # int, float, double, bool, void
        self.size_bits = size_bits  # 8, 16, 32, 64 for sized types

    def __repr__(self):
        return f"PrimitiveType(name={self.name}, size_bits={self.size_bits})"


class VectorType(TypeNode):
    """Vector types (vec2, vec3, vec4, float3, etc.)."""

    def __init__(self, element_type: TypeNode, size: int, **kwargs):
        super().__init__(**kwargs)
        self.element_type = element_type
        self.size = size  # 2, 3, 4

    def __repr__(self):
        return f"VectorType(element_type={self.element_type}, size={self.size})"


class MatrixType(TypeNode):
    """Matrix types (mat4, float4x4, etc.)."""

    def __init__(self, element_type: TypeNode, rows: int, cols: int, **kwargs):
        super().__init__(**kwargs)
        self.element_type = element_type
        self.rows = rows
        self.cols = cols

    def __repr__(self):
        return f"MatrixType(element_type={self.element_type}, rows={self.rows}, cols={self.cols})"


class ArrayType(TypeNode):
    """Array types with static or dynamic sizing."""

    def __init__(
        self,
        element_type: TypeNode,
        size: Optional[Union[int, ASTNode]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.element_type = element_type
        self.size = size  # None for dynamic arrays, int or expression for static

    def __repr__(self):
        return f"ArrayType(element_type={self.element_type}, size={self.size})"


class PointerType(TypeNode):
    """Pointer types for languages that support them."""

    def __init__(self, pointee_type: TypeNode, is_mutable: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.pointee_type = pointee_type
        self.is_mutable = is_mutable

    def __repr__(self):
        return f"PointerType(pointee_type={self.pointee_type}, is_mutable={self.is_mutable})"


class ReferenceType(TypeNode):
    """Reference types for languages like Rust."""

    def __init__(self, referenced_type: TypeNode, is_mutable: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.referenced_type = referenced_type
        self.is_mutable = is_mutable

    def __repr__(self):
        return f"ReferenceType(referenced_type={self.referenced_type}, is_mutable={self.is_mutable})"


class FunctionType(TypeNode):
    """Function pointer/reference types."""

    def __init__(self, return_type: TypeNode, param_types: List[TypeNode], **kwargs):
        super().__init__(**kwargs)
        self.return_type = return_type
        self.param_types = param_types

    def __repr__(self):
        return f"FunctionType(return_type={self.return_type}, param_types={self.param_types})"


class GenericType(TypeNode):
    """Generic/template type parameters."""

    def __init__(self, name: str, constraints: List[TypeNode] = None, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.constraints = constraints or []  # Type constraints/bounds

    def __repr__(self):
        return f"GenericType(name={self.name}, constraints={self.constraints})"


class NamedType(TypeNode):
    """User-defined types (structs, enums, etc.)."""

    def __init__(self, name: str, generic_args: List[TypeNode] = None, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.generic_args = generic_args or []

    def __repr__(self):
        return f"NamedType(name={self.name}, generic_args={self.generic_args})"


# ============================================================================
# SHADER/PROGRAM STRUCTURE
# ============================================================================


class ShaderStage(Enum):
    """Shader pipeline stages."""

    VERTEX = "vertex"
    FRAGMENT = "fragment"
    GEOMETRY = "geometry"
    TESSELLATION_CONTROL = "tessellation_control"
    TESSELLATION_EVALUATION = "tessellation_evaluation"
    COMPUTE = "compute"
    RAY_GENERATION = "ray_generation"
    RAY_INTERSECTION = "ray_intersection"
    RAY_CLOSEST_HIT = "ray_closest_hit"
    RAY_MISS = "ray_miss"
    RAY_ANY_HIT = "ray_any_hit"
    RAY_CALLABLE = "ray_callable"


class ExecutionModel(Enum):
    """Different execution models supported."""

    GRAPHICS_PIPELINE = "graphics_pipeline"
    COMPUTE_KERNEL = "compute_kernel"
    RAY_TRACING = "ray_tracing"
    GENERAL_PURPOSE = "general_purpose"


class ShaderNode(ASTNode):
    """Root node representing a complete shader program."""

    def __init__(
        self,
        name: str,
        execution_model: ExecutionModel,
        stages: Dict[ShaderStage, "StageNode"] = None,
        structs: List["StructNode"] = None,
        functions: List["FunctionNode"] = None,
        global_variables: List["VariableNode"] = None,
        constants: List["ConstantNode"] = None,
        imports: List["ImportNode"] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = name
        self.execution_model = execution_model
        self.stages = stages or {}
        self.structs = structs or []
        self.functions = functions or []
        self.global_variables = global_variables or []
        self.constants = constants or []
        self.imports = imports or []

    def __repr__(self):
        return f"ShaderNode(name={self.name}, execution_model={self.execution_model})"


class StageNode(ASTNode):
    """Individual shader stage (vertex, fragment, compute, etc.)."""

    def __init__(
        self,
        stage: ShaderStage,
        entry_point: "FunctionNode",
        local_variables: List["VariableNode"] = None,
        local_functions: List["FunctionNode"] = None,
        execution_config: Dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.stage = stage
        self.entry_point = entry_point
        self.local_variables = local_variables or []
        self.local_functions = local_functions or []
        self.execution_config = (
            execution_config or {}
        )  # For compute workgroup size, etc.

    def __repr__(self):
        return f"StageNode(stage={self.stage}, entry_point={self.entry_point.name})"


class ImportNode(ASTNode):
    """Import/include statements."""

    def __init__(
        self, path: str, alias: Optional[str] = None, items: List[str] = None, **kwargs
    ):
        super().__init__(**kwargs)
        self.path = path
        self.alias = alias
        self.items = items  # For selective imports

    def __repr__(self):
        return f"ImportNode(path={self.path}, alias={self.alias}, items={self.items})"


# ============================================================================
# DECLARATIONS
# ============================================================================


class StructNode(ASTNode):
    """Struct/class declarations."""

    def __init__(
        self,
        name: str,
        members: List["StructMemberNode"],
        generic_params: List["GenericParameterNode"] = None,
        attributes: List["AttributeNode"] = None,
        inheritance: List[NamedType] = None,
        visibility: str = "public",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = name
        self.members = members
        self.generic_params = generic_params or []
        self.attributes = attributes or []
        self.inheritance = inheritance or []  # Base classes/traits
        self.visibility = visibility

    def __repr__(self):
        return f"StructNode(name={self.name}, members={len(self.members)})"


class StructMemberNode(ASTNode):
    """Individual struct member."""

    def __init__(
        self,
        name: str,
        member_type: TypeNode,
        default_value: Optional["ExpressionNode"] = None,
        attributes: List["AttributeNode"] = None,
        visibility: str = "public",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = name
        self.member_type = member_type
        self.default_value = default_value
        self.attributes = attributes or []
        self.visibility = visibility

    def __repr__(self):
        return f"StructMemberNode(name={self.name}, member_type={self.member_type})"


class EnumNode(ASTNode):
    """Enumeration declarations."""

    def __init__(
        self,
        name: str,
        variants: List["EnumVariantNode"],
        underlying_type: Optional[TypeNode] = None,
        attributes: List["AttributeNode"] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = name
        self.variants = variants
        self.underlying_type = underlying_type
        self.attributes = attributes or []

    def __repr__(self):
        return f"EnumNode(name={self.name}, variants={len(self.variants)})"


class EnumVariantNode(ASTNode):
    """Individual enum variant."""

    def __init__(
        self,
        name: str,
        value: Optional["ExpressionNode"] = None,
        fields: List[TypeNode] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = name
        self.value = value
        self.fields = fields or []  # For tagged unions

    def __repr__(self):
        return f"EnumVariantNode(name={self.name})"


class FunctionNode(ASTNode):
    """Function declarations."""

    def __init__(
        self,
        name: str,
        return_type: TypeNode,
        parameters: List["ParameterNode"],
        body: Optional["BlockNode"] = None,
        generic_params: List["GenericParameterNode"] = None,
        attributes: List["AttributeNode"] = None,
        visibility: str = "public",
        qualifiers: List[str] = None,
        is_unsafe: bool = False,
        is_async: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = name
        self.return_type = return_type
        self.parameters = parameters
        self.body = body
        self.generic_params = generic_params or []
        self.attributes = attributes or []
        self.visibility = visibility
        self.qualifiers = qualifiers or []  # __global__, __device__, inline, etc.
        self.is_unsafe = is_unsafe
        self.is_async = is_async

    def __repr__(self):
        return f"FunctionNode(name={self.name}, return_type={self.return_type})"


class ParameterNode(ASTNode):
    """Function parameter."""

    def __init__(
        self,
        name: str,
        param_type: TypeNode,
        default_value: Optional["ExpressionNode"] = None,
        attributes: List["AttributeNode"] = None,
        is_mutable: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = name
        self.param_type = param_type
        self.default_value = default_value
        self.attributes = attributes or []
        self.is_mutable = is_mutable

    def __repr__(self):
        return f"ParameterNode(name={self.name}, param_type={self.param_type})"


class VariableNode(ASTNode):
    """Variable declarations."""

    def __init__(
        self,
        name: str,
        var_type: TypeNode,
        initial_value: Optional["ExpressionNode"] = None,
        attributes: List["AttributeNode"] = None,
        qualifiers: List[str] = None,
        is_mutable: bool = True,
        visibility: str = "private",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = name
        self.var_type = var_type
        self.initial_value = initial_value
        self.attributes = attributes or []
        self.qualifiers = qualifiers or []  # const, volatile, __shared__, etc.
        self.is_mutable = is_mutable
        self.visibility = visibility

        # Legacy compatibility
        self.vtype = var_type
        self.semantic = self.get_semantic_from_attributes()

    def get_semantic_from_attributes(self):
        """Extract semantic information from attributes for legacy compatibility."""
        for attr in self.attributes:
            if attr.name in ["position", "color", "texcoord", "normal"]:
                return attr.name
        return None

    def __repr__(self):
        return f"VariableNode(name={self.name}, var_type={self.var_type})"


class ConstantNode(ASTNode):
    """Compile-time constants."""

    def __init__(
        self,
        name: str,
        const_type: TypeNode,
        value: "ExpressionNode",
        visibility: str = "public",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = name
        self.const_type = const_type
        self.value = value
        self.visibility = visibility

    def __repr__(self):
        return f"ConstantNode(name={self.name}, const_type={self.const_type})"


class GenericParameterNode(ASTNode):
    """Generic/template parameter."""

    def __init__(
        self,
        name: str,
        constraints: List[TypeNode] = None,
        default_type: Optional[TypeNode] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = name
        self.constraints = constraints or []
        self.default_type = default_type

    def __repr__(self):
        return f"GenericParameterNode(name={self.name})"


class AttributeNode(ASTNode):
    """Attributes/annotations/decorators."""

    def __init__(self, name: str, arguments: List["ExpressionNode"] = None, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.arguments = arguments or []

    def __repr__(self):
        return f"AttributeNode(name={self.name})"


# ============================================================================
# STATEMENTS
# ============================================================================


class StatementNode(ASTNode):
    """Base class for all statements."""


class BlockNode(StatementNode):
    """Block of statements."""

    def __init__(self, statements: List[StatementNode], **kwargs):
        super().__init__(**kwargs)
        self.statements = statements

    def __repr__(self):
        return f"BlockNode(statements={len(self.statements)})"


class ExpressionStatementNode(StatementNode):
    """Expression used as a statement."""

    def __init__(self, expression: "ExpressionNode", **kwargs):
        super().__init__(**kwargs)
        self.expression = expression

    def __repr__(self):
        return f"ExpressionStatementNode(expression={self.expression})"


class AssignmentNode(StatementNode):
    """Assignment operations."""

    def __init__(
        self,
        target: "ExpressionNode",
        value: "ExpressionNode",
        operator: str = "=",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target = target
        self.value = value
        self.operator = operator

        # Legacy compatibility
        self.left = target
        self.right = value

    def __repr__(self):
        return f"AssignmentNode(target={self.target}, operator={self.operator}, value={self.value})"


class IfNode(StatementNode):
    """Conditional statements."""

    def __init__(
        self,
        condition: "ExpressionNode",
        then_branch: StatementNode,
        else_branch: Optional[StatementNode] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.condition = condition
        self.then_branch = then_branch
        self.else_branch = else_branch

        # Legacy compatibility
        self.if_condition = condition
        self.if_body = then_branch
        self.else_if_conditions = []
        self.else_if_bodies = []
        self.else_body = else_branch

    def __repr__(self):
        return f"IfNode(condition={self.condition})"


class ForNode(StatementNode):
    """For loop statements."""

    def __init__(
        self,
        init: Optional[StatementNode],
        condition: Optional["ExpressionNode"],
        update: Optional["ExpressionNode"],
        body: StatementNode,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.init = init
        self.condition = condition
        self.update = update
        self.body = body

    def __repr__(self):
        return f"ForNode(condition={self.condition})"


class ForInNode(StatementNode):
    """For-in loop (Rust, Python style)."""

    def __init__(
        self, pattern: str, iterable: "ExpressionNode", body: StatementNode, **kwargs
    ):
        super().__init__(**kwargs)
        self.pattern = pattern
        self.iterable = iterable
        self.body = body

    def __repr__(self):
        return f"ForInNode(pattern={self.pattern})"


class WhileNode(StatementNode):
    """While loop statements."""

    def __init__(self, condition: "ExpressionNode", body: StatementNode, **kwargs):
        super().__init__(**kwargs)
        self.condition = condition
        self.body = body

    def __repr__(self):
        return f"WhileNode(condition={self.condition})"


class LoopNode(StatementNode):
    """Infinite loop (Rust style)."""

    def __init__(self, body: StatementNode, label: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.body = body
        self.label = label

    def __repr__(self):
        return f"LoopNode(label={self.label})"


class MatchNode(StatementNode):
    """Pattern matching (Rust, functional languages)."""

    def __init__(
        self, expression: "ExpressionNode", arms: List["MatchArmNode"], **kwargs
    ):
        super().__init__(**kwargs)
        self.expression = expression
        self.arms = arms

    def __repr__(self):
        return f"MatchNode(arms={len(self.arms)})"


class MatchArmNode(ASTNode):
    """Pattern matching arm."""

    def __init__(
        self,
        pattern: "PatternNode",
        guard: Optional["ExpressionNode"],
        body: StatementNode,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pattern = pattern
        self.guard = guard
        self.body = body

    def __repr__(self):
        return f"MatchArmNode(pattern={self.pattern})"


class SwitchNode(StatementNode):
    """Switch statements."""

    def __init__(
        self,
        expression: "ExpressionNode",
        cases: List["CaseNode"],
        default_case: Optional[StatementNode] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.expression = expression
        self.cases = cases
        self.default_case = default_case

    def __repr__(self):
        return f"SwitchNode(cases={len(self.cases)})"


class CaseNode(ASTNode):
    """Switch case."""

    def __init__(
        self, value: "ExpressionNode", statements: List[StatementNode], **kwargs
    ):
        super().__init__(**kwargs)
        self.value = value
        self.statements = statements

    def __repr__(self):
        return f"CaseNode(value={self.value})"


class ReturnNode(StatementNode):
    """Return statements."""

    def __init__(self, value: Optional["ExpressionNode"] = None, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def __repr__(self):
        return f"ReturnNode(value={self.value})"


class BreakNode(StatementNode):
    """Break statements."""

    def __init__(self, label: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.label = label

    def __repr__(self):
        return f"BreakNode(label={self.label})"


class ContinueNode(StatementNode):
    """Continue statements."""

    def __init__(self, label: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.label = label

    def __repr__(self):
        return f"ContinueNode(label={self.label})"


# ============================================================================
# EXPRESSIONS
# ============================================================================


class ExpressionNode(ASTNode):
    """Base class for all expressions."""

    def __init__(self, expression_type: Optional[TypeNode] = None, **kwargs):
        super().__init__(**kwargs)
        self.expression_type = expression_type  # Type of the expression result

        # Legacy compatibility for code generators that expect these
        self.vtype = expression_type
        self.name = getattr(self, "identifier", None)
        self.semantic = None


class LiteralNode(ExpressionNode):
    """Literal values."""

    def __init__(self, value: Any, literal_type: TypeNode, **kwargs):
        super().__init__(literal_type, **kwargs)
        self.value = value
        self.literal_type = literal_type

    def __repr__(self):
        return f"LiteralNode(value={self.value}, literal_type={self.literal_type})"


class IdentifierNode(ExpressionNode):
    """Variable/function identifiers."""

    def __init__(self, name: str, **kwargs):
        super().__init__(**kwargs)
        self.identifier = name
        self.name = name  # Legacy compatibility

    def __repr__(self):
        return f"IdentifierNode(name={self.identifier})"


class BinaryOpNode(ExpressionNode):
    """Binary operations."""

    def __init__(
        self, left: ExpressionNode, operator: str, right: ExpressionNode, **kwargs
    ):
        super().__init__(**kwargs)
        self.left = left
        self.operator = operator
        self.right = right
        self.op = operator  # Legacy compatibility

    def __repr__(self):
        return f"BinaryOpNode(left={self.left}, operator={self.operator}, right={self.right})"


class UnaryOpNode(ExpressionNode):
    """Unary operations."""

    def __init__(
        self, operator: str, operand: ExpressionNode, is_postfix: bool = False, **kwargs
    ):
        super().__init__(**kwargs)
        self.operator = operator
        self.operand = operand
        self.is_postfix = is_postfix
        self.op = operator  # Legacy compatibility

    def __repr__(self):
        return f"UnaryOpNode(operator={self.operator}, operand={self.operand}, is_postfix={self.is_postfix})"


class TernaryOpNode(ExpressionNode):
    """Ternary conditional operator."""

    def __init__(
        self,
        condition: ExpressionNode,
        true_expr: ExpressionNode,
        false_expr: ExpressionNode,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.condition = condition
        self.true_expr = true_expr
        self.false_expr = false_expr

    def __repr__(self):
        return f"TernaryOpNode(condition={self.condition}, true_expr={self.true_expr}, false_expr={self.false_expr})"


class FunctionCallNode(ExpressionNode):
    """Function calls."""

    def __init__(
        self,
        function: ExpressionNode,
        arguments: List[ExpressionNode],
        generic_args: List[TypeNode] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.function = function
        self.arguments = arguments
        self.generic_args = generic_args or []

        # Legacy compatibility
        self.name = function
        self.args = arguments

    def __repr__(self):
        return f"FunctionCallNode(function={self.function}, arguments={len(self.arguments)})"


class MemberAccessNode(ExpressionNode):
    """Member access (dot operator)."""

    def __init__(self, object_expr: ExpressionNode, member: str, **kwargs):
        super().__init__(**kwargs)
        self.object_expr = object_expr
        self.member = member

        # Legacy compatibility
        self.object = object_expr

    def __repr__(self):
        return f"MemberAccessNode(object={self.object_expr}, member={self.member})"


class PointerAccessNode(ExpressionNode):
    """Pointer member access (arrow operator)."""

    def __init__(self, pointer_expr: ExpressionNode, member: str, **kwargs):
        super().__init__(**kwargs)
        self.pointer_expr = pointer_expr
        self.member = member

    def __repr__(self):
        return f"PointerAccessNode(pointer={self.pointer_expr}, member={self.member})"


class ArrayAccessNode(ExpressionNode):
    """Array indexing."""

    def __init__(
        self, array_expr: ExpressionNode, index_expr: ExpressionNode, **kwargs
    ):
        super().__init__(**kwargs)
        self.array_expr = array_expr
        self.index_expr = index_expr

        # Legacy compatibility
        self.array = array_expr
        self.index = index_expr

    def __repr__(self):
        return f"ArrayAccessNode(array={self.array_expr}, index={self.index_expr})"


class SwizzleNode(ExpressionNode):
    """Vector swizzling (vec.xyz, vec.xxyy, etc.)."""

    def __init__(self, vector_expr: ExpressionNode, components: str, **kwargs):
        super().__init__(**kwargs)
        self.vector_expr = vector_expr
        self.components = components  # "xyz", "xxy", etc.

    def __repr__(self):
        return f"SwizzleNode(vector={self.vector_expr}, components={self.components})"


class CastNode(ExpressionNode):
    """Type casting."""

    def __init__(self, expression: ExpressionNode, target_type: TypeNode, **kwargs):
        super().__init__(target_type, **kwargs)
        self.expression = expression
        self.target_type = target_type

    def __repr__(self):
        return f"CastNode(expression={self.expression}, target_type={self.target_type})"


class ConstructorNode(ExpressionNode):
    """Type constructors (vec3(1,2,3), MyStruct{field: value})."""

    def __init__(
        self,
        constructor_type: TypeNode,
        arguments: List[ExpressionNode],
        named_arguments: Dict[str, ExpressionNode] = None,
        **kwargs,
    ):
        super().__init__(constructor_type, **kwargs)
        self.constructor_type = constructor_type
        self.arguments = arguments
        self.named_arguments = named_arguments or {}

    def __repr__(self):
        return f"ConstructorNode(constructor_type={self.constructor_type}, arguments={len(self.arguments)})"


class LambdaNode(ExpressionNode):
    """Lambda/closure expressions."""

    def __init__(
        self,
        parameters: List[ParameterNode],
        body: Union[ExpressionNode, BlockNode],
        captures: List[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.parameters = parameters
        self.body = body
        self.captures = captures or []  # Captured variables

    def __repr__(self):
        return f"LambdaNode(parameters={len(self.parameters)})"


# ============================================================================
# PATTERN MATCHING
# ============================================================================


class PatternNode(ASTNode):
    """Base class for patterns in pattern matching."""


class WildcardPatternNode(PatternNode):
    """Wildcard pattern (_)."""

    def __repr__(self):
        return "WildcardPatternNode()"


class IdentifierPatternNode(PatternNode):
    """Identifier pattern (variable binding)."""

    def __init__(self, name: str, **kwargs):
        super().__init__(**kwargs)
        self.name = name

    def __repr__(self):
        return f"IdentifierPatternNode(name={self.name})"


class LiteralPatternNode(PatternNode):
    """Literal pattern."""

    def __init__(self, literal: LiteralNode, **kwargs):
        super().__init__(**kwargs)
        self.literal = literal

    def __repr__(self):
        return f"LiteralPatternNode(literal={self.literal})"


class StructPatternNode(PatternNode):
    """Struct destructuring pattern."""

    def __init__(
        self, type_name: str, field_patterns: Dict[str, PatternNode], **kwargs
    ):
        super().__init__(**kwargs)
        self.type_name = type_name
        self.field_patterns = field_patterns

    def __repr__(self):
        return f"StructPatternNode(type_name={self.type_name})"


# ============================================================================
# GPU/GRAPHICS SPECIFIC NODES
# ============================================================================


class TextureNode(ExpressionNode):
    """Texture sampling operations."""

    def __init__(
        self,
        texture_expr: ExpressionNode,
        sampler_expr: ExpressionNode,
        coordinates: ExpressionNode,
        level: Optional[ExpressionNode] = None,
        offset: Optional[ExpressionNode] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.texture_expr = texture_expr
        self.sampler_expr = sampler_expr
        self.coordinates = coordinates
        self.level = level
        self.offset = offset

    def __repr__(self):
        return (
            f"TextureNode(texture={self.texture_expr}, coordinates={self.coordinates})"
        )


class AtomicOpNode(ExpressionNode):
    """Atomic operations for GPU computing."""

    def __init__(
        self,
        operation: str,
        target: ExpressionNode,
        arguments: List[ExpressionNode],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.operation = operation  # atomicAdd, atomicCAS, etc.
        self.target = target
        self.arguments = arguments

    def __repr__(self):
        return f"AtomicOpNode(operation={self.operation}, target={self.target})"


class SyncNode(StatementNode):
    """Synchronization operations."""

    def __init__(
        self, sync_type: str, arguments: List[ExpressionNode] = None, **kwargs
    ):
        super().__init__(**kwargs)
        self.sync_type = sync_type  # __syncthreads, barrier, etc.
        self.arguments = arguments or []

    def __repr__(self):
        return f"SyncNode(sync_type={self.sync_type})"


class BuiltinVariableNode(ExpressionNode):
    """Built-in variables (gl_Position, threadIdx, etc.)."""

    def __init__(self, builtin_name: str, component: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.builtin_name = builtin_name
        self.component = component

    def __repr__(self):
        return f"BuiltinVariableNode(builtin_name={self.builtin_name}, component={self.component})"


# ============================================================================
# MEMORY AND RESOURCE MANAGEMENT
# ============================================================================


class BufferNode(ASTNode):
    """Buffer resource declarations."""

    def __init__(
        self,
        name: str,
        buffer_type: TypeNode,
        binding: Optional[int] = None,
        set_: Optional[int] = None,
        access: str = "read_write",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = name
        self.buffer_type = buffer_type
        self.binding = binding
        self.set = set_
        self.access = access  # read, write, read_write

    def __repr__(self):
        return f"BufferNode(name={self.name}, buffer_type={self.buffer_type})"


class TextureResourceNode(ASTNode):
    """Texture resource declarations."""

    def __init__(
        self,
        name: str,
        texture_type: str,
        format: Optional[str] = None,
        binding: Optional[int] = None,
        set_: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = name
        self.texture_type = texture_type  # texture2D, textureCube, etc.
        self.format = format
        self.binding = binding
        self.set = set_

    def __repr__(self):
        return (
            f"TextureResourceNode(name={self.name}, texture_type={self.texture_type})"
        )


class SamplerNode(ASTNode):
    """Sampler resource declarations."""

    def __init__(
        self,
        name: str,
        filter_mode: str = "linear",
        address_mode: str = "clamp",
        binding: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = name
        self.filter_mode = filter_mode
        self.address_mode = address_mode
        self.binding = binding

    def __repr__(self):
        return f"SamplerNode(name={self.name})"


# ============================================================================
# LEGACY COMPATIBILITY HELPERS
# ============================================================================

# Create aliases for backward compatibility
CbufferNode = StructNode  # cbuffer is essentially a struct
VectorConstructorNode = ConstructorNode  # Vector constructors are just constructors


# Helper functions for backward compatibility
def create_legacy_shader_node(structs, functions, global_variables, cbuffers):
    """Create a ShaderNode with legacy parameters."""
    return ShaderNode(
        name="LegacyShader",
        execution_model=ExecutionModel.GRAPHICS_PIPELINE,
        structs=structs or [],
        functions=functions or [],
        global_variables=global_variables or [],
        constants=cbuffers or [],  # Map cbuffers to constants
    )


# Helper to create array nodes with legacy interface
class ArrayNode(VariableNode):
    """Legacy array node for backward compatibility."""

    def __init__(self, element_type, name, size=None, semantic=None, **kwargs):
        array_type = ArrayType(element_type, size)
        super().__init__(name, array_type, **kwargs)
        self.element_type = element_type
        self.size = size
        if semantic:
            self.attributes.append(AttributeNode(semantic))

        # Legacy compatibility
        self.vtype = f"{element_type}[]" if size is None else f"{element_type}[{size}]"


# Legacy compatibility - ensure all required classes are available
TernaryOpNode = TernaryOpNode  # Already defined above
