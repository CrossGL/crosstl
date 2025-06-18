class ASTNode:
    """Base class for all AST nodes."""


class TernaryOpNode:
    def __init__(self, condition, true_expr, false_expr):
        self.condition = condition
        self.true_expr = true_expr
        self.false_expr = false_expr

    def __repr__(self):
        return f"TernaryOpNode(condition={self.condition}, true_expr={self.true_expr}, false_expr={self.false_expr})"


class ShaderNode:
    def __init__(
        self,
        structs,
        functions,
        global_variables,
        impl_blocks=None,
        use_statements=None,
    ):
        self.structs = structs
        self.functions = functions
        self.global_variables = global_variables
        self.impl_blocks = impl_blocks if impl_blocks else []
        self.use_statements = use_statements if use_statements else []

    def __repr__(self):
        return f"ShaderNode(structs={self.structs}, functions={self.functions}, global_variables={self.global_variables}, impl_blocks={self.impl_blocks}, use_statements={self.use_statements})"


class StructNode(ASTNode):
    """Node representing a struct declaration."""

    def __init__(self, name, members, attributes=None, visibility=None, generics=None):
        self.name = name
        self.members = members  # List of VariableNode objects
        self.attributes = (
            attributes if attributes else []
        )  # Rust attributes like #[repr(C)]
        self.visibility = visibility  # pub, pub(crate), etc.
        self.generics = generics if generics else []  # Generic parameters

    def __repr__(self):
        return f"StructNode(name={self.name}, members={self.members}, attributes={self.attributes}, visibility={self.visibility}, generics={self.generics})"

    def accept(self, visitor):
        return visitor.visit_StructNode(self)


class ImplNode(ASTNode):
    """Node representing an impl block."""

    def __init__(self, struct_name, functions, trait_name=None, generics=None):
        self.struct_name = struct_name
        self.functions = functions
        self.trait_name = trait_name  # For trait implementations
        self.generics = generics if generics else []

    def __repr__(self):
        return f"ImplNode(struct_name={self.struct_name}, functions={self.functions}, trait_name={self.trait_name}, generics={self.generics})"


class TraitNode(ASTNode):
    """Node representing a trait declaration."""

    def __init__(self, name, functions, generics=None, visibility=None):
        self.name = name
        self.functions = functions
        self.generics = generics if generics else []
        self.visibility = visibility

    def __repr__(self):
        return f"TraitNode(name={self.name}, functions={self.functions}, generics={self.generics}, visibility={self.visibility})"


class FunctionNode(ASTNode):
    def __init__(
        self,
        return_type,
        name,
        params,
        body,
        attributes=None,
        visibility=None,
        generics=None,
        is_unsafe=False,
    ):
        self.return_type = return_type
        self.name = name
        self.params = params
        self.body = body
        self.attributes = (
            attributes if attributes else []
        )  # Rust attributes like #[vertex_shader]
        self.visibility = visibility  # pub, pub(crate), etc.
        self.generics = generics if generics else []
        self.is_unsafe = is_unsafe

    def __repr__(self):
        return f"FunctionNode(return_type={self.return_type}, name={self.name}, params={self.params}, body={self.body}, attributes={self.attributes}, visibility={self.visibility}, generics={self.generics}, is_unsafe={self.is_unsafe})"


class VariableNode(ASTNode):
    def __init__(self, vtype, name, is_mutable=False, attributes=None):
        self.vtype = vtype
        self.name = name
        self.is_mutable = is_mutable  # mut keyword
        self.attributes = attributes if attributes else []  # Rust attributes

    def __repr__(self):
        return f"VariableNode(vtype='{self.vtype}', name='{self.name}', is_mutable={self.is_mutable}, attributes={self.attributes})"


class LetNode(ASTNode):
    """Node representing a let binding."""

    def __init__(self, name, value, vtype=None, is_mutable=False):
        self.name = name
        self.value = value
        self.vtype = vtype  # Optional type annotation
        self.is_mutable = is_mutable

    def __repr__(self):
        return f"LetNode(name={self.name}, value={self.value}, vtype={self.vtype}, is_mutable={self.is_mutable})"


class AssignmentNode(ASTNode):
    def __init__(self, left, right, operator="="):
        self.left = left
        self.right = right
        self.operator = operator

    def __repr__(self):
        return f"AssignmentNode(left={self.left}, operator='{self.operator}', right={self.right})"


class IfNode(ASTNode):
    def __init__(self, condition, if_body, else_body=None):
        self.condition = condition
        self.if_body = if_body
        self.else_body = else_body

    def __repr__(self):
        return f"IfNode(condition={self.condition}, if_body={self.if_body}, else_body={self.else_body})"


class ForNode(ASTNode):
    def __init__(self, pattern, iterable, body):
        self.pattern = pattern  # Pattern to match (e.g., variable name)
        self.iterable = iterable  # What to iterate over
        self.body = body

    def __repr__(self):
        return f"ForNode(pattern={self.pattern}, iterable={self.iterable}, body={self.body})"


class WhileNode(ASTNode):
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

    def __repr__(self):
        return f"WhileNode(condition={self.condition}, body={self.body})"


class LoopNode(ASTNode):
    """Node representing an infinite loop."""

    def __init__(self, body, label=None):
        self.body = body
        self.label = label  # Optional loop label

    def __repr__(self):
        return f"LoopNode(body={self.body}, label={self.label})"


class MatchNode(ASTNode):
    """Node representing a match expression."""

    def __init__(self, expression, arms):
        self.expression = expression
        self.arms = arms  # List of MatchArmNode objects

    def __repr__(self):
        return f"MatchNode(expression={self.expression}, arms={self.arms})"


class MatchArmNode(ASTNode):
    """Node representing a match arm."""

    def __init__(self, pattern, guard, body):
        self.pattern = pattern
        self.guard = guard  # Optional guard condition
        self.body = body

    def __repr__(self):
        return f"MatchArmNode(pattern={self.pattern}, guard={self.guard}, body={self.body})"


class ReturnNode(ASTNode):
    def __init__(self, value=None):
        self.value = value

    def __repr__(self):
        return f"ReturnNode(value={self.value})"


class BreakNode(ASTNode):
    def __init__(self, label=None, value=None):
        self.label = label  # Optional loop label
        self.value = value  # Optional break value

    def __repr__(self):
        return f"BreakNode(label={self.label}, value={self.value})"


class ContinueNode(ASTNode):
    def __init__(self, label=None):
        self.label = label  # Optional loop label

    def __repr__(self):
        return f"ContinueNode(label={self.label})"


class FunctionCallNode(ASTNode):
    def __init__(self, name, args, generics=None):
        self.name = name
        self.args = args
        self.generics = generics if generics else []

    def __repr__(self):
        return f"FunctionCallNode(name={self.name}, args={self.args}, generics={self.generics})"


class BinaryOpNode(ASTNode):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

    def __repr__(self):
        return f"BinaryOpNode(left={self.left}, op={self.op}, right={self.right})"


class MemberAccessNode(ASTNode):
    def __init__(self, object, member):
        self.object = object
        self.member = member

    def __repr__(self):
        return f"MemberAccessNode(object={self.object}, member={self.member})"


class VectorConstructorNode:
    def __init__(self, type_name, args):
        self.type_name = type_name
        self.args = args

    def __repr__(self):
        return f"VectorConstructorNode(type_name={self.type_name}, args={self.args})"


class UnaryOpNode(ASTNode):
    def __init__(self, op, operand):
        self.op = op
        self.operand = operand

    def __repr__(self):
        return f"UnaryOpNode(operator={self.op}, operand={self.operand})"

    def __str__(self):
        return f"({self.op}{self.operand})"


class UseNode(ASTNode):
    """Node representing a use statement."""

    def __init__(self, path, alias=None, visibility=None):
        self.path = path  # The path being imported
        self.alias = alias  # Optional alias (as keyword)
        self.visibility = visibility  # pub use

    def __repr__(self):
        return f"UseNode(path={self.path}, alias={self.alias}, visibility={self.visibility})"


class AttributeNode(ASTNode):
    """Node representing an attribute (e.g., #[derive(Debug)])."""

    def __init__(self, name, args=None):
        self.name = name
        self.args = args if args else []

    def __repr__(self):
        return f"AttributeNode(name={self.name}, args={self.args})"


class GenericParameterNode(ASTNode):
    """Node representing a generic parameter."""

    def __init__(self, name, bounds=None, default=None):
        self.name = name
        self.bounds = bounds if bounds else []  # Trait bounds
        self.default = default  # Default type

    def __repr__(self):
        return f"GenericParameterNode(name={self.name}, bounds={self.bounds}, default={self.default})"


class ArrayAccessNode(ASTNode):
    """Node representing array/slice access."""

    def __init__(self, array, index):
        self.array = array
        self.index = index

    def __repr__(self):
        return f"ArrayAccessNode(array={self.array}, index={self.index})"


class RangeNode(ASTNode):
    """Node representing a range expression."""

    def __init__(self, start, end, inclusive=False):
        self.start = start
        self.end = end
        self.inclusive = inclusive  # .. vs ..=

    def __repr__(self):
        return (
            f"RangeNode(start={self.start}, end={self.end}, inclusive={self.inclusive})"
        )


class TupleNode(ASTNode):
    """Node representing a tuple."""

    def __init__(self, elements):
        self.elements = elements

    def __repr__(self):
        return f"TupleNode(elements={self.elements})"


class ArrayNode(ASTNode):
    """Node representing an array literal."""

    def __init__(self, elements):
        self.elements = elements

    def __repr__(self):
        return f"ArrayNode(elements={self.elements})"


class ReferenceNode(ASTNode):
    """Node representing a reference (&expr)."""

    def __init__(self, expression, is_mutable=False):
        self.expression = expression
        self.is_mutable = is_mutable

    def __repr__(self):
        return (
            f"ReferenceNode(expression={self.expression}, is_mutable={self.is_mutable})"
        )


class DereferenceNode(ASTNode):
    """Node representing a dereference (*expr)."""

    def __init__(self, expression):
        self.expression = expression

    def __repr__(self):
        return f"DereferenceNode(expression={self.expression})"


class CastNode(ASTNode):
    """Node representing a type cast (expr as Type)."""

    def __init__(self, expression, target_type):
        self.expression = expression
        self.target_type = target_type

    def __repr__(self):
        return f"CastNode(expression={self.expression}, target_type={self.target_type})"


class BlockNode(ASTNode):
    """Node representing a block expression."""

    def __init__(self, statements, expression=None):
        self.statements = statements
        self.expression = expression  # Optional final expression

    def __repr__(self):
        return f"BlockNode(statements={self.statements}, expression={self.expression})"


class ConstNode(ASTNode):
    """Node representing a const declaration."""

    def __init__(self, name, vtype, value, visibility=None):
        self.name = name
        self.vtype = vtype
        self.value = value
        self.visibility = visibility

    def __repr__(self):
        return f"ConstNode(name={self.name}, vtype={self.vtype}, value={self.value}, visibility={self.visibility})"


class StaticNode(ASTNode):
    """Node representing a static declaration."""

    def __init__(self, name, vtype, value, is_mutable=False, visibility=None):
        self.name = name
        self.vtype = vtype
        self.value = value
        self.is_mutable = is_mutable
        self.visibility = visibility

    def __repr__(self):
        return f"StaticNode(name={self.name}, vtype={self.vtype}, value={self.value}, is_mutable={self.is_mutable}, visibility={self.visibility})"


class StructInitializationNode(ASTNode):
    """Node representing struct initialization syntax: Name { field: value, ... }"""

    def __init__(self, struct_name, fields):
        self.struct_name = struct_name
        self.fields = fields  # List of (field_name, field_value) tuples

    def __repr__(self):
        return f"StructInitializationNode(struct_name={self.struct_name}, fields={self.fields})"
