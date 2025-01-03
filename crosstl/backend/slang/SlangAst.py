class ASTNode:
    pass


class ShaderNode(ASTNode):
    """
    Represents a shader node in the AST.

    Attributes:
        imports (List[ImportNode]): The imports of the shader
        exports (List[ExportNode]): The exports of the shader
        structs (List[StructNode]): The structs of the shader
        typedefs (List[TypedefNode]): The typedefs of the shader
        functions (List[FunctionNode]): The functions of the shader
        global_vars (List[VariableNode]): The global variables of the shader
        cbuffers (List[VariableNode]): The constant buffers of the shader

    """

    def __init__(
        self,
        imports,
        exports,
        structs,
        typedefs,
        functions,
        global_vars,
        cbuffers,
    ):
        self.imports = imports
        self.exports = exports
        self.structs = structs
        self.typedefs = typedefs
        self.functions = functions
        self.global_vars = global_vars
        self.cbuffers = cbuffers

    def __repr__(self):
        return f"ShaderNode(imports={self.imports}, exports={self.exports}, structs={self.structs}, typedefs={self.typedefs}, functions={self.functions}), global_vars={self.global_vars}, cbuffers={self.cbuffers}"


class ImportNode(ASTNode):
    """

    Represents an import node in the AST.

    Attributes:
        module_name (str): The name of the module to import

    """

    def __init__(self, module_name):
        self.module_name = module_name

    def __repr__(self):
        return f"ImportNode(module_name='{self.module_name}')"


class ExportNode(ASTNode):
    """

    Represents an export node in the AST.

    Attributes:
        item (str): The item to export
    """

    def __init__(self, item):
        self.item = item

    def __repr__(self):
        return f"ExportNode(item={self.item})"


class StructNode(ASTNode):
    """

    Represents a struct node in the AST.

    Attributes:
        name (str): The name of the struct
        members (List[VariableNode]): The members of the struct

    """

    def __init__(self, name, members):
        self.name = name
        self.members = members

    def __repr__(self):
        return f"StructNode(name='{self.name}', members={self.members})"


class TypedefNode(ASTNode):
    """

    Represents a typedef node in the AST.

    Attributes:
        original_type (str): The original type
        new_type (str): The new type

    """

    def __init__(self, original_type, new_type):
        self.original_type = original_type
        self.new_type = new_type

    def __repr__(self):
        return f"TypedefNode(original_type='{self.original_type}', new_type='{self.new_type}')"


class FunctionNode(ASTNode):
    """

    Represents a function node in the AST.

    Attributes:
        return_type (str): The return type of the function
        name (str): The name of the function
        params (List[VariableNode]): The parameters of the function
        body (List[ASTNode]): The body of the function
        is_generic (bool): Whether the function is generic
        type_function (str): The type of function

    """

    def __init__(
        self,
        return_type,
        name,
        params,
        body,
        is_generic=False,
        qualifier=None,
        semantic=None,
    ):
        self.return_type = return_type
        self.name = name
        self.params = params
        self.body = body
        self.is_generic = is_generic
        self.qualifier = qualifier
        self.semantic = semantic

    def __repr__(self):
        return f"FunctionNode(return_type='{self.return_type}', name='{self.name}', params={self.params}, body={self.body}, is_generic={self.is_generic}, qualifier={self.qualifier}, semantic={self.semantic})"


class VariableNode(ASTNode):
    """


    Represents a variable node in the AST.

    Attributes:
        vtype (str): The type of the variable
        name (str): The name of the variable
        attributes (List[AttributeNode]): The attributes of the variable

    """

    def __init__(self, vtype, name, semantic=None):
        self.vtype = vtype
        self.name = name
        self.semantic = semantic

    def __repr__(self):
        return f"VariableNode(vtype='{self.vtype}', name='{self.name}', semantic={self.semantic})"


class AssignmentNode(ASTNode):
    """


    Represents an assignment node in the AST.

    Attributes:
        left (VariableNode): The left side of the assignment
        right (ASTNode): The right side of the assignment
        operator (str): The operator of the assignment

    """

    def __init__(self, left, right, operator="="):
        self.left = left
        self.right = right
        self.operator = operator

    def __repr__(self):
        return f"AssignmentNode(left={self.left}, operator='{self.operator}', right={self.right})"


class IfNode(ASTNode):
    """

    Represents an if node in the AST.

    Attributes:
        condition (ASTNode): The condition of the if statement
        if_body (List[ASTNode]): The body of the if statement

    """

    def __init__(self, condition, if_body, else_body=None):
        self.condition = condition
        self.if_body = if_body
        self.else_body = else_body

    def __repr__(self):
        return f"IfNode(condition={self.condition}, if_body={self.if_body}, else_body={self.else_body})"


class ForNode(ASTNode):
    """

    Represents a for node in the AST.

    Attributes:
        init (ASTNode): The initialization of the for loop
        condition (ASTNode): The condition of the for loop
        update (ASTNode): The update of the for loop
        body (List[ASTNode]): The body of the for loop

    """

    def __init__(self, init, condition, update, body):
        self.init = init
        self.condition = condition
        self.update = update
        self.body = body

    def __repr__(self):
        return f"ForNode(init={self.init}, condition={self.condition}, update={self.update}, body={self.body})"


class ReturnNode(ASTNode):
    """

    Represents a return node in the AST.

    Attributes:
        value (ASTNode): The value to return

    """

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"ReturnNode(value={self.value})"


class FunctionCallNode(ASTNode):
    """

    Represents a function call node in the AST.


    Attributes:
        name (str): The name of the function
        args (List[ASTNode]): The arguments of the function

    """

    def __init__(self, name, args):
        self.name = name
        self.args = args

    def __repr__(self):
        return f"FunctionCallNode(name='{self.name}', args={self.args})"


class BinaryOpNode(ASTNode):
    """

    Represents a binary operation node in the AST.

    Attributes:
        left (ASTNode): The left side of the operation
        op (str): The operator of the operation
        right (ASTNode): The right side of the operation

    """

    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

    def __repr__(self):
        return f"BinaryOpNode(left={self.left}, op='{self.op}', right={self.right})"


class UnaryOpNode(ASTNode):
    """

    Represents a unary operation node in the AST.

    Attributes:
        op (str): The operator of the operation
        operand (ASTNode): The operand of the operation

    """

    def __init__(self, op, operand):
        self.op = op
        self.operand = operand

    def __repr__(self):
        return f"UnaryOpNode(op='{self.op}', operand={self.operand})"


class MemberAccessNode(ASTNode):
    """

    Represents a member access node in the AST.

    Attributes:
        object (ASTNode): The object to access the member of
        member (str): The member to access

    """

    def __init__(self, object, member):
        self.object = object
        self.member = member

    def __repr__(self):
        return f"MemberAccessNode(object={self.object}, member='{self.member}')"


class VectorConstructorNode(ASTNode):
    """

    Represents a vector constructor node in the AST.

    Attributes:

        type_name (str): The type of the vector
        args (List[ASTNode]): The arguments of the vector

    """

    def __init__(self, type_name, args):
        self.type_name = type_name
        self.args = args

    def __repr__(self):
        return f"VectorConstructorNode(type_name='{self.type_name}', args={self.args})"


class TernaryOpNode(ASTNode):
    """

    Represents a ternary operation node in the AST.

    Attributes:

        condition (ASTNode): The condition of the operation
        true_expr (ASTNode): The true expression of the operation
        false_expr (ASTNode): The false expression of the operation

    """

    def __init__(self, condition, true_expr, false_expr):
        self.condition = condition
        self.true_expr = true_expr
        self.false_expr = false_expr

    def __repr__(self):
        return f"TernaryOpNode(condition={self.condition}, true_expr={self.true_expr}, false_expr={self.false_expr})"


class GenericNode(ASTNode):
    """

    Represents a generic node in the AST.
    Attributes:

            type_params (List[str]): The type parameters of the generic
            body (List[ASTNode]): The body of the generic

    """

    def __init__(self, type_params, body):
        self.type_params = type_params
        self.body = body

    def __repr__(self):
        return f"GenericNode(type_params={self.type_params}, body={self.body})"


class ExtensionNode(ASTNode):
    """

    Represents an extension node in the AST.

    Attributes:
        name (str): The name of the extension
        body (List[ASTNode]): The body of the extension

    """

    def __init__(self, name, body):
        self.name = name
        self.body = body

    def __repr__(self):
        return f"ExtensionNode(name='{self.name}', body={self.body})"
