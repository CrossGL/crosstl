"""Restore the pointer contract of buffer-attributed helper parameters.

The Metal frontend lowers every ``[[buffer(n)]]`` pointer parameter to a
portable ``StructuredBuffer``/``RWStructuredBuffer`` declaration.  On a stage
entry that is the resource binding.  On an ordinary helper (MLX's
``GEMVTKernel::run``, for example) the attribute has no binding meaning in
Metal: the parameter is still a plain pointer that receives whatever pointer
the caller passes, including one the caller already advanced by a batch
offset.  A resource handle cannot carry that rebase, so a helper reading
``buffer_load(mat, i)`` would index the caller's unrebased root.

This pass rewrites such helper parameters back to device/constant pointers,
and their ``buffer_load``/``buffer_store`` accesses back to indexing, so the
HLSL generator's existing pointer ABI forwards each caller's logical offset.
It only rewrites a parameter when every use and every call site has a form
that the pointer ABI already supports; anything else keeps the resource form.
"""

from ..ast import (
    AST_CHILD_FIELD_EXCLUSIONS,
    ArrayAccessNode,
    AssignmentNode,
    ASTNode,
    BinaryOpNode,
    ExpressionStatementNode,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    LambdaNode,
    NamedType,
    PointerType,
    UnaryOpNode,
    VariableNode,
)
from .stage_utils import collect_stage_entry_records

_RESOURCE_READ_ONLY = {"StructuredBuffer": True, "RWStructuredBuffer": False}
_POINTER_ADDRESS_SPACES = frozenset({"constant", "device", "global", "storage"})
_POINTER_UPDATE_OPERATORS = frozenset({"+=", "-="})
_POINTER_STEP_OPERATORS = frozenset({"++", "--"})


def _identifier_name(node):
    return node.name if isinstance(node, IdentifierNode) else None


def _pointer_root_name(node):
    """Return the identifier at the root of a ``root + offset`` chain."""
    while isinstance(node, BinaryOpNode) and node.operator in {"+", "-"}:
        node = node.left
    return _identifier_name(node)


def _call_name(node):
    function = getattr(node, "function", None)
    if isinstance(function, IdentifierNode):
        return function.name
    return function if isinstance(function, str) else None


def _attribute_names(parameter):
    return {
        str(getattr(attribute, "name", attribute)).lower()
        for attribute in getattr(parameter, "attributes", []) or []
    }


def _buffer_parameter_contract(parameter):
    """Return ``(element type, read only, address space)`` for a helper buffer."""
    value = getattr(parameter, "param_type", None)
    if not isinstance(value, NamedType) or value.name not in _RESOURCE_READ_ONLY:
        return None
    generic_args = list(getattr(value, "generic_args", None) or [])
    attributes = _attribute_names(parameter)
    if len(generic_args) != 1 or "buffer" not in attributes:
        return None
    address_space = "constant" if "constant" in attributes else "device"
    return generic_args[0], _RESOURCE_READ_ONLY[value.name], address_space


def _storage_pointer_parameter(parameter):
    value = getattr(parameter, "param_type", None)
    if not isinstance(value, PointerType):
        return False
    address_space = str(value.address_space or "").strip().lower()
    if not address_space:
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(parameter, "qualifiers", []) or []
        }
        address_space = next(
            (
                candidate
                for candidate in ("device", "global", "storage")
                if candidate in qualifiers
            ),
            "",
        )
    return address_space in _POINTER_ADDRESS_SPACES


def _resource_type(value):
    return isinstance(value, NamedType) and value.name in _RESOURCE_READ_ONLY


def _walk_with_parents(root):
    """Yield ``(node, parent, field)`` once per structural node below ``root``."""
    visited = set()
    stack = [(root, None, None)]
    while stack:
        node, parent, field = stack.pop()
        if id(node) in visited:
            continue
        visited.add(id(node))
        yield node, parent, field
        children = []
        for name, value in vars(node).items():
            if isinstance(value, ASTNode):
                if name not in AST_CHILD_FIELD_EXCLUSIONS:
                    children.append((value, node, name))
            elif isinstance(value, (list, tuple)):
                if name not in AST_CHILD_FIELD_EXCLUSIONS:
                    children.extend(
                        (child, node, name)
                        for child in value
                        if isinstance(child, ASTNode)
                    )
        children.reverse()
        stack.extend(children)


class _BodyIndex:
    """One structural walk of a function body, shared by every analysis query.

    The analysis asks the same body about each of its buffer parameters and
    then for its call sites; a single walk answers all of those questions.
    The index is only valid until the body is rewritten.
    """

    __slots__ = (
        "parents",
        "statement_calls",
        "nested_scope",
        "variables",
        "uses",
        "calls",
    )

    def __init__(self, body):
        self.parents = {}
        self.statement_calls = set()
        self.nested_scope = False
        self.variables = set()
        # Identifier name -> [(node, parent, field)] in walk order.
        self.uses = {}
        self.calls = []
        for node, parent, field in _walk_with_parents(body):
            self.parents[id(node)] = (parent, field)
            if isinstance(node, ExpressionStatementNode):
                self.statement_calls.add(id(node.expression))
            if isinstance(node, (LambdaNode, FunctionNode)):
                self.nested_scope = True
            if isinstance(node, VariableNode):
                self.variables.add(node.name)
            if isinstance(node, IdentifierNode):
                self.uses.setdefault(node.name, []).append((node, parent, field))
            if isinstance(node, FunctionCallNode):
                self.calls.append(node)


class _HelperBufferParameters:
    def __init__(self, ast, stage_entry_types):
        self.ast = ast
        self.entry_ids = {
            entry_id
            for entry_id, _stage, _function in collect_stage_entry_records(
                ast, None, stage_entry_types
            )
        }
        self.functions = []
        seen = set()
        for function in self.all_functions():
            if id(function) not in seen:
                seen.add(id(function))
                self.functions.append(function)
        self.body_indexes = {}
        self.by_name = {}
        for function in self.functions:
            self.by_name.setdefault(function.name, []).append(function)
        self.global_resources = {
            variable.name
            for variable in self.global_variables()
            if _resource_type(getattr(variable, "var_type", None))
        }

    def all_functions(self):
        for function in getattr(self.ast, "functions", []) or []:
            if isinstance(function, FunctionNode):
                yield function
        for _stage_type, stage in getattr(self.ast, "stages", {}).items():
            entry_point = getattr(stage, "entry_point", None)
            if isinstance(entry_point, FunctionNode):
                yield entry_point
            for function in getattr(stage, "local_functions", []) or []:
                if isinstance(function, FunctionNode):
                    yield function

    def global_variables(self):
        yield from getattr(self.ast, "global_variables", []) or []
        for _stage_type, stage in getattr(self.ast, "stages", {}).items():
            yield from getattr(stage, "local_variables", []) or []

    def body_index(self, function):
        index = self.body_indexes.get(id(function))
        if index is None:
            index = _BodyIndex(function.body)
            self.body_indexes[id(function)] = index
        return index

    def unique_function(self, name):
        candidates = self.by_name.get(name, ())
        return candidates[0] if len(candidates) == 1 else None

    def candidates(self):
        found = {}
        for function in self.functions:
            if id(function) in self.entry_ids or function.body is None:
                continue
            if getattr(function, "generic_params", None):
                continue
            if self.unique_function(function.name) is not function:
                continue
            for index, parameter in enumerate(function.parameters or []):
                contract = _buffer_parameter_contract(parameter)
                if contract is not None:
                    found[(id(function), index)] = (function, parameter, contract)
        return found

    def body_uses(self, function, name):
        """Classify every use of ``name``; return ``None`` for unsupported forms."""
        uses = {"forwarded": []}
        index = self.body_index(function)
        if index.nested_scope or name in index.variables:
            return None
        parents = index.parents
        statement_calls = index.statement_calls
        for node, parent, field in index.uses.get(name, ()):
            if isinstance(parent, AssignmentNode) and field == "target":
                if parent.operator in _POINTER_UPDATE_OPERATORS:
                    continue
                return None
            if isinstance(parent, UnaryOpNode) and parent.operator in (
                _POINTER_STEP_OPERATORS
            ):
                continue
            # `name + offset` chains are pointer arithmetic on the same root.
            view = node
            while (
                isinstance(parent, BinaryOpNode)
                and field == "left"
                and parent.operator in {"+", "-"}
            ):
                view = parent
                parent, field = parents.get(id(parent), (None, None))
            if isinstance(parent, ArrayAccessNode) and field == "array_expr":
                continue
            if isinstance(parent, UnaryOpNode) and parent.operator == "*":
                continue
            if not (isinstance(parent, FunctionCallNode) and field == "arguments"):
                return None
            call_name = _call_name(parent)
            arguments = list(parent.arguments or [])
            position = next(
                index for index, argument in enumerate(arguments) if argument is view
            )
            if call_name == "buffer_load" and len(arguments) == 2:
                if position != 0:
                    return None
                continue
            if call_name == "buffer_store" and len(arguments) == 3:
                if position != 0 or id(parent) not in statement_calls:
                    return None
                continue
            callee = self.unique_function(call_name) if call_name else None
            if callee is None or len(callee.parameters or []) != len(arguments):
                return None
            uses["forwarded"].append((callee, position))
        return uses

    def call_sites(self):
        sites = []
        for caller in self.functions:
            if caller.body is None:
                continue
            for node in self.body_index(caller).calls:
                callee = self.unique_function(_call_name(node))
                if callee is not None:
                    sites.append((caller, callee, list(node.arguments or [])))
        return sites

    def argument_supported(self, caller, argument, valid):
        # The pointer ABI resolves `root + offset` arguments to the root's view
        # plus the forwarded element offset, exactly as for source pointers.
        name = _pointer_root_name(argument)
        if name is None:
            return False
        for index, parameter in enumerate(caller.parameters or []):
            if parameter.name != name:
                continue
            if _storage_pointer_parameter(parameter):
                return True
            if id(caller) in self.entry_ids:
                return _resource_type(getattr(parameter, "param_type", None))
            return (id(caller), index) in valid
        return name in self.global_resources

    def run(self):
        candidates = self.candidates()
        uses = {}
        for key, (function, parameter, _contract) in candidates.items():
            classified = self.body_uses(function, parameter.name)
            if classified is not None:
                uses[key] = classified
        valid = set(uses)
        sites = {}
        for caller, callee, arguments in self.call_sites():
            sites.setdefault(id(callee), []).append((caller, arguments))
        changed = True
        while changed:
            changed = False
            for key in sorted(valid):
                function, _parameter, _contract = candidates[key]
                supported = all(
                    (id(callee), position) in valid
                    or _storage_pointer_parameter(callee.parameters[position])
                    for callee, position in uses[key]["forwarded"]
                )
                index = key[1]
                for caller, arguments in sites.get(id(function), ()):
                    if not supported:
                        break
                    supported = len(arguments) == len(
                        function.parameters
                    ) and self.argument_supported(caller, arguments[index], valid)
                if not supported:
                    valid.discard(key)
                    changed = True
        rewrites = {}
        for key in sorted(valid):
            function, parameter, contract = candidates[key]
            rewrites.setdefault(key[0], (function, []))[1].append((parameter, contract))
        for function, parameters in rewrites.values():
            for parameter, contract in parameters:
                self.retype(parameter, contract)
            self.rewrite_accesses(
                function, {parameter.name for parameter, _contract in parameters}
            )
        return self.ast

    def retype(self, parameter, contract):
        element_type, read_only, address_space = contract
        parameter.param_type = PointerType(
            element_type,
            is_mutable=True,
            address_space=address_space,
            source_location=getattr(parameter.param_type, "source_location", None),
        )
        parameter.qualifiers = (
            ["const", address_space] if read_only else [address_space]
        ) + [
            qualifier
            for qualifier in parameter.qualifiers or []
            if str(qualifier).lower() not in {"const", "constant", "device"}
        ]
        parameter.attributes = [
            attribute
            for attribute in parameter.attributes or []
            if str(getattr(attribute, "name", attribute)).lower()
            not in {"buffer", "constant"}
        ]

    def rewrite_accesses(self, function, names):
        """Turn ``buffer_load``/``buffer_store`` of ``names`` into indexing."""
        replacements = {}
        # Every node built here holds original arguments that may themselves be
        # replaced, so each one is patched along with the original body.
        created = []
        for node, _parent, _field in _walk_with_parents(function.body):
            if not isinstance(node, FunctionCallNode):
                continue
            arguments = list(node.arguments or [])
            if not arguments or _pointer_root_name(arguments[0]) not in names:
                continue
            location = getattr(node, "source_location", None)
            call_name = _call_name(node)
            if call_name == "buffer_load":
                access = ArrayAccessNode(
                    arguments[0], arguments[1], source_location=location
                )
                replacements[id(node)] = access
                created.append(access)
            elif call_name == "buffer_store":
                access = ArrayAccessNode(
                    arguments[0], arguments[1], source_location=location
                )
                store = AssignmentNode(access, arguments[2], source_location=location)
                replacements[id(node)] = store
                created.extend((access, store))
        if not replacements:
            return
        nodes = [node for node, _parent, _field in _walk_with_parents(function.body)]
        nodes.extend(created)
        for node in nodes:
            for name, value in list(vars(node).items()):
                if name in AST_CHILD_FIELD_EXCLUSIONS:
                    continue
                if isinstance(value, ASTNode) and id(value) in replacements:
                    setattr(node, name, replacements[id(value)])
                elif isinstance(value, list):
                    for position, child in enumerate(value):
                        if isinstance(child, ASTNode) and id(child) in replacements:
                            value[position] = replacements[id(child)]


def lower_helper_buffer_parameters(ast, stage_entry_types):
    """Rewrite buffer-attributed helper parameters to pointers in place.

    ``ast`` must already be a private copy: the HLSL generator calls this after
    its reference-alias lowering, which deep-copies the caller's source AST.
    """
    if not isinstance(ast, ASTNode):
        return ast
    return _HelperBufferParameters(ast, frozenset(stage_entry_types)).run()
