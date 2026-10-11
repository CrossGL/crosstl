"""Keep logical float values distinct from integer-backed atomic storage."""

from contextlib import contextmanager

from ..ast import (
    ArrayAccessNode,
    FunctionCallNode,
    IdentifierNode,
    MemberAccessNode,
    PointerAccessNode,
    UnaryOpNode,
    VariableNode,
)
from .array_utils import format_c_style_array_declaration, split_array_type_suffix


class OpenGLFloatAtomicStorageError(ValueError):
    project_diagnostic_code = (
        "project.translate.opengl-float-atomic-storage-unsupported"
    )
    missing_capabilities = ("float-atomic-storage-lowering",)

    def __init__(self, reason, node=None):
        super().__init__(f"Cannot preserve OpenGL float atomic storage: {reason}")
        self.reason = reason
        self.source_location = getattr(node, "source_location", None)


class FloatAtomicStorage:
    """Discover allocations with normal binding resolution, then emit word storage.

    Discovery output is never published when a float atomic is found. Reusing the
    normal resolver retains resource specialization, checked indices and scopes.
    """

    OPERATIONS = {
        "atomicLoad": 1,
        "atomicStore": 2,
        "atomicExchange": 2,
        "atomicAdd": 2,
        "atomicCompareExchangeWeak": 3,
        "atomicCompSwap": 3,
    }
    WORD_TYPES = {"float": "uint", "vec2": "uvec2", "vec3": "uvec3", "vec4": "uvec4"}

    def __init__(self, generator, *, discover=False, roots=()):
        self.generator = generator
        self.discover = discover
        self.roots = set(roots)
        self.declared = set()
        self.raw_nodes = set()
        self.types = {}
        self.definitions = []
        self.helpers = {}

    def identifier(self, key, name):
        return self.generator.glsl_generated_module_identifier(
            ("float-storage", key), name
        )

    def physical_type(self, logical):
        g = self.generator
        source_base, source_suffix = split_array_type_suffix(
            g.type_name_string(logical)
        )
        if source_suffix:
            return self.physical_type(source_base) + source_suffix
        if g.glsl_half_width(logical) is not None:
            raise OpenGLFloatAtomicStorageError("narrow-float-storage-allocation")
        logical = g.map_type(logical)
        base, suffix = split_array_type_suffix(logical)
        if suffix:
            return self.physical_type(base) + suffix
        if base in self.WORD_TYPES:
            return self.WORD_TYPES[base]
        source = g.glsl_source_type_identifier_name(base)
        fields = g.struct_member_types.get(source)
        if fields is None:
            if base.startswith(("mat", "dmat")):
                raise OpenGLFloatAtomicStorageError("matrix-storage-layout")
            return base
        if source in self.types:
            return self.types[source]["physical"]
        physical = self.identifier(("type", source), f"crossgl_words_{source}")
        encode = self.identifier(("encode", source), f"crossgl_encode_{source}")
        decode = self.identifier(("decode", source), f"crossgl_decode_{source}")
        self.types[source] = {"physical": physical, "encode": encode, "decode": decode}
        declarations = []
        for field, kind in fields.items():
            name = g.glsl_struct_member_name(source, field)
            declarations.append(
                "    "
                + format_c_style_array_declaration(self.physical_type(kind), name)
                + ";\n"
            )
        self.definitions.append(f"struct {physical} {{\n{''.join(declarations)}}};\n")
        value = self.identifier(("codec-local", "value"), "crossgl_storage_value")
        result = self.identifier(("codec-local", "result"), "crossgl_storage_result")
        for encoding, name, output, argument in (
            (True, encode, physical, base),
            (False, decode, base, physical),
        ):
            body = f"    {output} {result};\n"
            for field, kind in fields.items():
                member = g.glsl_struct_member_name(source, field)
                body += self.copy_value(
                    kind, f"{result}.{member}", f"{value}.{member}", encoding
                )
            self.definitions.append(
                f"{output} {name}({argument} {value}) {{\n{body}    return {result};\n}}\n"
            )
        return physical

    def copy_value(self, logical, destination, source, encoding, depth=0):
        logical = self.generator.map_type(logical)
        base, suffix = split_array_type_suffix(logical)
        if suffix:
            end = suffix.find("]")
            extent = suffix[1:end]
            if not extent.isdigit() or int(extent) < 1:
                raise OpenGLFloatAtomicStorageError("unresolved-array-storage-extent")
            index = self.identifier(
                ("codec-index", depth), f"crossgl_storage_index{depth}"
            )
            body = self.copy_value(
                base + suffix[end + 1 :],
                f"{destination}[{index}]",
                f"{source}[{index}]",
                encoding,
                depth + 1,
            )
            return f"    for (int {index} = 0; {index} < {extent}; ++{index}) {{\n{body}    }}\n"
        return f"    {destination} = {self.convert(logical, source, encoding)};\n"

    def convert(self, logical, expression, encoding):
        g = self.generator
        logical = g.map_type(logical)
        if logical in self.WORD_TYPES:
            function = "floatBitsToUint" if encoding else "uintBitsToFloat"
            return f"{function}({expression})"
        base, suffix = split_array_type_suffix(logical)
        if suffix:
            raise OpenGLFloatAtomicStorageError("whole-array-storage-transfer")
        source = g.glsl_source_type_identifier_name(base)
        if source in g.struct_member_types:
            self.physical_type(logical)
            function = self.types[source]["encode" if encoding else "decode"]
            return f"{function}({expression})"
        return expression

    def declaration(self, logical, name):
        if self.discover or name not in self.roots:
            return logical
        self.declared.add(name)
        return self.physical_type(logical)

    def take_definitions(self):
        result = "".join(self.definitions)
        self.definitions.clear()
        return result

    def root(self, expr):
        g = self.generator
        if isinstance(expr, PointerAccessNode):
            return self.root(expr.pointer_expr)
        if isinstance(expr, MemberAccessNode):
            return self.root(expr.object)
        if isinstance(expr, ArrayAccessNode):
            return self.root(expr.array)
        if isinstance(expr, UnaryOpNode) and expr.op == "*":
            return self.root(expr.operand)
        if (
            isinstance(expr, FunctionCallNode)
            and g.function_call_name(expr) == "buffer_load"
            and "buffer_load" not in g.function_return_types
        ):
            return self.root(expr.arguments[0])
        if not isinstance(expr, (IdentifierNode, VariableNode, str)):
            return None
        name = expr if isinstance(expr, str) else expr.name
        alias = g.current_identifier_aliases.get(
            name, g.glsl_module_identifier_name(name)
        )
        if (
            name in g.local_variable_types
            and alias not in g.glsl_hoisted_shared_alias_names
            and not g.is_structured_buffer_type(g.local_variable_types[name])
            and "*" not in g.type_name_string(g.local_variable_types[name])
        ):
            return None
        binding = g.glsl_workgroup_pointer_binding(
            expr, g.glsl_workgroup_pointer_aliases()
        )
        if binding is None:
            binding = g.glsl_storage_pointer_binding(
                expr, g.glsl_storage_pointer_aliases()
            )
        if binding is not None:
            if binding.get("pointer_reinterpretation"):
                raise OpenGLFloatAtomicStorageError("reinterpreted-storage-view", expr)
            return binding["root"]
        return alias

    def logical_access(self, expr):
        if self.discover or not self.roots or id(expr) in self.raw_nodes:
            return None
        if self.root(expr) not in self.roots:
            return None
        kind = self.generator.map_type(self.generator.expression_result_type(expr))
        if (
            kind
            and not self.generator.is_structured_buffer_type(kind)
            and not any(c in kind for c in "*&")
        ):
            return kind
        return None

    @contextmanager
    def raw(self, expr):
        saved = self.raw_nodes.copy()
        while expr is not None:
            self.raw_nodes.add(id(expr))
            if isinstance(expr, MemberAccessNode):
                expr = expr.object
            elif isinstance(expr, ArrayAccessNode):
                expr = expr.array
            elif isinstance(expr, PointerAccessNode):
                expr = expr.pointer_expr
            elif isinstance(expr, UnaryOpNode) and expr.op == "*":
                expr = expr.operand
            elif (
                isinstance(expr, FunctionCallNode)
                and self.generator.function_call_name(expr) == "buffer_load"
            ):
                expr = expr.arguments[0]
            else:
                break
        try:
            yield
        finally:
            self.raw_nodes = saved

    def read(self, expr, is_main):
        kind = self.logical_access(expr)
        if kind is None:
            return None
        with self.raw(expr):
            value = self.generator.generate_expression(expr, is_main)
        return self.convert(kind, value, False)

    def buffer_value(self, buffer, kind, value, encoding):
        if self.discover or self.root(buffer) not in self.roots:
            return value
        return self.convert(kind, value, encoding)

    def atomic(self, operation, args):
        g = self.generator
        if not args or g.map_type(g.expression_result_type(args[0])) != "float":
            return None
        if g.glsl_half_width(g.glsl_source_expression_type(args[0])) is not None:
            raise OpenGLFloatAtomicStorageError("float-atomic-payload-width", args[0])
        if operation not in self.OPERATIONS:
            raise OpenGLFloatAtomicStorageError(
                "unsupported-float-atomic-operation", args[0]
            )
        if len(args) != self.OPERATIONS[operation]:
            raise OpenGLFloatAtomicStorageError("atomic-argument-count", args[0])
        for builtin in (
            "floatBitsToUint",
            "uintBitsToFloat",
            "atomicOr",
            "atomicExchange",
            "atomicCompSwap",
        ):
            if builtin in g.function_return_types or builtin in g.global_variable_types:
                raise OpenGLFloatAtomicStorageError("target-builtin-shadowed", args[0])
        g.validate_glsl_storage_pointer_mutation_target(args[0])
        g.validate_glsl_buffer_block_member_access(args[0], "read_write")
        g.validate_glsl_buffer_block_atomic_value_arguments(
            operation, args, args[0], "float"
        )
        with self.raw(args[0]):
            storage = g.glsl_expected_compare_storage(args[0])
        root = self.root(args[0])
        if storage is None or root is None:
            raise OpenGLFloatAtomicStorageError(
                "unresolved-storage-allocation", args[0]
            )
        if operation == "atomicCompareExchangeWeak":
            if (
                g.map_type(g.expression_result_type(args[1])) != "float"
                or not g.glsl_reference_argument_is_lvalue(args[1])
                or self.root(args[1]) in self.roots | {root}
            ):
                raise OpenGLFloatAtomicStorageError(
                    "expected-value-requires-private-float", args[1]
                )
        if self.discover:
            self.roots.add(root)
            # Traverse values too: nested atomic expressions can select another allocation.
            values = [g.generate_expression(value) for value in args[1:]]
            return f"crossgl_pending_float_atomic({', '.join(storage['arguments'] + values)})"
        if root not in self.roots:
            raise OpenGLFloatAtomicStorageError(
                "storage-plan-changed-between-passes", args[0]
            )
        return self.helper_call(operation, storage, args[1:])

    def helper_call(self, operation, storage, values, kind="float"):
        g = self.generator
        key = operation, storage["target"], len(storage["arguments"]), kind
        if key not in self.helpers:
            self.helpers[key] = {
                **storage,
                "operation": operation,
                "kind": kind,
                "name": self.identifier(("helper", key), "crossgl_float_storage"),
            }
        arguments = storage["arguments"] + [
            g.generate_expression_with_expected(value, kind) for value in values
        ]
        return f"{self.helpers[key]['name']}({', '.join(arguments)})"

    def unary_update(self, expr, operator):
        kind = self.logical_access(expr.operand)
        if kind is None:
            return None
        if kind not in self.WORD_TYPES:
            with self.raw(expr.operand):
                return self.generator.generate_expression(expr)
        with self.raw(expr.operand):
            storage = self.generator.glsl_expected_compare_storage(expr.operand)
        if storage is None:
            raise OpenGLFloatAtomicStorageError("unresolved-update-storage", expr)
        postfix = bool(getattr(expr, "is_postfix", False)) or expr.op in {
            "POST_INCREMENT",
            "POST_DECREMENT",
        }
        return self.helper_call(
            ("post" if postfix else "pre") + operator, storage, [], kind
        )

    def validate_reference(self, expr):
        kind = self.logical_access(expr)
        if kind in self.WORD_TYPES or kind in self.generator.struct_member_types:
            raise OpenGLFloatAtomicStorageError(
                "logical-storage-reference-escape", expr
            )

    def assignment(self, node, left, right, operator, statement_context):
        g = self.generator
        kind = self.logical_access(left)
        if kind is None:
            return None
        if kind not in self.WORD_TYPES and kind not in g.struct_member_types:
            with self.raw(left):
                return g.generate_assignment(node, statement_context=statement_context)
        if operator != "=":
            right_type = g.glsl_value_type_info(g.glsl_source_expression_type(right))
            if (
                right_type
                and right_type["family"] == "float"
                and right_type["bits"] != 32
            ):
                raise OpenGLFloatAtomicStorageError(
                    "mixed-precision-storage-update", right
                )
            if kind not in self.WORD_TYPES or operator not in {"+=", "-=", "*=", "/="}:
                raise OpenGLFloatAtomicStorageError("storage-update-operator", left)
            with self.raw(left):
                storage = g.glsl_expected_compare_storage(left)
            if storage is None:
                raise OpenGLFloatAtomicStorageError("unresolved-update-storage", left)
            return self.helper_call(operator, storage, [right], kind)
        with self.raw(left):
            target = g.generate_glsl_buffer_block_mutation_target(left)
        value = g.generate_expression_with_expected(right, kind)
        result = f"{target} = {self.convert(kind, value, True)}"
        return result if statement_context else self.convert(kind, f"({result})", False)

    def helper_definitions(self):
        code = ""
        for key, helper in self.helpers.items():
            operation = helper["operation"]
            kind = helper["kind"]
            roles = [
                "expected",
                "desired",
                "observed",
                "matched",
                "compared",
                "result",
            ] + [f"index{i}" for i in range(len(helper["arguments"]))]
            names = {
                role: self.identifier(("local", key, role), f"crossgl_float_{role}")
                for role in roles
            }
            target, parameters = helper["target"], []
            for i in range(len(helper["arguments"])):
                name = names[f"index{i}"]
                target = target.replace(f"[index{i}]", f"[{name}]")
                parameters.append(f"uint {name}")
            compare = operation == "atomicCompareExchangeWeak"
            if operation in {"atomicCompSwap", "atomicCompareExchangeWeak"}:
                parameters.append(
                    f"{'inout ' if compare else ''}float {names['expected']}"
                )
            if operation != "atomicLoad" and not operation.startswith(("pre", "post")):
                parameters.append(f"{kind} {names['desired']}")
            observed, expected, desired = (
                names["observed"],
                names["expected"],
                names["desired"],
            )
            return_type = (
                "bool" if compare else "void" if operation == "atomicStore" else kind
            )
            if operation == "atomicLoad":
                body = f"return uintBitsToFloat(atomicOr({target}, 0u));"
            elif operation in {"atomicStore", "atomicExchange"}:
                exchange = f"atomicExchange({target}, floatBitsToUint({desired}))"
                body = (
                    f"{exchange};"
                    if operation == "atomicStore"
                    else f"return uintBitsToFloat({exchange});"
                )
            elif operation in {"atomicCompSwap", "atomicCompareExchangeWeak"}:
                body = f"uint {observed} = atomicCompSwap({target}, floatBitsToUint({expected}), floatBitsToUint({desired}));\n"
                if compare:
                    matched = names["matched"]
                    body += f"    bool {matched} = {observed} == floatBitsToUint({expected});\n    if (!{matched}) {expected} = uintBitsToFloat({observed});\n    return {matched};"
                else:
                    body += f"    return uintBitsToFloat({observed});"
            elif operation == "atomicAdd":
                compared = names["compared"]
                body = f"uint {observed} = atomicOr({target}, 0u);\n    while (true) {{\n        uint {compared} = {observed};\n        {observed} = atomicCompSwap({target}, {compared}, floatBitsToUint(uintBitsToFloat({compared}) + {desired}));\n        if ({observed} == {compared}) return uintBitsToFloat({observed});\n    }}"
            else:
                result = names["result"]
                if operation.startswith(("pre", "post")):
                    symbol = operation[-1]
                    body = f"{kind} {observed} = uintBitsToFloat({target});\n    {kind} {result} = {observed} {symbol} {kind}(1.0);\n    {target} = floatBitsToUint({result});\n    return {observed if operation.startswith('post') else result};"
                else:
                    body = f"{kind} {result} = uintBitsToFloat({target}) {operation[0]} {desired};\n    {target} = floatBitsToUint({result});\n    return {result};"
            code += f"{return_type} {helper['name']}({', '.join(parameters)}) {{\n    {body}\n}}\n\n"
        return code

    def verify(self):
        if self.roots != self.declared:
            raise OpenGLFloatAtomicStorageError(
                "unresolved-physical-storage-declaration"
            )
        if self.definitions:
            raise OpenGLFloatAtomicStorageError("unplanned-storage-conversion")
