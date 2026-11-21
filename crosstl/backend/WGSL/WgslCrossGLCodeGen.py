"""The WGSL importer: a WGSL module (W3C WGSL) written as a CrossGL program.

Module-scope declarations become CrossGL declarations of the shader block, in an order where every name is declared
before it is used (:mod:`.wgsl_module_context`): structures, constants and overrides (as specialization constants,
``const T name @constant_id(N) = value;``), resources with their ``@set``/``@binding``, workgroup variables as
``shared``, private variables, functions, then one stage block per entry point. Expressions are typed and written by
:mod:`.wgsl_expressions` and :mod:`.wgsl_builtin_calls`.

WGSL semantics that CrossGL does not give implicitly are written out:

* a ``var`` without an initializer holds the zero value of its type (section 7.3), and so does every workgroup
  variable when the workgroup starts (section 7.4); the importer initializes them, the workgroup variables from
  local invocation 0 before a ``barrier()`` at the start of each compute entry point that uses them;
* a ``loop`` with a ``continuing`` block (section 9.4.5) is a ``while (true)`` loop that runs the continuing
  statements and the ``break if`` test at the end of the body and before each ``continue``;
* a ``switch`` clause never falls through (section 9.4.3), so each ends with ``break``;
* a pointer is written as the memory it points to: a pointer parameter is an ``inout`` parameter, and a ``let``
  bound to a pointer names its memory, with every run-time index of it read once, where the ``let`` is.

A construct with no CrossGL form is refused with :class:`~.wgsl_types.WgslImportError`.
"""

from __future__ import annotations

from crosstl.translator.lexer import KEYWORDS as CROSSGL_KEYWORDS
from crosstl.translator.parsing.vocabulary import (
    RESOURCE_ADDRESS_SPACE_NAMES,
    VARIABLE_QUALIFIER_NAMES,
)

from . import WgslAst as A
from .wgsl_builtin_calls import zero_value
from .wgsl_expressions import ExpressionMixin, Symbol
from .wgsl_module_context import ModuleContext, names_in
from .wgsl_types import (
    BOOL,
    I32,
    U32,
    Array,
    Atomic,
    Pointer,
    ResultStruct,
    Sampler,
    Scalar,
    StructType,
    Texture,
    Vector,
    WgslImportError,
    concretize,
    crossgl_format,
    element_scalar,
    is_abstract,
)

__all__ = ["WgslToCrossGLConverter"]

_STAGES = ("compute", "vertex", "fragment")
_BUILTINS = {
    "vertex_index": "gl_VertexIndex",
    "instance_index": "gl_InstanceIndex",
    "front_facing": "gl_FrontFacing",
    "frag_depth": "gl_FragDepth",
    "sample_index": "gl_SampleID",
    "primitive_index": "gl_PrimitiveID",
    "clip_distances": "gl_ClipDistance",
    "local_invocation_id": "gl_LocalInvocationID",
    "local_invocation_index": "gl_LocalInvocationIndex",
    "global_invocation_id": "gl_GlobalInvocationID",
    "workgroup_id": "gl_WorkGroupID",
    "num_workgroups": "gl_NumWorkGroups",
    "subgroup_invocation_id": "gl_SubgroupInvocationID",
    "subgroup_size": "gl_SubgroupSize",
    "num_subgroups": "gl_NumSubgroups",
    "subgroup_id": "gl_SubgroupID",
}
_INTERPOLATION = {"flat": "flat", "linear": "noperspective", "perspective": None}
_SAMPLING = {
    "centroid": "centroid",
    "sample": "sample",
    "center": None,
    "first": None,
    "either": None,
}
_IGNORED_FUNCTION_ATTRIBUTES = frozenset({"must_use", "diagnostic"})
# Names CrossGL reads as keywords or declaration qualifiers, which a declaration of the module is renamed from.
_CROSSGL_RESERVED = (
    frozenset(CROSSGL_KEYWORDS)
    | VARIABLE_QUALIFIER_NAMES
    | RESOURCE_ADDRESS_SPACE_NAMES
)


def _unparenthesized(text: str) -> str:
    """``text`` without one pair of parentheses around all of it, for a condition CrossGL parenthesizes itself."""
    if not (text.startswith("(") and text.endswith(")")):
        return text
    depth = 0
    for index, char in enumerate(text):
        depth += {"(": 1, ")": -1}.get(char, 0)
        if depth == 0 and index < len(text) - 1:
            return text
    return text[1:-1]


def _declared_names(module) -> set:
    """Every name the module declares: at module scope, as structure members, parameters and locals."""
    names = set()

    def walk(node):
        if isinstance(
            node,
            (
                A.Local,
                A.Param,
                A.StructMember,
                A.Struct,
                A.Function,
                A.GlobalVar,
                A.Const,
                A.Override,
                A.Alias,
            ),
        ):
            names.add(node.name)
        if isinstance(node, list):
            for item in node:
                walk(item)
        elif hasattr(node, "__dataclass_fields__") and not isinstance(node, A.Ident):
            for field in node.__dataclass_fields__:
                walk(getattr(node, field))

    walk(module.declarations)
    return names


def _attribute(attributes, name):
    for attribute in attributes:
        if attribute.name == name:
            return attribute
    return None


class _FunctionState:
    def __init__(self, name, return_type, stage=None):
        self.name = name
        self.return_type = return_type
        self.stage = stage
        self.loops = []  # (continuing lines or None, switch depth at the loop)
        self.switch_depth = 0


class WgslToCrossGLConverter(ExpressionMixin):
    """Writes a parsed WGSL module (:class:`~.WgslAst.Module`) as CrossGL source text."""

    def __init__(self, **_options):
        self.ctx = None

    # Program.

    def generate(self, module: A.Module) -> str:
        self.ctx = ModuleContext(module)
        self.scopes = []
        self.helpers = {}
        self.helper_lines = []
        self.pending = None
        self.function = None
        self.taken = (
            names_in(module)
            | self.ctx.all_names()
            | {
                member.name
                for struct in self.ctx.structs.values()
                for member in struct.members
            }
        )
        self.renames = {}
        for name in sorted(_declared_names(module)):
            if name in _CROSSGL_RESERVED or name.startswith("gl_") or name == "main":
                self.renames[name] = self.fresh(f"{name}_")
        self.struct_names = self.renames
        self.module_types = {}
        self.spec_ids = self._assign_spec_ids()
        body = []
        body += self._structs()
        body += self._values()
        body += self._globals()
        functions, entries = [], []
        for function in self.ctx.ordered_functions():
            stage = self._stage(function)
            if stage and any(stage == other for _f, other in entries):
                raise WgslImportError(
                    f"a second {stage} entry point ('{function.name}') is not supported",
                    function.line,
                )
            (entries if stage else functions).append((function, stage))
        function_lines = []
        for function, _stage in functions:
            function_lines += self._function(function)
        entry_lines = []
        for function, stage in entries:
            entry_lines += self._entry(function, stage)
        lines = ["shader main {"]
        lines += [
            f"    {line}"
            for line in body + self.helper_lines + function_lines + entry_lines
        ]
        lines.append("}")
        return "\n".join(lines) + "\n"

    def name(self, name: str) -> str:
        """The CrossGL spelling of a name the module declares: a CrossGL keyword or ``gl_`` name is renamed."""
        if name in self.renames:
            return self.renames[name]
        return name

    def fresh(self, base: str) -> str:
        name, index = base, 0
        while name in self.taken:
            index += 1
            name = f"{base}_{index}"
        self.taken.add(name)
        return name

    # Names in scope.

    def lookup(self, name):
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return self._module_symbol(name)

    def declare(self, name, symbol):
        self.scopes[-1][name] = symbol

    def _module_symbol(self, name):
        ctx = self.ctx
        if name in ctx.consts or name in ctx.overrides:
            return Symbol("const", self._value_type(name), self.name(name), stable=True)
        if name in ctx.globals:
            node = ctx.globals[name]
            space = self._address_space(node)
            type_ = (
                ctx.resolve(node.type) if node.type else self._global_init_type(node)
            )
            return Symbol("global", type_, self.name(name), space=space)
        return None

    def _value_type(self, name):
        if name not in self.module_types:
            node = self.ctx.consts.get(name) or self.ctx.overrides.get(name)
            if node.type is not None:
                type_ = self.ctx.resolve(node.type)
            else:
                self.module_types[name] = None
                type_ = self.type_of(node.init)
                if name in self.ctx.overrides:
                    type_ = concretize(type_)
            self.module_types[name] = type_
        type_ = self.module_types[name]
        if type_ is None:
            raise WgslImportError(f"'{name}' is declared in terms of itself", None)
        return type_

    def _global_init_type(self, node):
        if node.init is None:
            raise WgslImportError(f"'{node.name}' has no type", node.line)
        return concretize(self.type_of(node.init))

    def member_name(self, struct, member):
        return self.name(member)

    # Module-scope declarations.

    def _structs(self):
        lines = []
        for node in self.ctx.ordered_structs():
            lines.append(f"struct {self.name(node.name)} {{")
            for name, type_, attributes, line in self.ctx.struct_fields(node.name):
                semantics = self._io_attributes(attributes, None, line, member=True)
                lines.append(
                    f"    {self.declarator(type_, self.name(name), line)}{semantics};"
                )
            lines.append("};")
        return lines

    def declarator(self, type_, name, line=None):
        from .wgsl_types import crossgl_declarator

        return crossgl_declarator(type_, name, line, self.struct_names)

    def _assign_spec_ids(self):
        ids, used = {}, set()
        for node in self.ctx.overrides.values():
            attribute = _attribute(node.attributes, "id")
            if attribute is not None:
                value = self.ctx.eval_const(attribute.args[0])
                ids[node.name] = value
                used.add(value)
        next_id = 0
        for node in self.ctx.overrides.values():
            if node.name in ids or self._derived_override(node):
                continue
            while next_id in used:
                next_id += 1
            ids[node.name] = next_id
            used.add(next_id)
        return ids

    def _derived_override(self, node):
        """An override whose default names other overrides: CrossGL writes it as a constant expression of them."""
        return node.init is not None and bool(
            names_in(node.init) & set(self.ctx.overrides)
        )

    def _values(self):
        lines = []
        self.scopes = [{}]
        for node in self.ctx.ordered_values():
            type_ = concretize(self._value_type(node.name))
            declarator = self.declarator(type_, self.name(node.name), node.line)
            if isinstance(node, A.Override) and not self._derived_override(node):
                value = (
                    self.emit_value(node.init, type_)
                    if node.init
                    else zero_value(self, type_, node.line)
                )
                lines.append(
                    f"const {declarator} @constant_id({self.spec_ids[node.name]}) = {value};"
                )
            else:
                lines.append(
                    f"const {declarator} = {self.emit_value(node.init, type_)};"
                )
        self.scopes = []
        return lines

    @staticmethod
    def _address_space(node):
        if node.template_args:
            return node.template_args[0].name
        return "handle"

    def _globals(self):
        lines = []
        self.scopes = [{}]
        for node in self.ctx.globals.values():
            lines.append(self._global(node))
        self.scopes = []
        return lines

    def _binding(self, node):
        group, binding = _attribute(node.attributes, "group"), _attribute(
            node.attributes, "binding"
        )
        if group is None or binding is None:
            raise WgslImportError(
                f"resource '{node.name}' needs @group and @binding", node.line
            )
        values = [
            self.ctx.eval_const(group.args[0]),
            self.ctx.eval_const(binding.args[0]),
        ]
        return f" @set({values[0]}) @binding({values[1]})"

    def _global(self, node):
        space, line = self._address_space(node), node.line
        type_ = self.lookup(node.name).type
        if space == "storage":
            access = (
                node.template_args[1].name if len(node.template_args) > 1 else "read"
            )
            if not isinstance(type_, Array) or type_.count is not None:
                raise WgslImportError(
                    f"storage buffer '{node.name}' must be a runtime-sized array", line
                )
            kind = (
                "RWStructuredBuffer" if access == "read_write" else "StructuredBuffer"
            )
            element = self.type_expression(type_.elem, line)
            return f"{kind}<{element}> {self.name(node.name)}{self._binding(node)};"
        name = self.name(node.name)
        if space == "uniform":
            return f"uniform {self.declarator(type_, name, line)}{self._binding(node)};"
        if space == "workgroup":
            return f"shared {self.declarator(type_, name, line)};"
        if space == "private":
            value = (
                self.emit_value(node.init, type_)
                if node.init is not None
                else zero_value(self, type_, line)
            )
            return f"{self.declarator(type_, name, line)} = {value};"
        if space == "handle":
            return self._handle(node, type_)
        raise WgslImportError(f"address space '{space}' is not supported", line)

    def _handle(self, node, type_):
        line = node.line
        if isinstance(type_, Texture) and type_.is_storage:
            access = {"write": "write", "read": "read", "read_write": "read_write"}[
                type_.access
            ]
            qualifiers = (
                f" @{crossgl_format(type_.texel_format, line)} @access({access})"
            )
            return f"{self.type_text(type_, line)} {self.name(node.name)}{qualifiers}{self._binding(node)};"
        if isinstance(type_, (Texture, Sampler)):
            return f"{self.type_text(type_, line)} {self.name(node.name)}{self._binding(node)};"
        raise WgslImportError(f"'{node.name}' needs an address space", line)

    # Functions.

    @staticmethod
    def _stage(function):
        for attribute in function.attributes:
            if attribute.name in _STAGES:
                return attribute.name
        return None

    def _signature_type(self, node):
        type_ = self.ctx.resolve(node)
        return type_

    def _function(self, function):
        params, symbols = [], {}
        for param in function.params:
            type_ = self._signature_type(param.type)
            if isinstance(type_, Pointer):
                if isinstance(type_.elem, Atomic):
                    raise WgslImportError(
                        "a pointer to an atomic parameter is not supported", param.line
                    )
                name = self.name(param.name)
                params.append(f"inout {self.declarator(type_.elem, name, param.line)}")
                symbols[param.name] = Symbol("param", type_, name, space=type_.space)
            else:
                name = self.name(param.name)
                params.append(self.declarator(type_, name, param.line))
                symbols[param.name] = Symbol("param", type_, name, stable=True)
        return_type = (
            self._signature_type(function.return_type) if function.return_type else None
        )
        returned = (
            self.type_expression(return_type, function.line) if return_type else "void"
        )
        header = f"{returned} {self.name(function.name)}({', '.join(params)}) {{"
        body = self._function_body(function, symbols, return_type)
        return [header, *body, "}", ""]

    def _function_body(self, function, symbols, return_type, stage=None, prologue=()):
        self.function = _FunctionState(function.name, return_type, stage)
        self.scopes = [symbols]
        lines = list(prologue)
        lines += self._statements(function.body.statements)
        self.scopes = []
        self.function = None
        return [f"    {line}" for line in lines]

    def _entry(self, function, stage):
        params, symbols = [], {}
        for param in function.params:
            type_ = self._signature_type(param.type)
            semantics = self._io_attributes(
                param.attributes, stage, param.line, direction="in"
            )
            name = self.name(param.name)
            params.append(f"{self.declarator(type_, name, param.line)}{semantics}")
            symbols[param.name] = Symbol("param", type_, name, stable=True)
        return_type = (
            self._signature_type(function.return_type) if function.return_type else None
        )
        returned = (
            self.type_expression(return_type, function.line) if return_type else "void"
        )
        return_semantics = (
            self._io_attributes(
                function.return_attributes, stage, function.line, direction="out"
            )
            if return_type
            else ""
        )
        for attribute in function.attributes:
            if attribute.name not in (
                *_STAGES,
                "workgroup_size",
                *_IGNORED_FUNCTION_ATTRIBUTES,
            ):
                raise WgslImportError(
                    f"entry point attribute @{attribute.name} is not supported",
                    function.line,
                )
        lines = [f"{stage} {{"]
        prologue = []
        if stage == "compute":
            lines.append(f"    {self._workgroup_size(function)}")
            prologue = self._workgroup_zero_init(function)
        # CrossGL's entry point is the function named main in its stage block.
        header = f"{returned} main({', '.join(params)}){return_semantics} {{"
        body = self._function_body(function, symbols, return_type, stage, prologue)
        lines += [f"    {header}", *[f"    {line}" for line in body], "    }", "}", ""]
        return lines

    def _workgroup_size(self, function):
        attribute = _attribute(function.attributes, "workgroup_size")
        if attribute is None:
            raise WgslImportError(
                "a compute entry point needs @workgroup_size", function.line
            )
        parts = []
        for axis, index in (("x", 0), ("y", 1), ("z", 2)):
            if index >= len(attribute.args):
                parts.append(f"local_size_{axis} = 1")
                continue
            arg = attribute.args[index]
            value = self.ctx.eval_const(arg)
            if isinstance(value, int):
                parts.append(f"local_size_{axis} = {value}")
            elif isinstance(arg, A.Ident) and arg.name in self.spec_ids:
                parts.append(f"local_size_{axis}_id = {self.spec_ids[arg.name]}")
            else:
                raise WgslImportError(
                    "@workgroup_size must be constants or overrides", function.line
                )
        return f"layout({', '.join(parts)}) in;"

    def _reachable_names(self, function, seen=None):
        """The module-scope names ``function`` and the functions it calls name."""
        seen = set() if seen is None else seen
        names = set()
        for name in names_in(function.body) | names_in(function.params):
            names.add(name)
            if name in self.ctx.functions and name not in seen:
                seen.add(name)
                names |= self._reachable_names(self.ctx.functions[name], seen)
        return names

    def _workgroup_zero_init(self, function):
        used = self._reachable_names(function)
        variables = [
            node
            for node in self.ctx.globals.values()
            if node.name in used and self._address_space(node) == "workgroup"
        ]
        if not variables:
            return []
        self.scopes = [{}]
        lines = ["if (gl_LocalInvocationIndex == 0u) {"]
        for node in variables:
            symbol = self.lookup(node.name)
            lines += [
                f"    {line}"
                for line in self._zero_store(symbol.text, symbol.type, node.line)
            ]
        lines += ["}", "barrier();"]
        self.scopes = []
        return lines

    def _zero_store(self, target, type_, line):
        """Statements that store the zero value of ``type_`` to ``target``."""
        if isinstance(type_, Array) and (type_.count is None or type_.count > 4):
            index = self.fresh("zero_index")
            count = type_.count if type_.count is not None else type_.count_text
            if count is None:
                raise WgslImportError(
                    "a runtime-sized workgroup array is not supported", line
                )
            inner = self._zero_store(f"{target}[{index}]", type_.elem, line)
            return [
                f"for (int {index} = 0; {index} < int({count}); {index}++) {{",
                *[f"    {s}" for s in inner],
                "}",
            ]
        return [f"{target} = {zero_value(self, type_, line)};"]

    def _io_attributes(self, attributes, stage, line, direction=None, member=False):
        parts = []
        for attribute in attributes:
            name = attribute.name
            if name == "location":
                parts.append(f"@location({self.ctx.eval_const(attribute.args[0])})")
            elif name == "builtin":
                parts.append(
                    f"@ {self._builtin_semantic(attribute.args[0].name, stage, direction, line)}"
                )
            elif name == "interpolate":
                parts += self._interpolation(attribute, line)
            elif name == "invariant":
                parts.append("@ invariant")
            elif name in ("align", "size") and member:
                raise WgslImportError(
                    f"structure member attribute @{name} is not supported", line
                )
            elif name == "blend_src":
                raise WgslImportError(
                    "@blend_src (dual-source blending) is not supported", line
                )
            else:
                raise WgslImportError(f"attribute @{name} is not supported here", line)
        return "".join(f" {part}" for part in parts)

    @staticmethod
    def _builtin_semantic(name, stage, direction, line):
        if name == "position":
            return (
                "gl_FragCoord"
                if stage == "fragment" and direction == "in"
                else "gl_Position"
            )
        if name == "sample_mask":
            return "gl_SampleMaskIn" if direction == "in" else "gl_SampleMask"
        if name not in _BUILTINS:
            raise WgslImportError(f"built-in value '{name}' is not supported", line)
        return _BUILTINS[name]

    @staticmethod
    def _interpolation(attribute, line):
        names = [arg.name for arg in attribute.args]
        kind = _INTERPOLATION.get(names[0], "") if names else ""
        if kind == "" or (len(names) > 1 and names[1] not in _SAMPLING):
            raise WgslImportError(
                f"@interpolate({', '.join(names)}) is not supported", line
            )
        parts = [f"@ {kind}"] if kind else []
        if len(names) > 1 and _SAMPLING[names[1]]:
            parts.append(f"@ {_SAMPLING[names[1]]}")
        return parts

    # Statements.

    def _statements(self, statements):
        self.scopes.append({})
        lines = []
        for statement in statements:
            lines += self._statement(statement)
        self.scopes.pop()
        return lines

    def _block(self, block, prefix="", extra=()):
        inner = self._statements(block.statements)
        return [
            f"{prefix}{{",
            *[f"    {line}" for line in inner],
            *[f"    {line}" for line in extra],
            "}",
        ]

    def _statement(self, node):
        method = getattr(self, f"_statement_{type(node).__name__}", None)
        if method is None:
            raise WgslImportError(
                f"statement {type(node).__name__} is not supported",
                getattr(node, "line", None),
            )
        saved, self.pending = self.pending, []
        try:
            lines = method(node)
            return [*self.pending, *lines]
        finally:
            self.pending = saved

    def hoist(self, type_, text, line=None):
        """A fresh local holding ``text``, declared before the current statement; its name."""
        if self.pending is None:
            raise WgslImportError(
                "this expression needs a temporary where none can be declared", line
            )
        name = self.fresh("tmp")
        self.pending.append(f"{self.declarator(type_, name, line)} = {text};")
        return name

    def hoist_uniform_load(self, node, result):
        """``workgroupUniformLoad(p)``: the load between two workgroup barriers (section 17.12.2)."""
        if self.pending is None:
            raise WgslImportError(
                "workgroupUniformLoad is not supported here", node.line
            )
        value = self.emit(node.args[0])
        self.pending.append("barrier();")
        name = self.hoist(result, value, node.line)
        self.pending.append("barrier();")
        return name

    def result_struct_call(self, node, result):
        raise WgslImportError(f"{node.callee.name} is not supported", node.line)

    def result_member(self, node, base_type):
        raise WgslImportError(
            f"the result of {base_type.kind} is not supported", node.line
        )

    def _statement_Block(self, node):
        return self._block(node)

    def _statement_Local(self, node):
        line = node.line
        if node.kind == "var" and node.template_args:
            if node.template_args[0].name != "function":
                raise WgslImportError(
                    "a function-scope var must be in the function address space", line
                )
        if node.type is not None:
            type_ = self.ctx.resolve(node.type)
        else:
            type_ = concretize(self.type_of(node.init))
        if isinstance(type_, Pointer):
            if node.kind != "let":
                raise WgslImportError("only a let can hold a pointer", line)
            self.declare(
                node.name,
                Symbol(
                    "alias", type_, self.reference_text(node.init), space=type_.space
                ),
            )
            return []
        name = self.name(node.name)
        value = (
            self.emit_value(node.init, type_)
            if node.init is not None
            else zero_value(self, type_, line)
        )
        self.declare(
            node.name,
            Symbol(node.kind, type_, name, space="function", stable=node.kind != "var"),
        )
        prefix = "const " if node.kind == "const" else ""
        return [f"{prefix}{self.declarator(type_, name, line)} = {value};"]

    def reference_text(self, node):
        """The memory a pointer expression points to, with each run-time index read now (see the module)."""
        if isinstance(node, A.Unary) and node.op == "&":
            return self._reference(node.operand)
        if isinstance(node, A.Ident):
            return self.lookup(node.name).text
        return self._reference(node)

    def _reference(self, node):
        if isinstance(node, A.Unary) and node.op == "*":
            return self.reference_text(node.operand)
        if isinstance(node, A.Ident):
            return self.emit(node)
        if isinstance(node, A.Member):
            base_type = self.deref(self.type_of(node.base))
            base = self._reference(node.base)
            if isinstance(base_type, StructType):
                return f"{base}.{self.member_name(base_type.name, node.name)}"
            return f"{base}.{node.name}"
        if isinstance(node, A.Index):
            base = self._reference(node.base)
            index_type = concretize(self.type_of(node.index))
            index = self.emit(node.index, index_type)
            if not self.is_stable(node.index):
                index = self.hoist(index_type, index, node.line)
            return f"{base}[{index}]"
        raise WgslImportError(
            "unsupported pointer expression", getattr(node, "line", None)
        )

    def _statement_Assign(self, node):
        if node.lhs is None:
            if self.has_side_effects(node.rhs):
                return [f"{self.emit(node.rhs)};"]
            return []
        target_type = self.deref(self.type_of(node.lhs))
        target = self.emit(node.lhs)
        if isinstance(target_type, Pointer):
            target_type = target_type.elem
        if node.op == "=":
            return [f"{target} = {self.emit_value(node.rhs, target_type)};"]
        op = node.op[:-1]
        rhs_type = self.deref(self.type_of(node.rhs))
        want = None
        if is_abstract(rhs_type):
            from .wgsl_types import with_scalar

            want = with_scalar(
                rhs_type, U32 if op in ("<<", ">>") else element_scalar(target_type)
            )
        rhs = self.emit(node.rhs, want)
        if op in ("/", "%") and element_scalar(target_type) in (I32, U32):
            divisor = self.ctx.eval_const(node.rhs)
            scalar = element_scalar(target_type)
            safe = (
                isinstance(divisor, int)
                and divisor != 0
                and (scalar == U32 or divisor != -1)
            )
            if safe and not isinstance(rhs_type, Vector):
                return [f"{target} {node.op} {rhs};"]
            if not self.is_simple_reference(node.lhs):
                raise WgslImportError(
                    f"'{node.op}' on this left-hand side is not supported", node.line
                )
            value = self.emit(A.Binary(op, node.lhs, node.rhs, node.line), target_type)
            return [f"{target} = {value};"]
        if op in ("<<", ">>"):
            rhs = self._masked_shift_amount(
                node.rhs, rhs, target_type, concretize(want or rhs_type)
            )
        return [f"{target} {node.op} {rhs};"]

    def is_simple_reference(self, node):
        if isinstance(node, A.Ident):
            return True
        if isinstance(node, A.Member):
            return self.is_simple_reference(node.base)
        if isinstance(node, A.Index):
            return self.is_simple_reference(node.base) and self.is_stable(node.index)
        if isinstance(node, A.Unary) and node.op == "*":
            return self.is_simple_reference(node.operand)
        return False

    def has_side_effects(self, node):
        if isinstance(node, A.Call):
            return True
        if isinstance(node, list):
            return any(self.has_side_effects(item) for item in node)
        if hasattr(node, "__dataclass_fields__"):
            return any(
                self.has_side_effects(getattr(node, name))
                for name in node.__dataclass_fields__
            )
        return False

    def _statement_Increment(self, node):
        return [f"{self.emit(node.target)}{node.op};"]

    def _statement_CallStatement(self, node):
        return [f"{self.emit(node.call)};"]

    def emit_user_call(self, node):
        function = self.ctx.functions[node.callee.name]
        args = []
        for arg, param in zip(node.args, function.params):
            type_ = self.ctx.resolve(param.type)
            if isinstance(type_, Pointer):
                args.append(self.reference_text(arg))
            else:
                args.append(self.emit_value(arg, type_))
        return f"{self.name(node.callee.name)}({', '.join(args)})"

    def _statement_If(self, node):
        condition = _unparenthesized(self.emit(node.condition))
        lines = self._block(node.then, f"if ({condition}) ")
        otherwise = node.otherwise
        if otherwise is None:
            return lines
        if isinstance(otherwise, A.If):
            nested = self._statement(otherwise)
            if len(nested) and nested[0].startswith("if ("):
                lines[-1] = f"}} else {nested[0]}"
                return lines + nested[1:]
            return lines[:-1] + ["} else {", *[f"    {line}" for line in nested], "}"]
        else_lines = self._block(otherwise)
        lines[-1] = "} else {"
        return lines + else_lines[1:]

    def _statement_While(self, node):
        condition = self._condition(node.condition)
        self.function.loops.append((None, self.function.switch_depth))
        try:
            return self._block(node.body, f"while ({condition}) ")
        finally:
            self.function.loops.pop()

    def _condition(self, node):
        saved, self.pending = self.pending, None
        try:
            return _unparenthesized(self.emit(node))
        finally:
            self.pending = saved

    def _statement_For(self, node):
        self.scopes.append({})
        try:
            init = ""
            if node.init is not None:
                init_lines = self._statement(node.init)
                if len(init_lines) != 1:
                    raise WgslImportError(
                        "this for-loop initializer is not supported", node.line
                    )
                init = init_lines[0].rstrip(";")
            condition = (
                self._condition(node.condition) if node.condition is not None else ""
            )
            update = ""
            if node.update is not None:
                saved, self.pending = self.pending, None
                try:
                    update_lines = getattr(
                        self, f"_statement_{type(node.update).__name__}"
                    )(node.update)
                finally:
                    self.pending = saved
                if len(update_lines) != 1:
                    raise WgslImportError(
                        "this for-loop update is not supported", node.line
                    )
                update = update_lines[0].rstrip(";")
            self.function.loops.append((None, self.function.switch_depth))
            try:
                return self._block(node.body, f"for ({init}; {condition}; {update}) ")
            finally:
                self.function.loops.pop()
        finally:
            self.scopes.pop()

    def _statement_Loop(self, node):
        marker = (
            node if (node.continuing is not None or node.break_if is not None) else None
        )
        self.function.loops.append((marker, self.function.switch_depth))
        self.scopes.append({})
        try:
            lines = []
            for statement in node.body.statements:
                lines += self._statement(statement)
            if marker is not None:
                lines += self._expand_continuing(marker)
        finally:
            self.scopes.pop()
            self.function.loops.pop()
        return ["while (true) {", *[f"    {line}" for line in lines], "}"]

    def _expand_continuing(self, node):
        """The continuing statements and the ``break if`` test of ``node``.

        WGSL lets the continuing block name the body's declarations (section 9.4.5), so it is written in the body's
        scope: after the body, and before each ``continue`` (:meth:`_statement_Continue`).
        """
        lines = []
        if node.continuing is not None:
            lines += self._statements(node.continuing.statements)
        if node.break_if is not None:
            condition = self._condition(node.break_if.condition)
            lines += [f"if ({condition}) {{", "    break;", "}"]
        return lines

    def _statement_Continue(self, node):
        if not self.function.loops:
            raise WgslImportError("continue outside a loop", node.line)
        marker, depth = self.function.loops[-1]
        if marker is None:
            return ["continue;"]
        if self.function.switch_depth > depth and marker.break_if is not None:
            raise WgslImportError(
                "continue inside a switch of a loop with 'break if' is not supported",
                node.line,
            )
        expanded = self._expand_continuing(marker)
        return ["{", *[f"    {line}" for line in expanded], "    continue;", "}"]

    def _statement_Break(self, node):
        return ["break;"]

    def _statement_Return(self, node):
        if node.value is None:
            return ["return;"]
        return [f"return {self.emit_value(node.value, self.function.return_type)};"]

    def _statement_Discard(self, node):
        return ["discard;"]

    def _statement_ConstAssert(self, node):
        return []

    def _statement_Switch(self, node):
        selector_type = concretize(self.type_of(node.selector))
        for clause in node.clauses:
            for selector in clause.selectors:
                if selector != "default" and not is_abstract(self.type_of(selector)):
                    selector_type = concretize(self.type_of(selector))
        selector = _unparenthesized(self.emit(node.selector, selector_type))
        lines = [f"switch ({selector}) {{"]
        self.function.switch_depth += 1
        try:
            for clause in node.clauses:
                labels = []
                for value in clause.selectors:
                    if value == "default":
                        labels.append("default:")
                    else:
                        labels.append(f"case {self.emit(value, selector_type)}:")
                lines.append(f"    {' '.join(labels)}")
                body = self._block(clause.body, "", ("break;",))
                lines += [f"    {line}" for line in body]
        finally:
            self.function.switch_depth -= 1
        lines.append("}")
        return lines

    # Helpers for integer division.

    def require_helper(self, kind, type_):
        key = (kind, type_)
        if key in self.helpers:
            return self.helpers[key]
        scalar = element_scalar(type_)
        name = self.fresh(
            f"wgsl_{kind}_{'u' if scalar == U32 else 'i'}{type_.size if isinstance(type_, Vector) else ''}"
        )
        self.helpers[key] = name
        if isinstance(type_, Vector):
            scalar_helper = self.require_helper(kind, scalar)
            vector = self.type_text(type_)
            parts = ", ".join(
                f"{scalar_helper}(a.{c}, b.{c})" for c in "xyzw"[: type_.size]
            )
            self.helper_lines += [
                f"{vector} {name}({vector} a, {vector} b) {{",
                f"    return {vector}({parts});",
                "}",
                "",
            ]
            return name
        text = self.type_text(scalar)
        if scalar == U32:
            special = "b == 0u"
        else:
            special = "b == 0 || (a == (-2147483647 - 1) && b == -1)"
        if kind == "div":
            body = f"return ({special}) ? a : (a / b);"
        else:
            body = f"return ({special}) ? {'0u' if scalar == U32 else '0'} : (a % b);"
        self.helper_lines += [
            f"{text} {name}({text} a, {text} b) {{",
            f"    {body}",
            "}",
            "",
        ]
        return name


__all__ += ["BOOL", "ResultStruct", "Scalar"]
