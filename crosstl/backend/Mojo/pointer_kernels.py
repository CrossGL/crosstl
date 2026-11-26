"""Mojo GPU kernels that address memory through ``UnsafePointer`` parameters, imported as CrossGL compute entries.

A Mojo kernel is an ordinary ``def`` that ``DeviceContext.enqueue_function`` launches with a grid and block size the
host picks at dispatch. When its buffers are pointer parameters::

    def probe(fin: UnsafePointer[Float32, ImmutAnyOrigin], outp: UnsafePointer[Float32, MutAnyOrigin]):
        var i: Int = global_idx.x
        outp[i] = fin[i]

the CrossGL entry binds each pointer as a storage buffer, in parameter order, and takes its workgroup size from the
host, as the CUDA and HIP importers write a ``__global__`` kernel::

    @compute
    @host_workgroup_size
    fn probe(@group(0) @binding(0) var<storage, read> fin: array<float>,
             @group(0) @binding(1) var<storage, read_write> outp: array<float>) { ... }

An immutable origin (``ImmutAnyOrigin``, ``ImmutUntrackedOrigin``, ``StaticConstantOrigin``) makes the buffer
``read``; any other origin ``read_write``. ``max.gpu``'s ``global_idx`` is the global invocation index, ``grid_dim``
the number of workgroups (the Mojo standard library's ``gpu/id.mojo``). A pointer in another address space than
global memory, or a parameter that is neither a pointer nor a scalar or vector value, has no buffer form and is
refused by name.

A program whose kernels are all pointer kernels is written at the top level, as the CUDA importer writes one: the
CrossGL parser applies ``@host_workgroup_size`` to a top-level ``@compute fn`` and to the helpers it calls.
"""

from __future__ import annotations

__all__ = [
    "POINTER_KERNEL_BUILTINS",
    "is_pointer_kernel",
    "is_pointer_kernel_program",
    "pointer_kernel_header",
    "pointer_kernel_method_call",
    "refuse_parametric_call",
    "top_level_program",
    "unsafe_pointer_arguments",
]

# max.gpu's index values with the CrossGL built-in each one reads.
POINTER_KERNEL_BUILTINS = {
    "global_idx": "gl_GlobalInvocationID",
    "grid_dim": "gl_NumWorkGroups",
}

_POINTER = "UnsafePointer["
_IMMUTABLE_ORIGINS = ("Immut", "StaticConstantOrigin")
_GLOBAL_ADDRESS_SPACES = {"AddressSpace.GLOBAL", "AddressSpace.GENERIC"}


def _split_arguments(text: str) -> list[str]:
    """``text``'s top-level comma-separated arguments; brackets and parentheses keep their commas."""
    arguments, depth, start = [], 0, 0
    for index, char in enumerate(text):
        depth += {"[": 1, "(": 1, "]": -1, ")": -1}.get(char, 0)
        if char == "," and depth == 0:
            arguments.append(text[start:index].strip())
            start = index + 1
    arguments.append(text[start:].strip())
    return [argument for argument in arguments if argument]


def unsafe_pointer_arguments(vtype) -> tuple[str, str, str] | None:
    """``(element type, origin, address space)`` of the type ``UnsafePointer[T, origin, address_space=...]``, with an
    empty string for a part the type leaves out, or ``None`` when ``vtype`` is not an ``UnsafePointer``.
    """
    if not isinstance(vtype, str):
        return None
    text = vtype.strip()
    if not (text.startswith(_POINTER) and text.endswith("]")):
        return None
    element, origin, space = "", "", ""
    positional = []
    for argument in _split_arguments(text[len(_POINTER) : -1]):
        name, equals, value = argument.partition("=")
        if equals and name.strip().isidentifier():
            name, value = name.strip(), value.strip()
            if name == "address_space":
                space = value
            elif name == "origin":
                origin = value
            elif name == "mut":
                origin = origin or (
                    "MutAnyOrigin" if value == "True" else "ImmutAnyOrigin"
                )
            elif name in {"type", "T"}:
                element = value
            continue
        positional.append(argument)
    if positional and not element:
        element = positional.pop(0)
    if positional and not origin:
        origin = positional.pop(0)
    if positional and not space:
        space = positional.pop(0)
    return element, origin, space


def is_pointer_kernel(function) -> bool:
    """Whether ``function`` takes at least one ``UnsafePointer`` parameter."""
    return any(
        unsafe_pointer_arguments(getattr(parameter, "vtype", None)) is not None
        for parameter in getattr(function, "params", None) or []
    )


def is_pointer_kernel_program(ast, kernel_names) -> bool:
    """Whether ``ast`` launches kernels and every one of ``kernel_names`` is a pointer kernel."""
    functions = {
        getattr(node, "name", None): node
        for node in getattr(ast, "functions", None) or []
    }
    kernels = [functions.get(name) for name in sorted(kernel_names or ())]
    return bool(kernels) and all(
        kernel is not None and is_pointer_kernel(kernel) for kernel in kernels
    )


def pointer_kernel_method_call(generator, receiver, method, arguments):
    """The CrossGL form of a method call the importer maps to no CrossGL function, or a refusal.

    ``copy()`` copies a value, which every CrossGL assignment does. Any other method (``SIMD.cast[...]``, ``eq``,
    ``reduce_and``, ``Atomic.max``, a ``warp`` function the importer does not map, a structure's own method) would
    reach CrossGL as Mojo text, which no generator reads as Mojo does, so it is refused by name.
    """
    if method == "copy" and not arguments:
        return receiver
    return generator.refuse_import(
        f"the Mojo method call '{receiver}.{method}(...)' has no CrossGL form in the importer",
        placeholder="0",
    )


def refuse_parametric_call(generator, callee):
    """Refuse a call of a parameterized Mojo function or method (``stack_allocation[...]()``, ``x.cast[T]()``)."""
    return generator.refuse_import(
        f"the parameterized Mojo call '{callee}(...)' has no CrossGL form in the importer",
        placeholder="0",
    )


def _buffer_parameter(generator, parameter, binding):
    element, origin, space = unsafe_pointer_arguments(parameter.vtype)
    name = generator.map_identifier_name(parameter.name)
    if space and space not in _GLOBAL_ADDRESS_SPACES:
        return generator.refuse_import(
            f"kernel parameter '{parameter.name}' points into {space}; a CrossGL storage buffer is global memory",
            placeholder=name,
        )
    if not element:
        return generator.refuse_import(
            f"kernel parameter '{parameter.name}' is an UnsafePointer without an element type",
            placeholder=name,
        )
    access = "read" if origin.startswith(_IMMUTABLE_ORIGINS) else "read_write"
    element_type = generator.map_type(element)
    return f"@group(0) @binding({binding}) var<storage, {access}> {name}: array<{element_type}>"


def pointer_kernel_header(generator, function, indent: str) -> str:
    """The CrossGL compute entry that declares ``function``'s pointer parameters as storage buffers, up to its body's
    opening brace."""
    parameters, binding = [], 0
    for parameter in getattr(function, "params", None) or []:
        if generator.is_receiver_parameter(parameter) or not parameter.vtype:
            continue
        if unsafe_pointer_arguments(parameter.vtype) is not None:
            parameters.append(_buffer_parameter(generator, parameter, binding))
            binding += 1
            continue
        if getattr(parameter, "default_value", None) is not None:
            generator.refuse_import(
                f"kernel parameter '{parameter.name}' has a default value, and a CrossGL entry parameter has none",
                placeholder="",
            )
        parameters.append(
            f"{generator.map_type(parameter.vtype)} {generator.map_identifier_name(parameter.name)}"
        )
    if function.return_type and generator.map_return_type(function.return_type) not in {
        "void",
        "None",
    }:
        generator.refuse_import(
            f"kernel '{function.name}' returns a value, and a compute entry point returns none",
            placeholder="",
        )
    name = generator.map_identifier_name(function.name)
    return (
        f"{indent}@compute\n{indent}@host_workgroup_size\n"
        f"{indent}fn {name}({', '.join(parameters)}) {{\n"
    )


def top_level_program(code: str) -> str:
    """``code``, a ``shader main { ... }`` program, with its declarations moved out of the block to the top level."""
    lines = code.splitlines()
    if not lines or lines[0] != "shader main {" or lines[-1] != "}":
        return code
    body = [line[4:] if line.startswith("    ") else line for line in lines[1:-1]]
    return "\n".join(body).strip("\n") + "\n"
