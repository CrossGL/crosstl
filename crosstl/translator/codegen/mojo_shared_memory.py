"""Mojo spelling of CrossGL workgroup-shared memory and of pointers into it.

A CrossGL ``shared`` (or ``groupshared``, ``threadgroup``, ``workgroup``) array is one array per thread block. In
Mojo it is ``stack_allocation[N, T, address_space=AddressSpace.SHARED]()``, called in the kernel, which returns a
pointer to the block's array; ``tile[i]`` reads and writes it as before. A two-dimensional ``T tile[A][B]`` is a
shared array of ``A`` rows of ``StaticTuple[T, B]``, so ``tile[i][j]`` still names one element. A helper function
that reads the array takes that pointer as an extra argument, as it takes a buffer. A shared scalar would need every
read rewritten to ``tile[0]`` and is refused.
"""

from ..ast import ArrayNode
from ..refusals import MojoRefusal

__all__ = ["MOJO_SHARED_ADDRESS_SPACES", "MojoSharedMemoryMixin"]

# The CrossGL address-space qualifiers of a pointee that name workgroup-shared memory.
MOJO_SHARED_ADDRESS_SPACES = frozenset(
    {"groupshared", "shared", "threadgroup", "workgroup"}
)
# The qualifiers that name memory a Mojo pointer with no address space reaches: device or generic memory.
MOJO_GENERIC_ADDRESS_SPACES = frozenset(
    {"device", "global", "constant", "thread", "private"}
)


def split_pointee_address_space(pointee):
    """``("threadgroup", "float")`` for the pointee ``threadgroup float``, ``(None, pointee)`` without one."""
    words = str(pointee).split(None, 1)
    if len(words) == 2 and words[0].lower() in (
        MOJO_SHARED_ADDRESS_SPACES | MOJO_GENERIC_ADDRESS_SPACES
    ):
        return words[0].lower(), words[1].strip()
    return None, str(pointee).strip()


class MojoSharedMemoryMixin:
    """Workgroup-shared memory members of ``MojoCodeGen``."""

    def mojo_pointer_type(self, pointee):
        """The Mojo pointer type for a CrossGL pointee, which may name its address space."""
        space, element_type = split_pointee_address_space(pointee)
        mapped = self.map_type(element_type)
        if space in MOJO_SHARED_ADDRESS_SPACES:
            self.requires_shared_memory_imports = True
            return (
                f"UnsafePointer[{mapped}, MutUntrackedOrigin, "
                "address_space=AddressSpace.SHARED]"
            )
        # Mojo 1.2 cannot infer a pointer's origin from a declaration; MutAnyOrigin is the origin a kernel's
        # buffer arguments carry and allows both reads and writes.
        return f"UnsafePointer[{mapped}, MutAnyOrigin]"

    def mojo_shared_declared_type(self, node):
        if isinstance(node, ArrayNode):
            return self.array_declared_type(node)
        return self.variable_declared_type(node)

    def mojo_shared_pointer_type_name(self, node):
        """The CrossGL pointer type of a shared array's Mojo pointer, ``threadgroup <element>*``."""
        name = getattr(node, "name", None)
        declared = self.mojo_shared_declared_type(node)
        if not declared or not self.is_array_type_name(declared):
            raise MojoRefusal(
                f"Mojo workgroup-shared scalar '{name}': only shared arrays have a Mojo spelling, a "
                "stack_allocation in AddressSpace.SHARED"
            )
        element_type, size = self.parse_array_type_name(declared)
        size = self.format_array_size(size)
        if size is None or getattr(node, "initial_value", None) is not None:
            raise MojoRefusal(
                f"Mojo workgroup-shared array '{name}' needs a fixed size and no initializer"
            )
        return f"threadgroup {element_type}*", element_type, size

    def generate_mojo_shared_declaration(self, node, indent):
        """``var tile = stack_allocation[N, T, address_space=AddressSpace.SHARED]()`` in the kernel."""
        pointer_type, element_type, size = self.mojo_shared_pointer_type_name(node)
        self.requires_shared_memory_imports = True
        name = getattr(node, "name", None)
        self.register_variable_type(name, pointer_type)
        return (
            f"{'    ' * indent}var {self.mojo_identifier(name)} = stack_allocation[{size}, "
            f"{self.map_type(element_type)}, address_space=AddressSpace.SHARED]()\n"
        )

    def mojo_stage_shared_arguments(self, stage):
        """The shared arrays of a stage, by name, as the pointer types helper functions take them as."""
        arguments = {}
        for node in getattr(stage, "local_variables", []) or []:
            if self.is_mojo_shared_variable(node):
                arguments[node.name] = self.mojo_shared_pointer_type_name(node)[0]
        return arguments
