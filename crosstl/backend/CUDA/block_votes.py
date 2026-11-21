"""CUDA and HIP block-wide barrier votes: ``__syncthreads_count``, ``__syncthreads_and`` and ``__syncthreads_or``.

Each vote is ``__syncthreads()`` that also reduces an ``int`` predicate over the threads of the block and returns
the result to every thread: the number of threads whose predicate is non-zero, or whether it is non-zero for all
of them or for any of them (CUDA C++ Programming Guide, "Synchronization Functions"; HIP mirrors CUDA). CrossGL has
no barrier that reduces a value, so the importers count with one workgroup variable:

1. the barrier ``__syncthreads()`` imports as, which orders global and shared memory like the vote does and also
   lets every thread finish reading the previous vote's count;
2. the thread with local invocation index 0 clears the counter, and a workgroup barrier publishes it;
3. each thread whose predicate holds adds 1 with an atomic, and a workgroup barrier ends the count;
4. every thread reads the count.

``and`` counts the threads whose predicate is zero and compares the count with 0, and ``or`` compares the count of
the others with 0, so neither depends on the block size. The call passes ``gl_LocalInvocationIndex`` to the
function, because some targets (WGSL) read a built-in only in the entry point and its callers pass it on. Like ``__syncthreads()``, a vote must be reached by every
thread of the block; the counter is read only between the barriers of one vote.
"""

from __future__ import annotations

from ...translator.semantics.warp import ORDERED_WORKGROUP_BARRIER

BLOCK_VOTES = ("__syncthreads_count", "__syncthreads_and", "__syncthreads_or")


class BlockVoteLibrary:
    """The CrossGL calls of the block votes of one program, and the counter function they share."""

    def __init__(self, used_names):
        self.used_names = set(used_names)
        self.function = None
        self.counter = None
        self.parameter = None
        self.index_parameter = None

    def handles(self, name):
        return str(name or "").lstrip(":") in BLOCK_VOTES

    def _unused(self, base):
        candidate = base
        while candidate in self.used_names:
            candidate = f"{candidate}_"
        self.used_names.add(candidate)
        return candidate

    def call(self, name, predicate):
        """The CrossGL ``i32`` expression of ``name(predicate)``, where ``predicate`` is a CrossGL ``bool``."""
        if self.function is None:
            self.function = self._unused("crossgl_block_count")
            self.counter = self._unused("crossgl_block_votes")
            self.parameter = self._unused("vote")
            self.index_parameter = self._unused("thread_index")
        name = str(name).lstrip(":")
        if name == "__syncthreads_count":
            return f"i32({self.function}({predicate}, gl_LocalInvocationIndex))"
        if name == "__syncthreads_and":
            return (
                f"i32({self.function}(!({predicate}), gl_LocalInvocationIndex) == 0u)"
            )
        return f"i32({self.function}({predicate}, gl_LocalInvocationIndex) != 0u)"

    def render(self):
        """The CrossGL text of the counter and its function, or ``""`` when the program has no vote."""
        if self.function is None:
            return ""
        ordered_barrier = "\n".join(
            f"    {function}();" for function in ORDERED_WORKGROUP_BARRIER
        )
        return (
            f"var<workgroup> {self.counter}: u32;\n"
            "\n"
            f"u32 {self.function}(bool {self.parameter}, u32 {self.index_parameter}) {{\n"
            f"{ordered_barrier}\n"
            f"    if ({self.index_parameter} == 0u) {{\n"
            f"        {self.counter} = 0u;\n"
            "    }\n"
            "    workgroupBarrier();\n"
            f"    if ({self.parameter}) {{\n"
            f"        atomicAdd({self.counter}, 1u);\n"
            "    }\n"
            "    workgroupBarrier();\n"
            f"    return {self.counter};\n"
            "}\n"
        )


def insert_block_vote_functions(output, library, header):
    """Insert the counter function ``library`` needs into ``output`` lines, after the ``header`` line."""
    if library is None:
        return
    text = library.render()
    if not text:
        return
    position = output.index(header) + 1 if header in output else 0
    output[position:position] = [text]
