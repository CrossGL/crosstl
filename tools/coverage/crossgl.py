"""How the coverage probes write CrossGL: the template layer.

The CrossGL IR has no compatibility guarantee. Every CrossGL probe is generated from this module and the expression
syntax :class:`~tools.coverage.dialect.Dialect` shares with GLSL, so a pull request that changes how CrossGL writes a
compute kernel, a storage buffer, a type or a built-in function updates this module too, and no probe text is pinned
anywhere else.
"""

from __future__ import annotations

from .dialect import Buffer, Dialect

# Built-in functions CrossGL spells as GLSL does rather than as the backend does. A rename applies only when the GLSL
# feature list has the new name; any other backend function is probed under its own spelling.
RENAMES = {
    "hlsl": {
        "lerp": "mix",
        "frac": "fract",
        "rsqrt": "inversesqrt",
        "mad": "fma",
        "countbits": "bitCount",
        "firstbithigh": "findMSB",
        "firstbitlow": "findLSB",
        "reversebits": "bitfieldReverse",
        "asint": "floatBitsToInt",
        "asuint": "floatBitsToUint",
        "atan2": "atan",
        "ddx": "dFdx",
        "ddy": "dFdy",
        "ddx_coarse": "dFdxCoarse",
        "ddx_fine": "dFdxFine",
        "ddy_coarse": "dFdyCoarse",
        "ddy_fine": "dFdyFine",
        "GroupMemoryBarrierWithGroupSync": "barrier",
        "GroupMemoryBarrier": "memoryBarrierShared",
        "AllMemoryBarrier": "memoryBarrier",
        "DeviceMemoryBarrier": "memoryBarrierBuffer",
        "InterlockedAdd": "atomicAdd",
        "InterlockedMin": "atomicMin",
        "InterlockedMax": "atomicMax",
        "InterlockedAnd": "atomicAnd",
        "InterlockedOr": "atomicOr",
        "InterlockedXor": "atomicXor",
        "WaveActiveAllTrue": "allInvocations",
        "WaveActiveAnyTrue": "anyInvocation",
        "WaveActiveAllEqual": "allInvocationsEqual",
    },
}


def spelling(language: str, name: str, glsl_names: set[str] | frozenset[str]) -> str:
    """The CrossGL name of ``language``'s built-in ``name``."""
    renamed = RENAMES.get(language, {}).get(name, name)
    return renamed if renamed in glsl_names else name


class CrossGL(Dialect):
    language = "crossgl"
    suffix = ".cgl"

    def kernel(self, buffers: list[Buffer], body: list[str]) -> str:
        lines = [
            "shader Probe {",
            "    compute {",
            "        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;",
        ]
        lines += [f"        buffer {kind}* {name};" for kind, name in buffers]
        lines += [
            "        void main() {",
            "            uint i = gl_GlobalInvocationID.x;",
        ]
        lines += [f"            {line}" for line in body]
        lines += ["        }", "    }", "}"]
        return "\n".join(lines) + "\n"


DIALECT = CrossGL()
