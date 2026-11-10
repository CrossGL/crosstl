"""A minimal SPIR-V grammar that holds exactly the items the language-level features anchor to."""

import json

from tools.coverage import constructs
from tools.coverage.model import Feature

_MODELS = [
    ("Vertex", ["Shader"]),
    ("TessellationControl", ["Tessellation"]),
    ("TessellationEvaluation", ["Tessellation"]),
    ("Geometry", ["Geometry"]),
    ("Fragment", ["Shader"]),
    ("GLCompute", ["Shader"]),
    ("Kernel", ["Kernel"]),
    ("TaskNV", ["MeshShadingNV"]),
    ("TaskEXT", ["MeshShadingEXT"]),
    ("MeshEXT", ["MeshShadingEXT"]),
    ("RayGenerationKHR", ["RayTracingNV", "RayTracingKHR"]),
    ("IntersectionKHR", ["RayTracingNV", "RayTracingKHR"]),
    ("AnyHitKHR", ["RayTracingNV", "RayTracingKHR"]),
    ("ClosestHitKHR", ["RayTracingNV", "RayTracingKHR"]),
    ("MissKHR", ["RayTracingNV", "RayTracingKHR"]),
    ("CallableKHR", ["RayTracingNV", "RayTracingKHR"]),
]


def grammar(drop: str = "") -> str:
    """A grammar with exactly the items the constructs anchor to, less ``drop``."""
    anchors = {a for c in constructs.CURATED for a in c.anchors}
    anchors |= {f"Capability.{name}" for name in constructs._WIDTHS}
    kinds: dict[str, list[dict]] = {
        "Capability": [],
        "StorageClass": [],
        "BuiltIn": [],
        "Decoration": [],
        "ExecutionModel": [],
        "ExecutionMode": [],
        "Dim": [],
        "ImageFormat": [],
    }
    instructions = []
    for anchor in sorted(anchors - {drop}):
        kind, _, name = anchor.rpartition(".")
        if kind:
            kinds[kind].append({"enumerant": name})
        else:
            instructions.append({"opname": name})
    kinds["ExecutionModel"] = [
        {"enumerant": name, "capabilities": caps} for name, caps in _MODELS
    ]
    return json.dumps(
        {
            "instructions": instructions,
            "operand_kinds": [
                {"kind": kind, "enumerants": values} for kind, values in kinds.items()
            ],
        }
    )


GRAMMAR = grammar()


def construct_feature(language: str, name: str) -> Feature:
    (feature,) = (f for f in constructs.features(language, GRAMMAR) if f.name == name)
    return feature
