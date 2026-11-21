"""The CrossGL constructs the Rust generator writes as rust-gpu code, and the typed refusal of everything else.

The generator writes ``#![no_std]`` crates for rust-gpu 0.11: compute entry points whose storage buffers, uniform
values, workgroup memory and built-in inputs are entry-point parameters (:mod:`.rust_gpu_interface`), glam vectors
and matrices, structs, and the built-in functions :mod:`.rust_gpu_builtins` names. Other stages and resources
(images, samplers, HLSL buffer objects, GLSL buffer blocks, constant buffers), and globals a program writes, which
rust-gpu cannot declare as statics, are refused here by name before any code is written, so a program either
compiles with rust-gpu or is refused with every reason listed.
"""

from __future__ import annotations

import re

from ..ast import (
    AssignmentNode,
    AtomicOpNode,
    BufferOpNode,
    CooperativeMatrixOpNode,
    IdentifierNode,
    MeshOpNode,
    PointerReinterpretNode,
    RayQueryOpNode,
    RayTracingOpNode,
    TextureNode,
    TextureOpNode,
    UnaryOpNode,
    VariableNode,
    WaveOpNode,
)
from .ast_walk import walk_ast_values
from .stage_utils import normalize_stage_name

# Entry-point stages the generator writes for rust-gpu.
RUST_GPU_STAGES = frozenset({"compute", "vertex", "fragment"})
_NODE_REASONS = (
    (TextureNode, "texture sampling"),
    (TextureOpNode, "texture operations"),
    (AtomicOpNode, "atomic operations"),
    (BufferOpNode, "buffer object operations"),
    (WaveOpNode, "subgroup operations"),
    (RayTracingOpNode, "ray tracing operations"),
    (RayQueryOpNode, "ray queries"),
    (MeshOpNode, "mesh shader outputs"),
    (CooperativeMatrixOpNode, "cooperative matrix operations"),
    (PointerReinterpretNode, "pointer reinterpretation"),
)


def _root_name(node):
    while node is not None and not isinstance(node, IdentifierNode):
        node = (
            getattr(node, "array_expr", None)
            or getattr(node, "object_expr", None)
            or getattr(node, "pointer", None)
        )
    return node.name if isinstance(node, IdentifierNode) else None


def written_names(ast) -> frozenset[str]:
    """The names a program assigns to, increments or decrements, through any element or member."""
    names = set()
    for value in walk_ast_values(ast):
        if isinstance(value, AssignmentNode):
            name = _root_name(getattr(value, "target", None))
        elif isinstance(value, UnaryOpNode) and str(
            getattr(value, "op", getattr(value, "operator", ""))
        ) in {"++", "--"}:
            name = _root_name(getattr(value, "operand", None))
        else:
            continue
        if name:
            names.add(name)
    return frozenset(names)


def program_stage_names(ast):
    """The normalized names of every stage ``ast`` declares, as a stage block or a stage-qualified function."""
    return _stage_names(ast)


def _stage_names(ast):
    stages = getattr(ast, "stages", None) or {}
    names = set()
    for stage_type in stages:
        names.add(normalize_stage_name(str(stage_type).split(".")[-1].lower()))
    for func in getattr(ast, "functions", None) or []:
        stage = generator_stage_name(func)
        if stage:
            names.add(stage)
    return names


def generator_stage_name(func):
    for qualifier in list(getattr(func, "qualifiers", None) or []) + [
        getattr(func, "qualifier", None)
    ]:
        name = normalize_stage_name(qualifier) if qualifier else None
        if name in {
            "vertex",
            "fragment",
            "compute",
            "geometry",
            "tessellation_control",
            "tessellation_evaluation",
            "mesh",
            "task",
            "amplification",
            "object",
            "ray_generation",
            "closest_hit",
            "any_hit",
            "miss",
            "intersection",
            "callable",
        }:
            return name
    return None


def _declarations(ast):
    yield from getattr(ast, "global_variables", None) or []
    for stage in (getattr(ast, "stages", None) or {}).values():
        yield from getattr(stage, "local_variables", None) or []


def refuse_outside_rust_gpu_subset(generator, ast) -> None:
    """Record a refusal on ``generator`` for each construct of ``ast`` the rust-gpu output does not lower."""
    for stage in sorted(_stage_names(ast) - RUST_GPU_STAGES):
        generator.refuse_lowering(
            f"stage: Rust output writes rust-gpu compute, vertex and fragment entry points; {stage} entry "
            "points are not lowered for rust-gpu"
        )
    written = written_names(ast)
    for node in _declarations(ast):
        _refuse_declaration(generator, node, written)
    if getattr(ast, "cbuffers", None):
        generator.refuse_lowering(
            "resource: constant buffers are not lowered to rust-gpu uniform blocks"
        )
    seen = set()
    for value in walk_ast_values(ast):
        for node_type, reason in _NODE_REASONS:
            if isinstance(value, node_type) and reason not in seen:
                seen.add(reason)
                generator.refuse_lowering(
                    f"{reason}: not lowered to rust-gpu's spirv_std"
                )


def _refuse_declaration(generator, node, written):
    name = getattr(node, "name", None)
    if not isinstance(node, VariableNode) or not name:
        return
    type_name = generator.rust_variable_source_type(node)
    if generator.rust_gpu_interface_info(node, type_name) is not None:
        return
    qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", None) or []}
    if generator.rust_resource_kind(type_name, node=node) is not None:
        generator.refuse_lowering(
            f"resource: '{name}' ({type_name}) is not lowered to a rust-gpu resource binding"
        )
        return
    if qualifiers & {"uniform", "buffer", "in", "out", "inout", "varying", "attribute"}:
        generator.refuse_lowering(
            f"resource: '{name}' ({' '.join(sorted(qualifiers))} {type_name}) is not lowered to a rust-gpu "
            "entry-point parameter"
        )
        return
    if name in written and "const" not in qualifiers:
        generator.refuse_lowering(
            f"global: '{name}' is written, and rust-gpu has no mutable statics"
        )


# Types and macros of Rust's ``alloc`` and ``std`` that a ``#![no_std]`` rust-gpu crate cannot name, and string
# constants, which rust-gpu cannot lower ("unsupported unsized `[u8]` constant").
_HOST_ONLY = (
    (re.compile(r"(?<![\w:])Vec<"), "a dynamic array (Vec) needs an allocator, which rust-gpu shaders do not have"),
    (re.compile(r"\bvec!\["), "a dynamic array (vec!) needs an allocator, which rust-gpu shaders do not have"),
    (re.compile(r"(?<![\w:])String\b"), "strings need an allocator, which rust-gpu shaders do not have"),
    (re.compile(r"&'static str\b"), "rust-gpu cannot lower string constants"),
)


def refuse_host_only_rust(generator, code: str) -> None:
    """Refuse a generated crate that names an allocator type or a string constant (each reason once)."""
    for pattern, reason in _HOST_ONLY:
        if pattern.search(code):
            generator.refuse_lowering(f"type: {reason}")
