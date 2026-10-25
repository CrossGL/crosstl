"""Write the fixed-sampler records of :mod:`..semantics.fixed_samplers` into a generated artifact.

Each record names a sampler that replaced a fixed sampler state, with the state the host must give it. The generator
adds the register (HLSL) or descriptor set and binding (SPIR-V) it emitted for that sampler, read back from its own
output, and writes the records the artifact uses as its ``crosstl-fixed-samplers`` header. A sampler the artifact
uses but does not declare under its recorded name is refused, because a host could not bind its state.
"""

from __future__ import annotations

import json
import re

from ..artifact_contracts import FIXED_SAMPLER_PREFIX, with_artifact_contract_header
from ..ast import IdentifierNode
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from ..semantics.fixed_samplers import ANNOTATION, fixed_sampler_header
from ..semantics.rewrite import child_slots, get_child

__all__ = [
    "FixedSamplerBindingError",
    "fixed_sampler_records",
    "parse_fixed_sampler_contract",
    "with_glsl_fixed_samplers",
    "with_hlsl_fixed_samplers",
    "with_spirv_fixed_samplers",
]


class FixedSamplerBindingError(TranslationDiagnostic):
    """A fixed sampler the artifact uses without a declaration the host could bind its state to."""

    def __init__(self, target, name):
        self.missing_capabilities = (f"{target}.fixed-sampler-binding",)
        super().__init__(
            f"fixed sampler '{name}' is used but not declared under that name, so its state cannot be bound",
            target=target,
            code=f"{CODE_PREFIX}{target}-fixed-sampler-binding-missing",
        )


def fixed_sampler_records(ast):
    """The fixed-sampler records ``ast`` carries, copied so a generator can complete them."""
    annotations = getattr(ast, "annotations", None) or {}
    return [dict(record) for record in annotations.get(ANNOTATION) or []]


def _referenced_names(ast):
    names, stack, seen = set(), [ast], set()
    while stack:
        node = stack.pop()
        if isinstance(node, list):
            stack.extend(node)
            continue
        if node is None or id(node) in seen or not hasattr(node, "__dict__"):
            continue
        seen.add(id(node))
        if isinstance(node, IdentifierNode):
            names.add(node.name)
        stack.extend(get_child(node, field, key) for field, key in child_slots(node))
    return names


def _used(records, ast):
    names = _referenced_names(ast)
    return [record for record in records if record["name"] in names]


def with_hlsl_fixed_samplers(code, records, ast):
    """``code`` with a ``// crosstl-fixed-samplers:`` header for each fixed sampler it uses."""
    resolved = []
    for record in _used(records, ast):
        name = re.escape(record["name"])
        match = re.search(
            rf"\b(?:SamplerState|SamplerComparisonState)\s+{name}\s*:\s*register\(\s*(s\d+)"
            r"\s*(?:,\s*space(\d+)\s*)?\)",
            code,
        )
        if match is None:
            raise FixedSamplerBindingError("directx", record["name"])
        record.update(register=match.group(1), space=int(match.group(2) or 0))
        resolved.append(record)
    if not resolved:
        return code
    return with_artifact_contract_header(code, fixed_sampler_header(resolved))


def with_glsl_fixed_samplers(code, records, pairs, ast):
    """GLSL ``code`` with a ``// crosstl-fixed-samplers:`` header naming the combined uniforms of each sampler.

    OpenGL GLSL has no separate samplers: each texture read through a sampler state is a combined uniform
    (:mod:`.sampler_pairs`), and the host binds a sampler object with that state to the uniform's texture unit
    (OpenGL 4.6 section 8.2). Each record lists those uniforms, with the unit the declaration names.
    """
    resolved = []
    for record in _used(records, ast):
        uniforms = []
        for pair in pairs or ():
            if pair.get("sampler") != record["name"] or pair["uniform"] in (
                entry["uniform"] for entry in uniforms
            ):
                continue
            match = re.search(
                rf"layout\([^)]*\bbinding\s*=\s*(\d+)[^)]*\)\s*uniform\s+\w+\s+"
                rf"{re.escape(pair['uniform'])}\b",
                code,
            )
            uniforms.append(
                {
                    "uniform": pair["uniform"],
                    "texture": pair["texture"],
                    "unit": int(match.group(1)) if match else None,
                }
            )
        if not uniforms:
            raise FixedSamplerBindingError("opengl", record["name"])
        record.update(uniforms=uniforms)
        resolved.append(record)
    if not resolved:
        return code
    return with_artifact_contract_header(code, fixed_sampler_header(resolved))


def with_spirv_fixed_samplers(code, records, ast):
    """SPIR-V assembly ``code`` with an ``OpSourceExtension "crosstl-fixed-samplers: ..."`` for its fixed samplers.

    A debug instruction rather than a comment, so the record survives assembly into the module; it has no semantic
    effect (SPIR-V 1.0 section 3.32.2) and sits before the first ``OpName``, in the debug section.
    """
    resolved = []
    for record in _used(records, ast):
        named = re.search(rf'OpName (%\w+) "{re.escape(record["name"])}"', code)
        identifier = re.escape(named.group(1)) if named else None
        found = {}
        for decoration in ("DescriptorSet", "Binding"):
            match = identifier and re.search(
                rf"OpDecorate {identifier} {decoration} (\d+)\b", code
            )
            if match:
                found[decoration] = int(match.group(1))
        if "Binding" not in found:
            raise FixedSamplerBindingError("vulkan", record["name"])
        record.update(set=found.get("DescriptorSet", 0), binding=found["Binding"])
        resolved.append(record)
    if not resolved:
        return code
    payload = fixed_sampler_header(resolved)[len("// ") :]
    literal = payload.replace("\\", "\\\\").replace('"', '\\"')
    lines = code.split("\n")
    index = next(
        i for i, line in enumerate(lines) if line.lstrip().startswith("OpName ")
    )
    indent = lines[index][: len(lines[index]) - len(lines[index].lstrip())]
    lines.insert(index, f'{indent}OpSourceExtension "{literal}"')
    return "\n".join(lines)


def parse_fixed_sampler_contract(source):
    """The fixed-sampler records of an HLSL or SPIR-V assembly artifact, or ``[]`` without any."""
    prefix = FIXED_SAMPLER_PREFIX[len("// ") :]
    for line in source.splitlines():
        text = line.strip()
        if text.startswith(FIXED_SAMPLER_PREFIX):
            return json.loads(text[len(FIXED_SAMPLER_PREFIX) :])["samplers"]
        if text.startswith(f'OpSourceExtension "{prefix}'):
            literal = text[len('OpSourceExtension "') : -1]
            payload = re.sub(r"\\(.)", r"\1", literal)
            return json.loads(payload[len(prefix) :])["samplers"]
    return []
