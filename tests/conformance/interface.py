"""The harness's storage-buffer contract and its discovery in each target language.

Every conformance kernel reads one storage buffer bound at slot 0 and writes one bound at slot 1, in the binding
namespace of its language: ``[[buffer(n)]]`` in Metal, ``register(tN)``/``register(uN)`` in HLSL, ``binding = N``
in GLSL and ``Binding N`` in descriptor set 0 in SPIR-V. A translation that renumbers, adds or removes a resource
changes the interface a host binds, so discovery raises :class:`InterfaceChanged` instead of guessing from names.
"""

from __future__ import annotations

import re
from pathlib import Path

INPUT, OUTPUT = 0, 1


class InterfaceChanged(ValueError):
    """The artifact does not expose the two-buffer compute interface of its source."""


ENTRY_PATTERNS = {
    "metal": r"\bkernel\s+void\s+(\w+)\s*\(",
    "hlsl": r"\[numthreads\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\]\s*void\s+(\w+)",
    "glsl": r"\blayout\s*\(([^)]*\blocal_size_x\b[^)]*)\)\s*in\s*;",
    "spirv": r'OpEntryPoint\s+GLCompute\s+%\S+\s+"([^"]+)"',
}

SCALAR_BYTES = {
    "int": 4,
    "uint": 4,
    "float": 4,
    "dword": 4,
    "bool": 4,
    "half": 2,
    "float16_t": 2,
    "int16_t": 2,
    "uint16_t": 2,
    "double": 8,
    "int64_t": 8,
    "uint64_t": 8,
}


def _single(pattern, text):
    found = list(re.finditer(pattern, text))
    if len(found) != 1:
        raise InterfaceChanged(f"Expected one compute entry point, found {len(found)}")
    return found[0]


def _strip_comments(text):
    return re.sub(r"//[^\n]*|/\*.*?\*/", " ", text, flags=re.S)


def _parameter_list(text, start):
    depth, index = 1, start
    while depth and index < len(text):
        depth += {"(": 1, ")": -1}.get(text[index], 0)
        index += 1
    if depth:
        raise InterfaceChanged("Unterminated kernel parameter list")
    return text[start : index - 1]


def _slots(found, what):
    slots = sorted(found)
    if slots != [INPUT, OUTPUT]:
        raise InterfaceChanged(f"Expected {what} slots [0, 1], found {slots}")
    return INPUT, OUTPUT


def _metal(text, group):
    entry = _single(ENTRY_PATTERNS["metal"], text)
    parameters = _parameter_list(text, entry.end())
    resources = re.findall(
        r"\[\[\s*(buffer|texture|sampler)\s*\(\s*(\d+)\s*\)\s*\]\]", parameters
    )
    kinds = {kind for kind, _ in resources}
    if kinds - {"buffer"}:
        raise InterfaceChanged(
            f"Unexpected Metal resource kinds: {sorted(kinds - {'buffer'})}"
        )
    return {
        "entry": entry.group(1),
        "group": group,
        "bindings": _slots([int(index) for _, index in resources], "Metal buffer"),
    }


def _hlsl_stride(type_name, text):
    type_name = type_name.strip()
    vector = re.fullmatch(r"(\w+?)([1-4])?", type_name)
    if vector and vector.group(1) in SCALAR_BYTES:
        return SCALAR_BYTES[vector.group(1)] * int(vector.group(2) or 1)
    struct = re.search(
        rf"\bstruct\s+{re.escape(type_name)}\s*\{{(.*?)\}}\s*;", text, flags=re.S
    )
    if not struct:
        return None
    total = 0
    for field_type, count in re.findall(
        r"(\w+)\s+\w+\s*(?:\[\s*(\d+)\s*\])?\s*;", struct.group(1)
    ):
        size = _hlsl_stride(field_type, text)
        if size is None:
            return None
        total += size * int(count or 1)
    return total or None


def _hlsl(text, group):
    entry = _single(ENTRY_PATTERNS["hlsl"], text)
    declared = re.findall(
        r"\b(RW)?(StructuredBuffer|ByteAddressBuffer)\s*(?:<\s*([\w\s]+?)\s*>)?\s+(\w+)\s*:\s*"
        r"register\(\s*([a-z])(\d+)\s*(?:,\s*space(\d+)\s*)?\)",
        text,
    )
    registers = re.findall(
        r"register\(\s*([a-z])(\d+)\s*(?:,\s*space(\d+)\s*)?\)", text
    )
    if len(registers) != len(declared) or any(
        space not in ("", "0") for *_, space in declared
    ):
        raise InterfaceChanged(f"Unexpected HLSL registers: {registers}")
    resources = {}
    for writable, kind, element, name, register, slot, _ in declared:
        if (register, bool(writable)) not in {("t", False), ("u", True)}:
            raise InterfaceChanged(
                f"Register class {register} does not match {writable}{kind} {name}"
            )
        resources[int(slot)] = {
            "register": f"{register}{slot}",
            "name": name,
            "kind": "structured" if kind == "StructuredBuffer" else "raw",
            "stride": _hlsl_stride(element, text) if element else 0,
        }
    bindings = _slots(list(resources), "HLSL register")
    if resources[OUTPUT]["register"] != "u1":
        raise InterfaceChanged("The output buffer must be the UAV u1")
    return {
        "entry": entry.group(4),
        "group": tuple(int(entry.group(i)) for i in (1, 2, 3)),
        "bindings": bindings,
        "resources": [resources[INPUT], resources[OUTPUT]],
    }


def _glsl(text, _group):
    layout = _single(ENTRY_PATTERNS["glsl"], text).group(1)
    size = {
        axis: int(value)
        for axis, value in re.findall(r"local_size_([xyz])\s*=\s*(\d+)", layout)
    }
    blocks = re.findall(r"layout\s*\(([^)]*)\)\s*(?:\w+\s+)*buffer\b", text)
    slots = []
    for qualifiers in blocks:
        binding = re.search(r"\bbinding\s*=\s*(\d+)", qualifiers)
        if binding is None:
            raise InterfaceChanged("A GLSL storage block has no explicit binding")
        slots.append(int(binding.group(1)))
    if re.search(r"\buniform\b", text):
        raise InterfaceChanged(
            "The GLSL artifact declares uniforms the host does not provide"
        )
    return {
        "entry": "main",
        "group": (size.get("x", 1), size.get("y", 1), size.get("z", 1)),
        "bindings": _slots(slots, "GLSL storage binding"),
    }


def _spirv(text, _group):
    entry = _single(ENTRY_PATTERNS["spirv"], text)
    size = re.search(
        r"OpExecutionMode\s+%\S+\s+LocalSize\s+(\d+)\s+(\d+)\s+(\d+)", text
    )
    if size is None:
        raise InterfaceChanged("Missing literal SPIR-V workgroup size")
    sets = {
        int(value)
        for value in re.findall(r"OpDecorate\s+%\S+\s+DescriptorSet\s+(\d+)", text)
    }
    if sets - {0}:
        raise InterfaceChanged(f"Unexpected descriptor sets {sorted(sets)}")
    bindings = [
        int(value) for value in re.findall(r"OpDecorate\s+%\S+\s+Binding\s+(\d+)", text)
    ]
    return {
        "entry": entry.group(1),
        "group": tuple(int(size.group(i)) for i in (1, 2, 3)),
        "bindings": _slots(bindings, "SPIR-V binding"),
    }


DISCOVER = {"metal": _metal, "hlsl": _hlsl, "glsl": _glsl, "spirv": _spirv}


def interface(path, target, source_group):
    """Entry point, one-dimensional workgroup size and (input, output) binding slots of an artifact."""
    text = _strip_comments(Path(path).read_text())
    found = DISCOVER[target](text, (source_group, 1, 1))
    group = tuple(found["group"])
    if min(group) < 1 or group[1:] != (1, 1):
        raise InterfaceChanged(
            f"Only one-dimensional workgroups are supported, found {group}"
        )
    found["group"] = group
    found["bindings"] = tuple(found["bindings"])
    return found


def hlsl_reflected_strides(dumpbin):
    """Structured-buffer element sizes reported by ``dxc -dumpbin``, keyed by resource name."""
    strides = {}
    for name, body in re.findall(
        r";\s*Resource bind info for (\w+)\s*\n(.*?)\n;\s*\}\s*(?:\n|$)",
        dumpbin,
        flags=re.S,
    ):
        element = re.search(r"\$Element;\s*;\s*Offset:\s*\d+\s+Size:\s*(\d+)", body)
        if element:
            strides[name] = int(element.group(1))
    return strides
