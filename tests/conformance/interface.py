"""The harness's buffer contract and its discovery in each target language.

Every conformance kernel reads one storage buffer bound at slot 0 and writes one bound at slot 1, in the binding
namespace of its language: ``[[buffer(n)]]`` in Metal, ``register(tN)``/``register(uN)`` in HLSL, ``binding = N``
in GLSL and ``Binding N`` in descriptor set 0 in SPIR-V.

A case may also give the host one constants block, which it uploads byte for byte in the source's layout. The
artifact declares it as a ``constant``-address-space reference argument in Metal, a ``cbuffer`` or
``ConstantBuffer`` at ``register(bN)`` in HLSL, a ``uniform`` block with an explicit binding in GLSL, and a
``Uniform`` ``Block`` variable or a ``PushConstant`` variable in SPIR-V. Discovery reports where the artifact
expects the block; whether the translation kept the source's offsets is decided by execution, not here.

A translation that renumbers, adds or removes a resource changes the interface a host binds, so discovery raises
:class:`InterfaceChanged` instead of guessing from names.
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


def _split_parameters(text):
    """Top-level comma-separated parameters; template arguments and attributes keep their commas."""
    parameters, depth, start = [], 0, 0
    for index, char in enumerate(text):
        depth += {"(": 1, "<": 1, "[": 1, ")": -1, ">": -1, "]": -1}.get(char, 0)
        if char == "," and depth == 0:
            parameters.append(text[start:index])
            start = index + 1
    parameters.append(text[start:])
    return [parameter.strip() for parameter in parameters if parameter.strip()]


def _constants(found, expected, what):
    """The single constants block of an artifact, or ``None`` when the case has none."""
    if not expected:
        if found:
            raise InterfaceChanged(
                f"The artifact declares a constants block the host does not provide: {what} {found}"
            )
        return None
    if len(found) != 1:
        raise InterfaceChanged(f"Expected one constants block, found {what} {found}")
    return found[0]


def _metal(text, group, constants):
    entry = _single(ENTRY_PATTERNS["metal"], text)
    storage, blocks, kinds = [], [], set()
    for parameter in _split_parameters(_parameter_list(text, entry.end())):
        resource = re.search(
            r"\[\[\s*(buffer|texture|sampler)\s*\(\s*(\d+)\s*\)\s*\]\]", parameter
        )
        if resource is None:
            continue
        kind, slot = resource.group(1), int(resource.group(2))
        if kind != "buffer":
            kinds.add(kind)
        elif re.match(r"(?:const\s+)?constant\b[^&*]*&", parameter):
            blocks.append({"slot": slot, "kind": "uniform"})
        else:
            storage.append(slot)
    if kinds:
        raise InterfaceChanged(f"Unexpected Metal resource kinds: {sorted(kinds)}")
    block = _constants(blocks, constants, "Metal constant buffers")
    if block is not None and block["slot"] in (INPUT, OUTPUT):
        raise InterfaceChanged(
            "The Metal constants block shares a storage-buffer index"
        )
    return {
        "entry": entry.group(1),
        "group": group,
        "bindings": _slots(storage, "Metal buffer"),
        "constants": block,
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


HLSL_REGISTER = r"register\(\s*([a-z])(\d+)\s*(?:,\s*space(\d+)\s*)?\)"


def _hlsl_constant_buffers(text):
    declared = re.findall(
        r"\b(?:cbuffer\s+\w+|ConstantBuffer\s*<\s*\w+\s*>\s+\w+)\s*:\s*"
        + HLSL_REGISTER,
        text,
    )
    if len(declared) != len(re.findall(r"\bcbuffer\b|\bConstantBuffer\s*<", text)):
        raise InterfaceChanged("An HLSL constant buffer has no explicit register")
    blocks = []
    for register, slot, space in declared:
        if register != "b" or space not in ("", "0"):
            raise InterfaceChanged(
                f"Unexpected HLSL constant-buffer register {register}{slot}, space{space or 0}"
            )
        blocks.append({"slot": int(slot), "kind": "uniform", "register": f"b{slot}"})
    return blocks


def _hlsl(text, group, constants):
    entry = _single(ENTRY_PATTERNS["hlsl"], text)
    blocks = _hlsl_constant_buffers(text)
    declared = re.findall(
        r"\b(RW)?(StructuredBuffer|ByteAddressBuffer)\s*(?:<\s*([\w\s]+?)\s*>)?\s+(\w+)\s*:\s*"
        r"register\(\s*([a-z])(\d+)\s*(?:,\s*space(\d+)\s*)?\)",
        text,
    )
    registers = re.findall(HLSL_REGISTER, text)
    if len(registers) != len(declared) + len(blocks) or any(
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
        "constants": _constants(blocks, constants, "HLSL constant buffers"),
    }


def _glsl(text, _group, constants):
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
    uniform_blocks = []
    for qualifiers in re.findall(r"layout\s*\(([^)]*)\)\s*uniform\s+\w+\s*\{", text):
        binding = re.search(r"\bbinding\s*=\s*(\d+)", qualifiers)
        if re.search(r"\bpush_constant\b", qualifiers):
            uniform_blocks.append({"slot": None, "kind": "push"})
        elif binding is None:
            raise InterfaceChanged("A GLSL uniform block has no explicit binding")
        else:
            uniform_blocks.append({"slot": int(binding.group(1)), "kind": "uniform"})
    if len(re.findall(r"\buniform\b", text)) != len(uniform_blocks):
        raise InterfaceChanged(
            "The GLSL artifact declares uniforms the host does not provide"
        )
    return {
        "entry": "main",
        "group": (size.get("x", 1), size.get("y", 1), size.get("z", 1)),
        "bindings": _slots(slots, "GLSL storage binding"),
        "constants": _constants(uniform_blocks, constants, "GLSL uniform blocks"),
    }


def _spirv_constant_blocks(text):
    """Variables of ``Uniform`` storage whose struct is a ``Block``, and ``PushConstant`` variables."""
    pointers = {
        name: (storage, pointee)
        for name, storage, pointee in re.findall(
            r"(%\S+)\s*=\s*OpTypePointer\s+(\w+)\s+(%\S+)", text
        )
    }
    blocks = set(re.findall(r"OpDecorate\s+(%\S+)\s+Block\b", text))
    found = {}
    for name, pointer, storage in re.findall(
        r"(%\S+)\s*=\s*OpVariable\s+(%\S+)\s+(Uniform|PushConstant)\b", text
    ):
        pointee = pointers.get(pointer, (None, None))[1]
        if storage == "PushConstant":
            found[name] = "push"
        elif pointee in blocks:
            found[name] = "uniform"
    return found


def _spirv(text, _group, constants):
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
    uniform = _spirv_constant_blocks(text)
    bound = re.findall(r"OpDecorate\s+(%\S+)\s+Binding\s+(\d+)", text)
    blocks = [
        {"slot": int(value), "kind": "uniform"}
        for name, value in bound
        if uniform.get(name) == "uniform"
    ]
    blocks += [
        {"slot": None, "kind": "push"} for kind in uniform.values() if kind == "push"
    ]
    return {
        "entry": entry.group(1),
        "group": tuple(int(size.group(i)) for i in (1, 2, 3)),
        "bindings": _slots(
            [int(value) for name, value in bound if name not in uniform],
            "SPIR-V binding",
        ),
        "constants": _constants(blocks, constants, "SPIR-V constant blocks"),
    }


DISCOVER = {"metal": _metal, "hlsl": _hlsl, "glsl": _glsl, "spirv": _spirv}


def interface(path, target, source_group, constants=False):
    """Entry point, one-dimensional workgroup size, (input, output) binding slots and constants block of an artifact.

    ``constants`` says whether the case gives the host a constants block; the result's ``constants`` is then the
    block's ``slot`` (``None`` for push constants) and ``kind`` (``uniform`` or ``push``), and ``None`` otherwise.
    """
    text = _strip_comments(Path(path).read_text())
    found = DISCOVER[target](text, (source_group, 1, 1), constants)
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
