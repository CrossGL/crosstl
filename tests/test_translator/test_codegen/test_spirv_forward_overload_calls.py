"""SPIR-V output resolves a call to an overloaded function declared after its caller.

The SPIR-V generator emits each function before the functions that call it and
resolves a call against the overloads it has already emitted. The dependency
order used to ignore overloaded names, so a caller declared before every
overload of its callee was emitted first, and the call was refused as
"neither a declared function nor a built-in". A call now depends on every
overload of its callee's name.
"""

from __future__ import annotations

import re

from crosstl._crosstl import translate
from tests.test_translator.test_codegen.test_SPIRV_codegen import (
    assert_spirv_module_validates,
)


def _generate(source, tmp_path):
    path = tmp_path / "overloads.cgl"
    path.write_text(source, encoding="utf-8")
    return translate(str(path), backend="vulkan")


def _scalar_type_names(spirv):
    """Map scalar type ids to ``int``/``float``.

    Ids are numeric unless SPIRV-Tools is installed to give them friendly names, so
    the declarations are read instead of the ids.
    """
    names = {tid: "int" for tid in re.findall(r"%(\w+) = OpTypeInt 32 1\b", spirv)}
    names.update(
        (tid, "float") for tid in re.findall(r"%(\w+) = OpTypeFloat 32\b", spirv)
    )
    return names


def _functions_by_result_type(spirv):
    """Map each defined function's result id to its result type name."""
    names = _scalar_type_names(spirv)
    return {
        fid: names.get(rtype, rtype)
        for fid, rtype in re.findall(r"%(\w+) = OpFunction %(\w+) None", spirv)
    }


def _call_targets(spirv):
    names = _scalar_type_names(spirv)
    return [
        (names.get(rtype, rtype), target)
        for rtype, target in re.findall(r"= OpFunctionCall %(\w+) %(\w+)", spirv)
    ]


AFTER_CALLER = """
shader OverloadAfterCaller {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* result;

        void main() {
            result[0] = helper(2) + scaled(1.5);
        }
    }
    float helper(int k) {
        return float(twice(k));
    }
    float scaled(float v) {
        return twice(v);
    }
    float twice(float x) { return x + x; }
    int twice(int x) { return x + x; }
}
"""


def test_call_to_overload_declared_after_caller_resolves_by_argument_type(tmp_path):
    spirv = _generate(AFTER_CALLER, tmp_path)
    result_types = _functions_by_result_type(spirv)
    calls = _call_targets(spirv)
    # helper calls the int overload and scaled the float overload of twice.
    int_twice = [fid for fid, rtype in result_types.items() if rtype == "int"]
    assert len(int_twice) == 1
    assert ("int", int_twice[0]) in calls
    float_calls = [target for rtype, target in calls if rtype == "float"]
    assert len(set(float_calls)) == 3  # helper, scaled and twice(float)
    # Every callee is defined before the function that calls it.
    for call in re.finditer(r"= OpFunctionCall %\w+ %(\w+)", spirv):
        assert spirv.index(f"%{call.group(1)} = OpFunction ") < call.start()
    assert_spirv_module_validates(spirv, tmp_path, target_env="vulkan1.0")


INTERLEAVED = """
shader InterleavedOverloads {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* result;

        void main() {
            result[0] = pick(0.5);
        }
    }
    float pick(float v) {
        return float(bridge(int(v)));
    }
    int bridge(int k) {
        return pick(k);
    }
    int pick(int k) {
        return k + 1;
    }
}
"""


def test_overload_chain_through_another_function_orders_each_overload(tmp_path):
    # pick(float) -> bridge -> pick(int): the int overload is emitted first,
    # then bridge, then the float overload, so no call is recursive.
    spirv = _generate(INTERLEAVED, tmp_path)
    result_types = _functions_by_result_type(spirv)
    calls = _call_targets(spirv)
    int_functions = [fid for fid, rtype in result_types.items() if rtype == "int"]
    assert len(int_functions) == 2
    assert {target for rtype, target in calls if rtype == "int"} == set(int_functions)
    assert_spirv_module_validates(spirv, tmp_path, target_env="vulkan1.0")


FAMILY_AFTER_CALLER = """
shader OverloadFamilyAfterCaller {
    fragment {
        vec4 main() {
            return encode(vec4(0.1, 0.2, 0.3, 1.0));
        }
    }
    vec4 encode(vec4 c) {
        return vec4(srgb(c), c.a);
    }
    float srgb(float c) { return c + 1.0; }
    vec3 srgb(vec3 c) { return vec3(srgb(c.r), srgb(c.g), srgb(c.b)); }
    vec3 srgb(vec4 c) { return srgb(c.rgb); }
}
"""


def test_overloads_calling_each_other_after_caller_keep_source_order(tmp_path):
    # encode depends on the whole srgb family; inside the family each overload
    # calls an earlier one, so the family keeps its source order.
    spirv = _generate(FAMILY_AFTER_CALLER, tmp_path)
    assert spirv.count("OpFunctionCall") == 6
    for call in re.finditer(r"= OpFunctionCall %\w+ %(\w+)", spirv):
        assert spirv.index(f"%{call.group(1)} = OpFunction ") < call.start()
    assert_spirv_module_validates(spirv, tmp_path, target_env="vulkan1.0")
