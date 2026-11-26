"""CUDA and HIP output: implicit conversions between vector types.

GLSL converts an ``int`` vector to a ``uint``, ``float`` or ``double`` vector of the same size, a ``uint`` vector to
a ``float`` or ``double`` one and a ``float`` vector to a ``double`` one wherever such a value initializes or is
assigned to a variable of the other type (GLSL 4.60, 4.1.10 "Implicit Conversions"), and CrossGL does the same.
CUDA's and HIP's vector types are distinct structures with no conversions between them, so ``uint3 n = v;`` with an
``int3 v`` does not compile ("no suitable user-defined conversion"). The generators convert each component through
a ``__device__`` helper. The helper is a template over its argument, so it converts whatever vector the expression
turns out to be; it is used only when the value's inferred type is a vector of the same size and another component
type, so a value that already has the target type is written as it is.
"""

_CONVERTIBLE_COMPONENTS = {"int", "uint", "float", "double"}


def convert_vector_value(generator, node, target_type, text):
    """``text``, the rendering of ``node``, converted to the vector ``target_type`` when its type differs."""
    if target_type is None or not text:
        return text
    target_info = generator.vector_type_info(generator.type_name_string(target_type))
    if (
        target_info is None
        or target_info["component_type"] not in _CONVERTIBLE_COMPONENTS
    ):
        return text
    value_info = generator.vector_type_info(generator.expression_result_type(node))
    if (
        value_info is None
        or value_info["component_type"] not in _CONVERTIBLE_COMPONENTS
        or value_info["component_type"] == target_info["component_type"]
        or len(value_info["components"]) != len(target_info["components"])
    ):
        return text
    return f"{_require_conversion_helper(generator, target_info)}({text})"


def _require_conversion_helper(generator, target_info):
    target = target_info["type"]
    helper_name = generator.sanitize_helper_name(f"cgl_convert_to_{target}")
    if helper_name in generator.helper_functions:
        return helper_name
    component_type = generator.map_type(target_info["component_type"])
    components = ", ".join(
        f"static_cast<{component_type}>(value.{component})"
        for component in target_info["components"]
    )
    generator.helper_functions[helper_name] = (
        "template <typename T>\n"
        f"__device__ inline {target} {helper_name}(T value)\n"
        "{\n"
        f"    return {target_info['constructor']}({components});\n"
        "}"
    )
    return helper_name
