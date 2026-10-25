"""GLSL extension types CrossGL cannot represent are refused at import instead of being dropped."""

import pytest

import crosstl
from crosstl.backend.GLSL.unsupported_types import GLSLTypeUnsupportedError

SOURCE = """#version 450
#extension GL_ARM_tensors : enable
layout(local_size_x = 1) in;
layout(set = 0, binding = 0) uniform tensorARM<float, 4> input_tensor;
layout(std430, binding = 1) buffer O { float o[]; };
void main() {
    float value;
    tensorReadARM(input_tensor, uint[](0, 0, 0, 0), value);
    o[0] = value;
}
"""


@pytest.mark.parametrize("backend", ["cgl", "opengl", "directx", "vulkan", "metal"])
def test_arm_tensors_are_refused(tmp_path, backend):
    # The declaration used to disappear, so every target named the undeclared input_tensor.
    path = tmp_path / "kernel.comp"
    path.write_text(SOURCE)
    with pytest.raises(GLSLTypeUnsupportedError) as raised:
        crosstl.translate(str(path), backend=backend)
    assert raised.value.type_name == "tensorARM"
    assert raised.value.code == "project.translate.opengl-import-type-unsupported"
