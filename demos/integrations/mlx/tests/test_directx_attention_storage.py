"""Host binding controls for the attention demo's explicit half storage."""

import pytest

from crosstl.project.native_runtime_drivers import _prepare_directx_buffers
from crosstl.translator.resource_storage import (
    BINARY16_STORAGE,
    resource_storage_header,
)
from demos.integrations.mlx.tests.kernels import test_attention_ds_runtime as derivative
from demos.integrations.mlx.tests.kernels import test_attention_odo_runtime as row_dot
from demos.integrations.mlx.tests.kernels import (
    test_attention_reduce_runtime as reduction,
)


@pytest.fixture(params=("row-dot", "derivative", "reduction"))
def binding_case(request, tmp_path):
    np = pytest.importorskip("numpy")
    family = request.param

    def build(dtype, damage=None):
        element = "float" if dtype == "float" else "uint16_t"
        if family == "row-dot":
            buffers, _ = row_dot._dataset(np, 64, dtype, "dyadic", 1)
            names = ("o", "cot_o", "odo")
            elements = (element, element, "float")
            readonly = 2
            constants = "cbuffer Parameters : register(b3) { int qL; };"
            inputs = [buffers[:3]]
        elif family == "derivative":
            buffers, _, parameters = derivative._dataset(np, dtype, "dyadic", 0, True)
            names = derivative.NAMES
            elements = (element, element, "float", "float", element, element)
            readonly = 4
            constants = "ConstantBuffer<SDPAVJPTileParams> p : register(b6);"
            inputs = [buffers]
        else:
            sources, initial, _ = reduction._dataset(np, dtype, "fractional", 4)
            names = ("src", "acc", "out_")
            elements = (element, "float", element)
            readonly = 1
            constants = "\n".join(
                f"cbuffer {name}_Constants : register(b{slot}) {{ int {name}; }};"
                for slot, name in enumerate(reduction.SCALARS, 3)
            )
            inputs = [[source, *initial] for source in sources]
        for values in inputs:
            for value in values:
                if value.dtype == np.dtype("<f2"):
                    words = value.view("<u2").reshape(-1)
                    words[:] = np.resize(
                        np.array([0, 0x8000, 1, 0x7C01, 0x7FFF, 0xFC01], dtype="<u2"),
                        words.shape,
                    )
        encodings = {
            name: BINARY16_STORAGE
            for name, value in zip(names, inputs[0])
            if value.dtype == np.dtype("<f2")
        }
        declarations = [
            f"{'RW' if slot >= readonly else ''}StructuredBuffer<{physical}> {name} : register({'u' if slot >= readonly else 't'}{slot});"
            for slot, (name, physical) in enumerate(zip(names, elements))
        ]
        if damage == "missing":
            encodings = {}
        elif damage == "incomplete":
            encodings.pop(names[0])
        elif damage == "extra":
            encodings["unknown"] = BINARY16_STORAGE
        elif damage == "typed-half":
            declarations[0] = declarations[0].replace("uint16_t", "float16_t")
        elif damage == "register":
            declarations[0] = declarations[0].replace("register(t0)", "register(t15)")
        source = (resource_storage_header(encodings) if encodings else "") + "\n".join(
            [*declarations, constants]
        )
        artifact, module = tmp_path / "bindings.hlsl", tmp_path / "bindings.dxil"
        artifact.write_text(source, encoding="utf-8")
        module.write_bytes(b"binding-test-module")
        if family == "row-dot":
            requests = [
                row_dot._directx_request(np, buffers, artifact, module, dtype, 1)
            ]
        elif family == "derivative":
            requests = [
                derivative._directx_request(
                    np, artifact, module, buffers, parameters, dtype, True
                )
            ]
        else:
            modules = {
                "target": "directx",
                "dtype": dtype,
                "modules": {
                    mode: {
                        "artifact": artifact,
                        "module": module,
                        "reflection": {"resources": []},
                    }
                    for mode in ("set", "add")
                },
            }
            requests = reduction._requests(np, modules, sources, initial)
        return family, names, inputs, requests

    return build


@pytest.mark.parametrize("dtype", ("float", "float16_t", "bfloat16_t"))
def test_attention_bindings_preserve_native_storage(binding_case, dtype):
    family, names, inputs, requests = binding_case(dtype)
    for stage, request in enumerate(requests):
        bindings = {
            item.name: item for item in _prepare_directx_buffers(request.buffers)
        }
        assert len(bindings) == len(names)
        for slot, (name, value) in enumerate(zip(names, inputs[stage])):
            binding = bindings["out" if name == "out_" else name]
            assert binding.binding_index == slot
            assert binding.stride == value.itemsize
            uploaded = family != "reduction" or slot == 0 or stage == 0
            assert binding.upload == uploaded
            if uploaded:
                expected = value.tobytes() + row_dot.GUARD
                if family == "derivative":
                    expected += b"\x00" * (-len(expected) % 4)
                assert binding.payload == expected
    if family == "derivative":
        bindings = requests[0].buffers
        assert (
            bindings["S"].allocation.allocation_id
            == bindings["dS"].allocation.allocation_id
            == "scores"
        )
    elif family == "reduction":
        for name in ("acc", "out"):
            assert {
                item.buffers[name].allocation.allocation_id for item in requests
            } == {name}
            assert [item.buffers[name].source for item in requests] == ["input"] * 3 + [
                "expectedOutput"
            ]


@pytest.mark.parametrize(
    "damage", ("missing", "incomplete", "extra", "typed-half", "register")
)
def test_attention_half_bindings_reject_mismatched_contract(binding_case, damage):
    with pytest.raises(AssertionError):
        binding_case("float16_t", damage)


@pytest.mark.parametrize("dtype", ("float", "bfloat16_t"))
def test_attention_non_half_bindings_reject_half_metadata(binding_case, dtype):
    with pytest.raises(AssertionError):
        binding_case(dtype, "extra")
