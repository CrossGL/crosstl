import pytest

from crosstl.backend.HIP.HipAst import FunctionCallNode, UnaryOpNode
from crosstl.backend.HIP.HipCrossGLCodeGen import HipToCrossGLConverter


def addr(name):
    return UnaryOpNode("&", name)


@pytest.mark.parametrize(
    "name,args,success_value,comment",
    [
        (
            "hipDevicePrimaryCtxGetState",
            ["device", addr("flags"), addr("active")],
            "hipSuccess",
            "HIP primary context get state: device: device, flags output: flags, "
            "active output: active",
        ),
        (
            "hipMemcpyBatchAsync",
            [
                "dsts",
                "srcs",
                "sizes",
                "count",
                "attrs",
                "attrIndices",
                "attrCount",
                addr("failIndex"),
                "stream",
            ],
            "hipSuccess",
            "HIP batched memory copy: destinations: dsts, sources: srcs, "
            "sizes: sizes, count: count, attributes: attrs, attribute indices: "
            "attrIndices, attribute count: attrCount, fail index output: failIndex, "
            "stream: stream",
        ),
        (
            "hipMemsetD2D32Async",
            ["dst", "pitch", "value", "width", "height", "stream"],
            "hipSuccess",
            "HIP driver 2D memory set 32-bit: pointer: dst, pitch: pitch, "
            "value: value, width: width, height: height, stream: stream",
        ),
        (
            "hiprtcGetLoweredName",
            ["program", '"kernel"', addr("lowered")],
            "HIPRTC_SUCCESS",
            'HIPRTC get lowered name: program: program, expression: "kernel", '
            "output: lowered",
        ),
        (
            "hipGraphAddExternalSemaphoresWaitNode",
            [addr("node"), "graph", "deps", "count", "params"],
            "hipSuccess",
            "HIP graph add external semaphore wait node: output: node, "
            "graph: graph, dependencies: deps, count: count, params: params",
        ),
        (
            "hipGetSurfaceObjectResourceDesc",
            [addr("desc"), "surface"],
            "hipErrorNotSupported",
            "HIP surface object resource descriptor query not supported by HIP "
            "runtime: surface: surface, output: desc",
        ),
    ],
)
def test_runtime_status_expression_reuses_statement_metadata(
    name, args, success_value, comment
):
    converter = HipToCrossGLConverter()
    node = FunctionCallNode(name, args)

    assert converter.format_hip_runtime_call(node) == [f"// {comment}"]

    visited_args = [converter.visit(arg) for arg in args]
    expression = converter.format_hip_runtime_expression_call(node, visited_args)

    assert expression == f"(/* {comment} */ {success_value})"
    assert f"{name}(" not in expression


@pytest.mark.parametrize(
    "name,args,expected",
    [
        ("hipGetErrorString", ["err"], '/* HIP error string: err */ ""'),
        ("hipGetErrorName", ["err"], '/* HIP error name: err */ ""'),
        ("hiprtcGetErrorString", ["rtc"], '/* HIPRTC error string: rtc */ ""'),
        ("hipApiName", ["api"], '/* HIP API name: api */ ""'),
        (
            "hipKernelNameRefByPtr",
            ["hostFunc", "stream"],
            '/* HIP kernel name for host function: hostFunc, stream: stream */ ""',
        ),
        ("hipGetStreamDeviceId", ["stream"], "/* HIP stream device id: stream */ 0"),
    ],
)
def test_non_status_runtime_helpers_keep_value_fallbacks(name, args, expected):
    converter = HipToCrossGLConverter()
    node = FunctionCallNode(name, args)

    expression = converter.format_hip_runtime_expression_call(node, args)

    assert expression == expected
    assert f"{name}(" not in expression
