"""Optional hipcc-backed smoke tests for native HIP frontend samples."""

import shutil
import subprocess

import pytest

from crosstl.backend.HIP.HipCrossGLCodeGen import HipToCrossGLConverter
from crosstl.backend.HIP.HipLexer import HipLexer
from crosstl.backend.HIP.HipParser import HipParser


def compile_hip_if_hipcc_available(hip_code, tmp_path):
    """Compile native HIP source when hipcc is available."""
    hipcc = shutil.which("hipcc")
    if hipcc is None:
        pytest.skip("hipcc is not installed")

    source_path = tmp_path / "native_smoke.hip"
    object_path = tmp_path / "native_smoke.o"
    source_path.write_text(hip_code, encoding="utf-8")

    result = subprocess.run(
        [hipcc, "-std=c++17", "-c", str(source_path), "-o", str(object_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + "\n\n" + hip_code


def convert_native_hip_to_crossgl(hip_code):
    tokens = HipLexer(hip_code).tokenize()
    ast = HipParser(tokens).parse()
    return HipToCrossGLConverter().generate(ast)


def test_native_hip_atomic_barrier_smoke_parses_and_compiles_if_available(tmp_path):
    """Smoke native HIP atomics, shared memory, barriers, and fences."""
    hip_code = """
    #include <hip/hip_runtime.h>

    __global__ void native_smoke(int* out, int* counter) {
        __shared__ int scratch[32];
        int lane = threadIdx.x;
        scratch[lane] = lane;
        __syncthreads();
        int old = atomicAdd(counter, 1);
        __threadfence();
        out[lane] = old + scratch[lane];
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Kernel: native_smoke" in crossgl
    assert "var<workgroup> scratch: array<i32, 32>;" in crossgl
    assert "var lane: i32 = gl_LocalInvocationID.x;" in crossgl
    assert "scratch[lane] = lane;" in crossgl
    assert "workgroupBarrier();" in crossgl
    assert "var old: i32 = atomicAdd(counter, 1);" in crossgl
    assert "memoryBarrier();" in crossgl
    assert "out[lane] = (old + scratch[lane]);" in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_vector_constructor_smoke_parses_and_compiles_if_available(
    tmp_path,
):
    """Smoke native HIP vector constructors and vector member reads."""
    hip_code = """
    #include <hip/hip_runtime.h>

    __global__ void vector_smoke(float* out) {
        float2 uv = make_float2(1.0f, 2.0f);
        float4 color = make_float4(uv.x, uv.y, 3.0f, 4.0f);
        out[threadIdx.x] = color.w;
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Kernel: vector_smoke" in crossgl
    assert "var uv: vec2<f32> = vec2<f32>(1.0f, 2.0f);" in crossgl
    assert "var color: vec4<f32> = vec4<f32>(uv.x, uv.y, 3.0f, 4.0f);" in crossgl
    assert "out[gl_LocalInvocationID.x] = color.w;" in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_texture_surface_smoke_parses_and_compiles_if_available(
    tmp_path,
):
    """Smoke native HIP texture and surface object access."""
    hip_code = """
    #include <hip/hip_runtime.h>

    __global__ void resource_smoke(hipTextureObject_t tex, hipSurfaceObject_t surf, float4* out) {
        float2 uv = make_float2(0.25f, 0.75f);
        int2 pixel = make_int2(threadIdx.x, 0);
        float4 sampled = tex2D<float4>(tex, uv.x, uv.y);
        float4 loaded = surf2Dread<float4>(
            surf,
            pixel.x * sizeof(float4),
            pixel.y
        );
        surf2Dwrite(sampled, surf, pixel.x * sizeof(float4), pixel.y);
        out[threadIdx.x] = loaded;
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Kernel: resource_smoke" in crossgl
    assert "sampler2D tex" in crossgl
    assert "image2D surf" in crossgl
    assert (
        "@group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>"
        in crossgl
    )
    assert "var uv: vec2<f32> = vec2<f32>(0.25f, 0.75f);" in crossgl
    assert "var pixel: vec2<i32> = vec2<i32>(gl_LocalInvocationID.x, 0);" in crossgl
    assert "var sampled: vec4<f32> = texture(tex, vec2<f32>(uv.x, uv.y));" in crossgl
    assert (
        "var loaded: vec4<f32> = imageLoad(surf, vec2<i32>(pixel.x, pixel.y));"
        in crossgl
    )
    assert "imageStore(surf, vec2<i32>(pixel.x, pixel.y), sampled);" in crossgl
    assert "out[gl_LocalInvocationID.x] = loaded;" in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_texture_fetch_modes_parse_and_compile_if_available(
    tmp_path,
):
    """Smoke native HIP texture fetch intrinsics across 1D, 2D, and 3D modes."""
    hip_code = """
    #include <hip/hip_runtime.h>

    __global__ void texture_fetch_modes(
        hipTextureObject_t ramp,
        hipTextureObject_t tex,
        hipTextureObject_t volume,
        float4* out
    ) {
        float2 uv = make_float2(0.25f, 0.75f);
        float3 uvw = make_float3(0.25f, 0.5f, 0.75f);
        float sample1D = tex1Dfetch<float>(ramp, threadIdx.x);
        float4 sample2D = tex2D<float4>(tex, uv.x, uv.y);
        float4 sample3D = tex3D<float4>(volume, uvw.x, uvw.y, uvw.z);
        out[threadIdx.x] = make_float4(
            sample1D,
            sample2D.x,
            sample3D.y,
            sample2D.w
        );
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Kernel: texture_fetch_modes" in crossgl
    assert "sampler1D ramp" in crossgl
    assert "sampler2D tex" in crossgl
    assert "sampler3D volume" in crossgl
    assert (
        "@group(0) @binding(3) var<storage, read_write> out: array<vec4<f32>>"
        in crossgl
    )
    assert "var uv: vec2<f32> = vec2<f32>(0.25f, 0.75f);" in crossgl
    assert "var uvw: vec3<f32> = vec3<f32>(0.25f, 0.5f, 0.75f);" in crossgl
    assert "var sample1D: f32 = texelFetch(ramp, gl_LocalInvocationID.x, 0);" in crossgl
    assert "var sample2D: vec4<f32> = texture(tex, vec2<f32>(uv.x, uv.y));" in crossgl
    assert (
        "var sample3D: vec4<f32> = texture("
        "volume, vec3<f32>(uvw.x, uvw.y, uvw.z));" in crossgl
    )
    assert (
        "out[gl_LocalInvocationID.x] = vec4<f32>("
        "sample1D, sample2D.x, sample3D.y, sample2D.w);" in crossgl
    )
    assert "tex1Dfetch<" not in crossgl
    assert "tex2D<" not in crossgl
    assert "tex3D<" not in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_texture_lod_grad_modes_parse_and_compile_if_available(
    tmp_path,
):
    """Smoke native HIP texture LOD and gradient intrinsics."""
    hip_code = """
    #include <hip/hip_runtime.h>

    __global__ void texture_lod_grad_modes(
        hipTextureObject_t ramp,
        hipTextureObject_t tex,
        hipTextureObject_t volume,
        float4* out
    ) {
        float u = 0.25f;
        float2 uv = make_float2(0.25f, 0.75f);
        float3 uvw = make_float3(0.25f, 0.5f, 0.75f);
        float du = 0.125f;
        float2 dxy = make_float2(0.125f, 0.25f);
        float4 dxyz = make_float4(0.125f, 0.25f, 0.5f, 0.0f);
        float4 rampLod = tex1DLod<float4>(ramp, u, 1.0f);
        float4 rampGrad = tex1DGrad<float4>(ramp, u, du, du);
        float4 texLod = tex2DLod<float4>(tex, uv.x, uv.y, 1.0f);
        float4 texGrad = tex2DGrad<float4>(tex, uv.x, uv.y, dxy, dxy);
        float4 volumeLod = tex3DLod<float4>(
            volume,
            uvw.x,
            uvw.y,
            uvw.z,
            1.0f
        );
        float4 volumeGrad = tex3DGrad<float4>(
            volume,
            uvw.x,
            uvw.y,
            uvw.z,
            dxyz,
            dxyz
        );
        out[threadIdx.x] = make_float4(
            rampLod.x + rampGrad.x,
            texLod.y + texGrad.y,
            volumeLod.z + volumeGrad.z,
            volumeGrad.w
        );
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Kernel: texture_lod_grad_modes" in crossgl
    assert "sampler1D ramp" in crossgl
    assert "sampler2D tex" in crossgl
    assert "sampler3D volume" in crossgl
    assert (
        "@group(0) @binding(3) var<storage, read_write> out: array<vec4<f32>>"
        in crossgl
    )
    assert "var u: f32 = 0.25f;" in crossgl
    assert "var uv: vec2<f32> = vec2<f32>(0.25f, 0.75f);" in crossgl
    assert "var uvw: vec3<f32> = vec3<f32>(0.25f, 0.5f, 0.75f);" in crossgl
    assert "var du: f32 = 0.125f;" in crossgl
    assert "var dxy: vec2<f32> = vec2<f32>(0.125f, 0.25f);" in crossgl
    assert "var dxyz: vec4<f32> = vec4<f32>(0.125f, 0.25f, 0.5f, 0.0f);" in crossgl
    assert "var rampLod: vec4<f32> = textureLod(ramp, u, 1.0f);" in crossgl
    assert "var rampGrad: vec4<f32> = textureGrad(ramp, u, du, du);" in crossgl
    assert (
        "var texLod: vec4<f32> = textureLod("
        "tex, vec2<f32>(uv.x, uv.y), 1.0f);" in crossgl
    )
    assert (
        "var texGrad: vec4<f32> = textureGrad("
        "tex, vec2<f32>(uv.x, uv.y), dxy, dxy);" in crossgl
    )
    assert (
        "var volumeLod: vec4<f32> = textureLod("
        "volume, vec3<f32>(uvw.x, uvw.y, uvw.z), 1.0f);" in crossgl
    )
    assert (
        "var volumeGrad: vec4<f32> = textureGrad("
        "volume, vec3<f32>(uvw.x, uvw.y, uvw.z), dxyz, dxyz);" in crossgl
    )
    assert (
        "out[gl_LocalInvocationID.x] = vec4<f32>("
        "(rampLod.x + rampGrad.x), (texLod.y + texGrad.y), "
        "(volumeLod.z + volumeGrad.z), volumeGrad.w);" in crossgl
    )
    assert "tex1DLod<" not in crossgl
    assert "tex1DGrad<" not in crossgl
    assert "tex2DLod<" not in crossgl
    assert "tex2DGrad<" not in crossgl
    assert "tex3DLod<" not in crossgl
    assert "tex3DGrad<" not in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_texture_layer_cube_modes_parse_and_compile_if_available(
    tmp_path,
):
    """Smoke native HIP layered and cubemap texture intrinsics."""
    hip_code = """
    #include <hip/hip_runtime.h>

    __global__ void texture_layer_cube_modes(
        hipTextureObject_t rampArray,
        hipTextureObject_t layers,
        hipTextureObject_t env,
        hipTextureObject_t probes,
        float4* out
    ) {
        float u = 0.25f;
        float layer = 1.0f;
        float2 uv = make_float2(0.25f, 0.75f);
        float3 dir = make_float3(0.25f, 0.5f, 0.75f);
        float du = 0.125f;
        float2 dxy = make_float2(0.125f, 0.25f);
        float4 dcube = make_float4(0.125f, 0.25f, 0.5f, 0.0f);
        float4 rampLayer = tex1DLayered<float4>(rampArray, u, layer);
        float4 rampLayerLod = tex1DLayeredLod<float4>(
            rampArray,
            u,
            layer,
            1.0f
        );
        float4 rampLayerGrad = tex1DLayeredGrad<float4>(
            rampArray,
            u,
            layer,
            du,
            du
        );
        float4 layerSample = tex2DLayered<float4>(layers, uv.x, uv.y, layer);
        float4 layerLod = tex2DLayeredLod<float4>(
            layers,
            uv.x,
            uv.y,
            layer,
            1.0f
        );
        float4 layerGrad = tex2DLayeredGrad<float4>(
            layers,
            uv.x,
            uv.y,
            layer,
            dxy,
            dxy
        );
        float4 cubeSample = texCubemap<float4>(env, dir.x, dir.y, dir.z);
        float4 cubeLod = texCubemapLod<float4>(
            env,
            dir.x,
            dir.y,
            dir.z,
            1.0f
        );
        float4 cubeGrad = texCubemapGrad<float4>(
            env,
            dir.x,
            dir.y,
            dir.z,
            dcube,
            dcube
        );
        float4 probeSample = texCubemapLayered<float4>(
            probes,
            dir.x,
            dir.y,
            dir.z,
            layer
        );
        float4 probeLod = texCubemapLayeredLod<float4>(
            probes,
            dir.x,
            dir.y,
            dir.z,
            layer,
            1.0f
        );
        float4 probeGrad = texCubemapLayeredGrad<float4>(
            probes,
            dir.x,
            dir.y,
            dir.z,
            layer,
            dcube,
            dcube
        );
        out[threadIdx.x] = make_float4(
            rampLayer.x + rampLayerLod.x + rampLayerGrad.x,
            layerSample.y + layerLod.y + layerGrad.y,
            cubeSample.z + cubeLod.z + cubeGrad.z,
            probeSample.w + probeLod.w + probeGrad.w
        );
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    expected_fragments = (
        "// Kernel: texture_layer_cube_modes",
        "sampler1DArray rampArray",
        "sampler2DArray layers",
        "samplerCube env",
        "samplerCubeArray probes",
        "@group(0) @binding(4) var<storage, read_write> out: array<vec4<f32>>",
        "var u: f32 = 0.25f;",
        "var layer: f32 = 1.0f;",
        "var uv: vec2<f32> = vec2<f32>(0.25f, 0.75f);",
        "var dir: vec3<f32> = vec3<f32>(0.25f, 0.5f, 0.75f);",
        "var du: f32 = 0.125f;",
        "var dxy: vec2<f32> = vec2<f32>(0.125f, 0.25f);",
        "var dcube: vec4<f32> = vec4<f32>(0.125f, 0.25f, 0.5f, 0.0f);",
        "var rampLayer: vec4<f32> = texture(rampArray, vec2<f32>(u, layer));",
        (
            "var rampLayerLod: vec4<f32> = textureLod("
            "rampArray, vec2<f32>(u, layer), 1.0f);"
        ),
        (
            "var rampLayerGrad: vec4<f32> = textureGrad("
            "rampArray, vec2<f32>(u, layer), du, du);"
        ),
        (
            "var layerSample: vec4<f32> = texture("
            "layers, vec3<f32>(uv.x, uv.y, layer));"
        ),
        (
            "var layerLod: vec4<f32> = textureLod("
            "layers, vec3<f32>(uv.x, uv.y, layer), 1.0f);"
        ),
        (
            "var layerGrad: vec4<f32> = textureGrad("
            "layers, vec3<f32>(uv.x, uv.y, layer), dxy, dxy);"
        ),
        (
            "var cubeSample: vec4<f32> = texture("
            "env, vec3<f32>(dir.x, dir.y, dir.z));"
        ),
        (
            "var cubeLod: vec4<f32> = textureLod("
            "env, vec3<f32>(dir.x, dir.y, dir.z), 1.0f);"
        ),
        (
            "var cubeGrad: vec4<f32> = textureGrad("
            "env, vec3<f32>(dir.x, dir.y, dir.z), dcube, dcube);"
        ),
        (
            "var probeSample: vec4<f32> = texture("
            "probes, vec4<f32>(dir.x, dir.y, dir.z, layer));"
        ),
        (
            "var probeLod: vec4<f32> = textureLod("
            "probes, vec4<f32>(dir.x, dir.y, dir.z, layer), 1.0f);"
        ),
        (
            "var probeGrad: vec4<f32> = textureGrad("
            "probes, vec4<f32>(dir.x, dir.y, dir.z, layer), dcube, dcube);"
        ),
        (
            "out[gl_LocalInvocationID.x] = vec4<f32>("
            "((rampLayer.x + rampLayerLod.x) + rampLayerGrad.x), "
            "((layerSample.y + layerLod.y) + layerGrad.y), "
            "((cubeSample.z + cubeLod.z) + cubeGrad.z), "
            "((probeSample.w + probeLod.w) + probeGrad.w));"
        ),
    )
    for fragment in expected_fragments:
        assert fragment in crossgl

    for raw_intrinsic in (
        "tex1DLayered<",
        "tex1DLayeredLod<",
        "tex1DLayeredGrad<",
        "tex2DLayered<",
        "tex2DLayeredLod<",
        "tex2DLayeredGrad<",
        "texCubemap<",
        "texCubemapLod<",
        "texCubemapGrad<",
        "texCubemapLayered<",
        "texCubemapLayeredLod<",
        "texCubemapLayeredGrad<",
    ):
        assert raw_intrinsic not in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_surface_layer_cube_modes_parse_and_compile_if_available(
    tmp_path,
):
    """Smoke native HIP layered and cubemap surface intrinsics."""
    hip_code = """
    #include <hip/hip_runtime.h>

    __global__ void surface_layer_cube_modes(
        hipSurfaceObject_t lineLayer,
        hipSurfaceObject_t layers,
        hipSurfaceObject_t cube,
        hipSurfaceObject_t cubeLayers,
        float4* out
    ) {
        int x = threadIdx.x;
        int y = 1;
        int face = 2;
        int layer = 3;
        float4 lineLoaded;
        float4 layerLoaded;
        float4 cubeLoaded;
        float4 probeLoaded;
        surf1DLayeredread<float4>(
            &lineLoaded,
            lineLayer,
            x * sizeof(float4),
            layer
        );
        surf2DLayeredread<float4>(
            &layerLoaded,
            layers,
            x * sizeof(float4),
            y,
            layer
        );
        surfCubemapread<float4>(
            &cubeLoaded,
            cube,
            x * sizeof(float4),
            y,
            face
        );
        surfCubemapLayeredread<float4>(
            &probeLoaded,
            cubeLayers,
            x * sizeof(float4),
            y,
            face,
            layer
        );
        surf1DLayeredwrite(lineLoaded, lineLayer, x * sizeof(float4), layer);
        surf2DLayeredwrite(layerLoaded, layers, x * sizeof(float4), y, layer);
        surfCubemapwrite(cubeLoaded, cube, x * sizeof(float4), y, face);
        surfCubemapLayeredwrite(
            &probeLoaded,
            cubeLayers,
            x * sizeof(float4),
            y,
            face,
            layer
        );
        out[threadIdx.x] = make_float4(
            lineLoaded.x,
            layerLoaded.y,
            cubeLoaded.z,
            probeLoaded.w
        );
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    expected_fragments = (
        "// Kernel: surface_layer_cube_modes",
        "image1DArray lineLayer",
        "image2DArray layers",
        "imageCube cube",
        "imageCubeArray cubeLayers",
        "@group(0) @binding(4) var<storage, read_write> out: array<vec4<f32>>",
        "var x: i32 = gl_LocalInvocationID.x;",
        "var y: i32 = 1;",
        "var face: i32 = 2;",
        "var layer: i32 = 3;",
        "var lineLoaded: vec4<f32>;",
        "var layerLoaded: vec4<f32>;",
        "var cubeLoaded: vec4<f32>;",
        "var probeLoaded: vec4<f32>;",
        "lineLoaded = imageLoad(lineLayer, vec2<i32>(x, layer));",
        "layerLoaded = imageLoad(layers, vec3<i32>(x, y, layer));",
        "cubeLoaded = imageLoad(cube, vec3<i32>(x, y, face));",
        "probeLoaded = imageLoad(cubeLayers, vec4<i32>(x, y, face, layer));",
        "imageStore(lineLayer, vec2<i32>(x, layer), lineLoaded);",
        "imageStore(layers, vec3<i32>(x, y, layer), layerLoaded);",
        "imageStore(cube, vec3<i32>(x, y, face), cubeLoaded);",
        "imageStore(cubeLayers, vec4<i32>(x, y, face, layer), probeLoaded);",
        (
            "out[gl_LocalInvocationID.x] = vec4<f32>("
            "lineLoaded.x, layerLoaded.y, cubeLoaded.z, probeLoaded.w);"
        ),
    )
    for fragment in expected_fragments:
        assert fragment in crossgl

    for raw_intrinsic in (
        "surf1DLayeredread",
        "surf1DLayeredwrite",
        "surf2DLayeredread",
        "surf2DLayeredwrite",
        "surfCubemapread",
        "surfCubemapwrite",
        "surfCubemapLayeredread",
        "surfCubemapLayeredwrite",
    ):
        assert raw_intrinsic not in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_texture_surface_lifecycle_parses_and_compiles_if_available(
    tmp_path,
):
    """Smoke native HIP texture and surface object lifecycle APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void resource_lifecycle(
        hipResourceDesc* resourceDesc,
        hipTextureDesc* textureDesc
    ) {
        hipTextureObject_t tex = 0;
        hipSurfaceObject_t surf = 0;
        hipCreateTextureObject(&tex, resourceDesc, textureDesc, NULL);
        hipCreateSurfaceObject(&surf, resourceDesc);
        hipDestroyTextureObject(tex);
        hipDestroySurfaceObject(surf);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Function: resource_lifecycle" in crossgl
    assert (
        "void resource_lifecycle("
        "ptr<hipResourceDesc> resourceDesc, ptr<hipTextureDesc> textureDesc)" in crossgl
    )
    assert "var tex: sampler = 0;" in crossgl
    assert "var surf: image2D = 0;" in crossgl
    assert (
        "// HIP texture object create: tex, resource: resourceDesc, "
        "texture desc: textureDesc, resource view: NULL"
    ) in crossgl
    assert "// HIP surface object create: surf, resource: resourceDesc" in crossgl
    assert "// HIP texture object destroy: tex" in crossgl
    assert "// HIP surface object destroy: surf" in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_array_backed_texture_descriptor_setup_parses_and_compiles_if_available(
    tmp_path,
):
    """Smoke native HIP array-backed texture-object descriptor setup."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void array_backed_texture_descriptor_setup(hipArray_t array) {
        hipTextureObject_t tex = 0;
        hipResourceDesc resource_desc = {0};
        hipTextureDesc texture_desc = {0};

        resource_desc.resType = hipResourceTypeArray;
        resource_desc.res.array.array = array;
        texture_desc.addressMode[0] = hipAddressModeClamp;
        texture_desc.addressMode[1] = hipAddressModeClamp;
        texture_desc.filterMode = hipFilterModePoint;
        texture_desc.readMode = hipReadModeElementType;
        texture_desc.normalizedCoords = 0;
        hipCreateTextureObject(&tex, &resource_desc, &texture_desc, NULL);
        hipDestroyTextureObject(tex);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    expected_fragments = (
        "// Function: array_backed_texture_descriptor_setup",
        "void array_backed_texture_descriptor_setup(ptr<void> array)",
        "var tex: sampler = 0;",
        "var resource_desc: hipResourceDesc = {0};",
        "var texture_desc: hipTextureDesc = {0};",
        "resource_desc.resType = hipResourceTypeArray;",
        "resource_desc.res.array.array = array;",
        "texture_desc.addressMode[0] = hipAddressModeClamp;",
        "texture_desc.addressMode[1] = hipAddressModeClamp;",
        "texture_desc.filterMode = hipFilterModePoint;",
        "texture_desc.readMode = hipReadModeElementType;",
        "texture_desc.normalizedCoords = 0;",
        "// HIP texture object create: tex, resource: (&resource_desc), "
        "texture desc: (&texture_desc), resource view: NULL",
        "// HIP texture object destroy: tex",
    )
    for expected in expected_fragments:
        assert expected in crossgl

    raw_calls = (
        "hipCreateTextureObject",
        "hipDestroyTextureObject",
    )
    for raw_call in raw_calls:
        assert f"{raw_call}(" not in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_texture_object_descriptor_queries_parse_and_compile_if_available(
    tmp_path,
):
    """Smoke native HIP texture-object descriptor query APIs and aliases."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void texture_object_descriptor_queries(
        hipResourceDesc* resource_desc,
        hipTextureDesc* texture_desc,
        hipResourceViewDesc* view_desc
    ) {
        hipTextureObject_t tex = 0;
        hipTextureObject_t alias_tex = 0;
        hipResourceDesc resource_out;
        hipTextureDesc texture_out;
        hipResourceViewDesc view_out;
        hipResourceDesc alias_resource_out;
        hipTextureDesc alias_texture_out;
        hipResourceViewDesc alias_view_out;

        hipCreateTextureObject(&tex, resource_desc, texture_desc, view_desc);
        hipTexObjectCreate(&alias_tex, resource_desc, texture_desc, view_desc);
        hipGetTextureObjectResourceDesc(&resource_out, tex);
        hipGetTextureObjectTextureDesc(&texture_out, tex);
        hipGetTextureObjectResourceViewDesc(&view_out, tex);
        hipTexObjectGetResourceDesc(&alias_resource_out, alias_tex);
        hipTexObjectGetTextureDesc(&alias_texture_out, alias_tex);
        hipTexObjectGetResourceViewDesc(&alias_view_out, alias_tex);
        hipDestroyTextureObject(tex);
        hipTexObjectDestroy(alias_tex);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    expected_fragments = (
        "// Function: texture_object_descriptor_queries",
        "void texture_object_descriptor_queries("
        "ptr<hipResourceDesc> resource_desc, "
        "ptr<hipTextureDesc> texture_desc, "
        "ptr<hipResourceViewDesc> view_desc)",
        "var tex: sampler = 0;",
        "var alias_tex: sampler = 0;",
        "var resource_out: hipResourceDesc;",
        "var texture_out: hipTextureDesc;",
        "var view_out: hipResourceViewDesc;",
        "var alias_resource_out: hipResourceDesc;",
        "var alias_texture_out: hipTextureDesc;",
        "var alias_view_out: hipResourceViewDesc;",
        "// HIP texture object create: tex, resource: resource_desc, "
        "texture desc: texture_desc, resource view: view_desc",
        "// HIP texture object create: alias_tex, resource: resource_desc, "
        "texture desc: texture_desc, resource view: view_desc",
        "// HIP texture object get resource desc: output: resource_out, "
        "texture: tex",
        "// HIP texture object get texture desc: output: texture_out, " "texture: tex",
        "// HIP texture object get resource view desc: output: view_out, "
        "texture: tex",
        "// HIP texture object get resource desc: output: alias_resource_out, "
        "texture: alias_tex",
        "// HIP texture object get texture desc: output: alias_texture_out, "
        "texture: alias_tex",
        "// HIP texture object get resource view desc: output: alias_view_out, "
        "texture: alias_tex",
        "// HIP texture object destroy: tex",
        "// HIP texture object destroy: alias_tex",
    )
    for expected in expected_fragments:
        assert expected in crossgl

    raw_calls = (
        "hipCreateTextureObject",
        "hipTexObjectCreate",
        "hipGetTextureObjectResourceDesc",
        "hipGetTextureObjectTextureDesc",
        "hipGetTextureObjectResourceViewDesc",
        "hipTexObjectGetResourceDesc",
        "hipTexObjectGetTextureDesc",
        "hipTexObjectGetResourceViewDesc",
        "hipDestroyTextureObject",
        "hipTexObjectDestroy",
    )
    for raw_call in raw_calls:
        assert f"{raw_call}(" not in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_resource_view_descriptor_edges_parse_and_compile_if_available(
    tmp_path,
):
    """Smoke HIP resource-view descriptor edge fields for texture objects."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void texture_resource_view_descriptor_edges(
        hipMipmappedArray_t mipmapped_array
    ) {
        hipTextureObject_t tex = 0;
        hipResourceDesc resource_desc = {0};
        hipTextureDesc texture_desc = {0};
        hipResourceViewDesc view_desc = {0};
        hipResourceViewDesc queried_view_desc;

        resource_desc.resType = hipResourceTypeMipmappedArray;
        resource_desc.res.mipmap.mipmap = mipmapped_array;
        texture_desc.addressMode[0] = hipAddressModeClamp;
        texture_desc.addressMode[1] = hipAddressModeBorder;
        texture_desc.filterMode = hipFilterModeLinear;
        texture_desc.normalizedCoords = 1;
        view_desc.format = hipResViewFormatFloat4;
        view_desc.width = 128;
        view_desc.height = 64;
        view_desc.depth = 1;
        view_desc.firstMipmapLevel = 1;
        view_desc.lastMipmapLevel = 3;
        view_desc.firstLayer = 2;
        view_desc.lastLayer = 5;

        hipCreateTextureObject(&tex, &resource_desc, &texture_desc, &view_desc);
        hipGetTextureObjectResourceViewDesc(&queried_view_desc, tex);
        hipDestroyTextureObject(tex);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    expected_fragments = (
        "// Function: texture_resource_view_descriptor_edges",
        "void texture_resource_view_descriptor_edges(hipMipmappedArray_t mipmapped_array)",
        "var tex: sampler = 0;",
        "var resource_desc: hipResourceDesc = {0};",
        "var texture_desc: hipTextureDesc = {0};",
        "var view_desc: hipResourceViewDesc = {0};",
        "var queried_view_desc: hipResourceViewDesc;",
        "resource_desc.resType = hipResourceTypeMipmappedArray;",
        "resource_desc.res.mipmap.mipmap = mipmapped_array;",
        "texture_desc.addressMode[0] = hipAddressModeClamp;",
        "texture_desc.addressMode[1] = hipAddressModeBorder;",
        "texture_desc.filterMode = hipFilterModeLinear;",
        "texture_desc.normalizedCoords = 1;",
        "view_desc.format = hipResViewFormatFloat4;",
        "view_desc.width = 128;",
        "view_desc.height = 64;",
        "view_desc.depth = 1;",
        "view_desc.firstMipmapLevel = 1;",
        "view_desc.lastMipmapLevel = 3;",
        "view_desc.firstLayer = 2;",
        "view_desc.lastLayer = 5;",
        "// HIP texture object create: tex, resource: (&resource_desc), "
        "texture desc: (&texture_desc), resource view: (&view_desc)",
        "// HIP texture object get resource view desc: output: queried_view_desc, "
        "texture: tex",
        "// HIP texture object destroy: tex",
    )
    for expected in expected_fragments:
        assert expected in crossgl

    raw_calls = (
        "hipCreateTextureObject",
        "hipGetTextureObjectResourceViewDesc",
        "hipDestroyTextureObject",
    )
    for raw_call in raw_calls:
        assert f"{raw_call}(" not in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_memory_lifecycle_parses_and_compiles_if_available(tmp_path):
    """Smoke native HIP allocation, memset, copy, and free APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void memory_lifecycle(float* host, size_t n) {
        float* device = NULL;
        hipMalloc((void**)&device, n * sizeof(float));
        hipMemset(device, 0, n * sizeof(float));
        hipMemcpy(device, host, n * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(host, device, n * sizeof(float), hipMemcpyDeviceToHost);
        hipFree(device);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Function: memory_lifecycle" in crossgl
    assert "void memory_lifecycle(ptr<f32> host, u32 n)" in crossgl
    assert "var device: ptr<f32> = NULL;" in crossgl
    assert "// HIP memory allocate: device, bytes: (n * sizeof(float))" in crossgl
    assert "// HIP memory set: device, value: 0, bytes: (n * sizeof(float))" in crossgl
    assert (
        "// HIP memory copy: host -> device, bytes: (n * sizeof(float)), "
        "kind: hipMemcpyHostToDevice"
    ) in crossgl
    assert (
        "// HIP memory copy: device -> host, bytes: (n * sizeof(float)), "
        "kind: hipMemcpyDeviceToHost"
    ) in crossgl
    assert "// HIP memory free: device" in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_managed_async_memory_parses_and_compiles_if_available(
    tmp_path,
):
    """Smoke native HIP managed, async, prefetch, and advisory memory APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void managed_async_memory(float* host, size_t n, hipStream_t stream) {
        float* managed = NULL;
        float* async_ptr = NULL;
        unsigned long long access_flags = 0;
        hipMemRangeAttribute attribute = hipMemRangeAttributeAccessedBy;

        hipMallocManaged((void**)&managed, n * sizeof(float));
        hipMallocAsync((void**)&async_ptr, n * sizeof(float), stream);
        hipMemPrefetchAsync(managed, n * sizeof(float), 0, stream);
        hipMemAdvise(managed, n * sizeof(float), hipMemAdviseSetReadMostly, 0);
        hipMemRangeGetAttribute(
            &access_flags, sizeof(access_flags), attribute,
            managed, n * sizeof(float)
        );
        hipStreamAttachMemAsync(
            stream, managed, n * sizeof(float), hipMemAttachSingle
        );
        hipMemcpyAsync(
            async_ptr, host, n * sizeof(float), hipMemcpyHostToDevice, stream
        );
        hipFreeAsync(async_ptr, stream);
        hipFree(managed);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Function: managed_async_memory" in crossgl
    assert (
        "void managed_async_memory(ptr<f32> host, u32 n, hipStream_t stream)" in crossgl
    )
    assert "var managed: ptr<f32> = NULL;" in crossgl
    assert "var async_ptr: ptr<f32> = NULL;" in crossgl
    assert "var access_flags: u64 = 0;" in crossgl
    assert "var attribute: hipMemRangeAttribute = hipMemRangeAttributeAccessedBy;" in (
        crossgl
    )
    assert "// HIP memory allocate: managed, bytes: (n * sizeof(float))" in crossgl
    assert (
        "// HIP async memory allocate: async_ptr, bytes: (n * sizeof(float)), "
        "stream: stream"
    ) in crossgl
    assert (
        "// HIP memory prefetch: pointer: managed, bytes: (n * sizeof(float)), "
        "device: 0, stream: stream"
    ) in crossgl
    assert (
        "// HIP memory advise: pointer: managed, bytes: (n * sizeof(float)), "
        "advice: hipMemAdviseSetReadMostly, device: 0"
    ) in crossgl
    assert (
        "// HIP memory range get attribute: output: access_flags, "
        "output bytes: sizeof(access_flags), attribute: attribute, "
        "pointer: managed, range bytes: (n * sizeof(float))"
    ) in crossgl
    assert (
        "// HIP stream attach memory: stream: stream, pointer: managed, "
        "bytes: (n * sizeof(float)), flags: hipMemAttachSingle"
    ) in crossgl
    assert (
        "// HIP memory copy: host -> async_ptr, bytes: (n * sizeof(float)), "
        "kind: hipMemcpyHostToDevice, stream: stream"
    ) in crossgl
    assert "// HIP async memory free: async_ptr, stream: stream" in crossgl
    assert "// HIP memory free: managed" in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_memory_pool_virtual_memory_parses_and_compiles_if_available(
    tmp_path,
):
    """Smoke native HIP memory-pool and virtual-memory lifecycle APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void memory_pool_virtual_memory(size_t bytes, hipStream_t stream) {
        hipMemPool_t pool;
        hipMemPool_t imported_pool;
        hipMemPoolProps pool_props;
        hipMemPoolPtrExportData export_data;
        hipMemAccessDesc access_desc;
        hipMemLocation location;
        hipMemGenericAllocationHandle_t allocation_handle;
        hipMemGenericAllocationHandle_t retained_handle;
        hipMemAllocationProp allocation_prop;
        hipMemAllocationHandleType handle_type;
        void* shareable_handle = NULL;
        void* pooled_ptr = NULL;
        void* imported_ptr = NULL;
        void* virtual_address = NULL;
        size_t threshold = bytes;
        size_t granularity = 0;
        unsigned int pool_flags = 0;
        unsigned long long access_flags = 0;

        hipMemPoolCreate(&pool, &pool_props);
        hipDeviceGetDefaultMemPool(&pool, 0);
        hipDeviceGetMemPool(&pool, 0);
        hipMallocFromPoolAsync(&pooled_ptr, bytes, pool, stream);
        hipMemPoolTrimTo(pool, bytes);
        hipMemPoolSetAttribute(
            pool, hipMemPoolAttrReleaseThreshold, &threshold
        );
        hipMemPoolGetAttribute(
            pool, hipMemPoolAttrReleaseThreshold, &threshold
        );
        hipMemPoolSetAccess(pool, &access_desc, 1);
        hipMemPoolGetAccess(&pool_flags, pool, &location);
        hipMemPoolExportToShareableHandle(
            shareable_handle, pool, handle_type, 0
        );
        hipMemPoolImportFromShareableHandle(
            &imported_pool, shareable_handle, handle_type, 0
        );
        hipMemPoolExportPointer(&export_data, pooled_ptr);
        hipMemPoolImportPointer(&imported_ptr, imported_pool, &export_data);
        hipMemGetAllocationGranularity(
            &granularity, &allocation_prop, hipMemAllocationGranularityMinimum
        );
        hipMemCreate(&allocation_handle, bytes, &allocation_prop, 0);
        hipMemAddressReserve(&virtual_address, bytes, granularity, NULL, 0);
        hipMemMap(virtual_address, bytes, 0, allocation_handle, 0);
        hipMemSetAccess(virtual_address, bytes, &access_desc, 1);
        hipMemGetAccess(&access_flags, &location, virtual_address);
        hipMemGetAllocationPropertiesFromHandle(
            &allocation_prop, allocation_handle
        );
        hipMemRetainAllocationHandle(&retained_handle, virtual_address);
        hipMemUnmap(virtual_address, bytes);
        hipMemRelease(retained_handle);
        hipMemRelease(allocation_handle);
        hipMemAddressFree(virtual_address, bytes);
        hipFreeAsync(pooled_ptr, stream);
        hipMemPoolDestroy(pool);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Function: memory_pool_virtual_memory" in crossgl
    assert "void memory_pool_virtual_memory(u32 bytes, hipStream_t stream)" in crossgl
    assert "var pool: hipMemPool_t;" in crossgl
    assert "var imported_pool: hipMemPool_t;" in crossgl
    assert "var pool_props: hipMemPoolProps;" in crossgl
    assert "var export_data: hipMemPoolPtrExportData;" in crossgl
    assert "var access_desc: hipMemAccessDesc;" in crossgl
    assert "var location: hipMemLocation;" in crossgl
    assert "var allocation_handle: hipMemGenericAllocationHandle_t;" in crossgl
    assert "var retained_handle: hipMemGenericAllocationHandle_t;" in crossgl
    assert "var allocation_prop: hipMemAllocationProp;" in crossgl
    assert "var handle_type: hipMemAllocationHandleType;" in crossgl
    assert "var shareable_handle: ptr<void> = NULL;" in crossgl
    assert "var pooled_ptr: ptr<void> = NULL;" in crossgl
    assert "var imported_ptr: ptr<void> = NULL;" in crossgl
    assert "var virtual_address: ptr<void> = NULL;" in crossgl
    assert "var threshold: u32 = bytes;" in crossgl
    assert "var granularity: u32 = 0;" in crossgl
    assert "var pool_flags: u32 = 0;" in crossgl
    assert "var access_flags: u64 = 0;" in crossgl
    assert (
        "// HIP memory pool create: output: pool, properties: (&pool_props)" in crossgl
    )
    assert "// HIP get default memory pool: output: pool, device: 0" in crossgl
    assert "// HIP get device memory pool: output: pool, device: 0" in crossgl
    assert (
        "// HIP async memory allocate from pool: pooled_ptr, "
        "bytes: bytes, pool: pool, stream: stream"
    ) in crossgl
    assert "// HIP memory pool trim: pool: pool, minimum bytes: bytes" in crossgl
    assert (
        "// HIP memory pool set attribute: pool: pool, "
        "attribute: hipMemPoolAttrReleaseThreshold, value: threshold"
    ) in crossgl
    assert (
        "// HIP memory pool get attribute: pool: pool, "
        "attribute: hipMemPoolAttrReleaseThreshold, output: threshold"
    ) in crossgl
    assert (
        "// HIP memory pool set access: pool: pool, "
        "descriptors: (&access_desc), count: 1"
    ) in crossgl
    assert (
        "// HIP memory pool get access: output: pool_flags, "
        "pool: pool, location: (&location)"
    ) in crossgl
    assert (
        "// HIP memory pool export to shareable handle: "
        "output: shareable_handle, pool: pool, handle type: handle_type, "
        "flags: 0"
    ) in crossgl
    assert (
        "// HIP memory pool import from shareable handle: "
        "output: imported_pool, handle: shareable_handle, "
        "handle type: handle_type, flags: 0"
    ) in crossgl
    assert (
        "// HIP memory pool export pointer: output: export_data, " "pointer: pooled_ptr"
    ) in crossgl
    assert (
        "// HIP memory pool import pointer: output: imported_ptr, "
        "pool: imported_pool, export data: (&export_data)"
    ) in crossgl
    assert (
        "// HIP virtual memory allocation granularity: output: granularity, "
        "properties: (&allocation_prop), option: hipMemAllocationGranularityMinimum"
    ) in crossgl
    assert (
        "// HIP virtual memory create allocation: output: allocation_handle, "
        "bytes: bytes, properties: (&allocation_prop), flags: 0"
    ) in crossgl
    assert (
        "// HIP virtual memory reserve address: output: virtual_address, "
        "bytes: bytes, alignment: granularity, address: NULL, flags: 0"
    ) in crossgl
    assert (
        "// HIP virtual memory map: pointer: virtual_address, bytes: bytes, "
        "offset: 0, handle: allocation_handle, flags: 0"
    ) in crossgl
    assert (
        "// HIP virtual memory set access: pointer: virtual_address, "
        "bytes: bytes, descriptors: (&access_desc), count: 1"
    ) in crossgl
    assert (
        "// HIP virtual memory get access: output: access_flags, "
        "location: (&location), pointer: virtual_address"
    ) in crossgl
    assert (
        "// HIP virtual memory allocation properties: "
        "output: allocation_prop, handle: allocation_handle"
    ) in crossgl
    assert (
        "// HIP virtual memory retain allocation handle: "
        "output: retained_handle, address: virtual_address"
    ) in crossgl
    assert (
        "// HIP virtual memory unmap: pointer: virtual_address, bytes: bytes" in crossgl
    )
    assert "// HIP virtual memory release allocation: retained_handle" in crossgl
    assert "// HIP virtual memory release allocation: allocation_handle" in crossgl
    assert (
        "// HIP virtual memory free address: pointer: virtual_address, " "bytes: bytes"
    ) in crossgl
    assert "// HIP async memory free: pooled_ptr, stream: stream" in crossgl
    assert "// HIP memory pool destroy: pool" in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_external_interop_parses_and_compiles_if_available(tmp_path):
    """Smoke native HIP external memory and semaphore lifecycle APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void external_interop_lifecycle(hipStream_t stream) {
        hipExternalMemory_t external_memory;
        hipExternalMemoryHandleDesc memory_desc;
        hipExternalMemoryBufferDesc buffer_desc;
        hipExternalMemoryMipmappedArrayDesc mipmap_desc;
        hipMipmappedArray_t mipmapped_array;
        hipExternalSemaphore_t external_semaphore;
        hipExternalSemaphoreHandleDesc semaphore_desc;
        hipExternalSemaphoreSignalParams signal_params;
        hipExternalSemaphoreWaitParams wait_params;
        void* mapped_ptr = NULL;

        hipImportExternalMemory(&external_memory, &memory_desc);
        hipExternalMemoryGetMappedBuffer(
            &mapped_ptr, external_memory, &buffer_desc
        );
        hipExternalMemoryGetMappedMipmappedArray(
            &mipmapped_array, external_memory, &mipmap_desc
        );
        hipImportExternalSemaphore(&external_semaphore, &semaphore_desc);
        hipSignalExternalSemaphoresAsync(
            &external_semaphore, &signal_params, 1, stream
        );
        hipWaitExternalSemaphoresAsync(
            &external_semaphore, &wait_params, 1, stream
        );
        hipDestroyExternalSemaphore(external_semaphore);
        hipFreeMipmappedArray(mipmapped_array);
        hipDestroyExternalMemory(external_memory);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Function: external_interop_lifecycle" in crossgl
    assert "void external_interop_lifecycle(hipStream_t stream)" in crossgl
    assert "var external_memory: hipExternalMemory_t;" in crossgl
    assert "var memory_desc: hipExternalMemoryHandleDesc;" in crossgl
    assert "var buffer_desc: hipExternalMemoryBufferDesc;" in crossgl
    assert "var mipmap_desc: hipExternalMemoryMipmappedArrayDesc;" in crossgl
    assert "var mipmapped_array: hipMipmappedArray_t;" in crossgl
    assert "var external_semaphore: hipExternalSemaphore_t;" in crossgl
    assert "var semaphore_desc: hipExternalSemaphoreHandleDesc;" in crossgl
    assert "var signal_params: hipExternalSemaphoreSignalParams;" in crossgl
    assert "var wait_params: hipExternalSemaphoreWaitParams;" in crossgl
    assert "var mapped_ptr: ptr<void> = NULL;" in crossgl
    assert (
        "// HIP import external memory: output: external_memory, "
        "descriptor: (&memory_desc)"
    ) in crossgl
    assert (
        "// HIP external memory mapped buffer: output: mapped_ptr, "
        "memory: external_memory, descriptor: (&buffer_desc)"
    ) in crossgl
    assert (
        "// HIP external memory mapped mipmapped array: output: mipmapped_array, "
        "memory: external_memory, descriptor: (&mipmap_desc)"
    ) in crossgl
    assert (
        "// HIP import external semaphore: output: external_semaphore, "
        "descriptor: (&semaphore_desc)"
    ) in crossgl
    assert (
        "// HIP signal external semaphores: semaphores: (&external_semaphore), "
        "params: (&signal_params), count: 1, stream: stream"
    ) in crossgl
    assert (
        "// HIP wait external semaphores: semaphores: (&external_semaphore), "
        "params: (&wait_params), count: 1, stream: stream"
    ) in crossgl
    assert "// HIP destroy external semaphore: external_semaphore" in crossgl
    assert "// HIP free mipmapped array: mipmapped_array" in crossgl
    assert "// HIP destroy external memory: external_memory" in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_opengl_graphics_interop_parses_and_compiles_if_available(
    tmp_path,
):
    """Smoke native HIP OpenGL graphics interop lifecycle APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void graphics_interop_lifecycle(
        unsigned int gl_buffer,
        unsigned int gl_image,
        unsigned int gl_target,
        hipStream_t stream
    ) {
        hipGraphicsResource_t buffer_resource;
        hipGraphicsResource_t image_resource;
        hipGraphicsResource_t resources[2];
        hipArray_t mapped_array;
        void* mapped_ptr = NULL;
        size_t mapped_bytes = 0;

        hipGraphicsGLRegisterBuffer(
            &buffer_resource, gl_buffer, hipGraphicsRegisterFlagsWriteDiscard
        );
        hipGraphicsGLRegisterImage(
            &image_resource,
            gl_image,
            gl_target,
            hipGraphicsRegisterFlagsSurfaceLoadStore
        );
        resources[0] = buffer_resource;
        resources[1] = image_resource;
        hipGraphicsMapResources(2, resources, stream);
        hipGraphicsResourceGetMappedPointer(
            &mapped_ptr, &mapped_bytes, buffer_resource
        );
        hipGraphicsSubResourceGetMappedArray(
            &mapped_array, image_resource, 0, 0
        );
        hipGraphicsUnmapResources(2, resources, stream);
        hipGraphicsUnregisterResource(image_resource);
        hipGraphicsUnregisterResource(buffer_resource);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Function: graphics_interop_lifecycle" in crossgl
    assert (
        "void graphics_interop_lifecycle(u32 gl_buffer, u32 gl_image, "
        "u32 gl_target, hipStream_t stream)"
    ) in crossgl
    assert "var buffer_resource: hipGraphicsResource_t;" in crossgl
    assert "var image_resource: hipGraphicsResource_t;" in crossgl
    assert "var resources: array<hipGraphicsResource_t, 2>;" in crossgl
    assert "var mapped_array: ptr<void>;" in crossgl
    assert "var mapped_ptr: ptr<void> = NULL;" in crossgl
    assert "var mapped_bytes: u32 = 0;" in crossgl
    assert (
        "// HIP OpenGL register buffer: output: buffer_resource, "
        "buffer: gl_buffer, flags: hipGraphicsRegisterFlagsWriteDiscard"
    ) in crossgl
    assert (
        "// HIP OpenGL register image: output: image_resource, "
        "image: gl_image, target: gl_target, "
        "flags: hipGraphicsRegisterFlagsSurfaceLoadStore"
    ) in crossgl
    assert "resources[0] = buffer_resource;" in crossgl
    assert "resources[1] = image_resource;" in crossgl
    assert (
        "// HIP graphics map resources: count: 2, "
        "resources: resources, stream: stream"
    ) in crossgl
    assert (
        "// HIP graphics mapped pointer: pointer output: mapped_ptr, "
        "size output: mapped_bytes, resource: buffer_resource"
    ) in crossgl
    assert (
        "// HIP graphics mapped subresource array: output: mapped_array, "
        "resource: image_resource, array index: 0, mip level: 0"
    ) in crossgl
    assert (
        "// HIP graphics unmap resources: count: 2, "
        "resources: resources, stream: stream"
    ) in crossgl
    assert "// HIP graphics unregister resource: image_resource" in crossgl
    assert "// HIP graphics unregister resource: buffer_resource" in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_stream_event_lifecycle_parses_and_compiles_if_available(
    tmp_path,
):
    """Smoke native HIP stream and event lifecycle APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void stream_event_lifecycle(float* elapsed_ms) {
        hipStream_t stream;
        hipEvent_t start;
        hipEvent_t stop;
        hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
        hipEventCreate(&start);
        hipEventCreate(&stop);
        hipEventRecord(start, stream);
        hipStreamWaitEvent(stream, start, 0);
        hipEventRecord(stop, stream);
        hipStreamSynchronize(stream);
        hipEventSynchronize(stop);
        hipEventElapsedTime(elapsed_ms, start, stop);
        hipEventDestroy(start);
        hipEventDestroy(stop);
        hipStreamDestroy(stream);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Function: stream_event_lifecycle" in crossgl
    assert "void stream_event_lifecycle(ptr<f32> elapsed_ms)" in crossgl
    assert "var stream: hipStream_t;" in crossgl
    assert "var start: hipEvent_t;" in crossgl
    assert "var stop: hipEvent_t;" in crossgl
    assert "// HIP stream create: stream, flags: hipStreamNonBlocking" in crossgl
    assert "// HIP event create: start" in crossgl
    assert "// HIP event create: stop" in crossgl
    assert "// HIP event record: start, stream: stream" in crossgl
    assert "// HIP stream wait event: stream waits for start, flags: 0" in crossgl
    assert "// HIP event record: stop, stream: stream" in crossgl
    assert "// HIP synchronize: stream" in crossgl
    assert "// HIP event synchronize: stop" in crossgl
    assert "// HIP event elapsed time: start -> stop, output: elapsed_ms" in crossgl
    assert "// HIP event destroy: start" in crossgl
    assert "// HIP event destroy: stop" in crossgl
    assert "// HIP stream destroy: stream" in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_graph_lifecycle_parses_and_compiles_if_available(tmp_path):
    """Smoke native HIP graph capture, node creation, instantiate, and launch APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void graph_lifecycle(hipStream_t stream) {
        hipGraph_t graph;
        hipGraph_t captured_graph;
        hipGraphExec_t exec;
        hipGraphNode_t empty_node;
        hipGraphNode_t error_node;
        char log[128];
        hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
        hipStreamEndCapture(stream, &captured_graph);
        hipGraphDestroy(captured_graph);
        hipGraphCreate(&graph, 0);
        hipGraphAddEmptyNode(&empty_node, graph, NULL, 0);
        hipGraphInstantiate(&exec, graph, &error_node, log, 128);
        hipGraphLaunch(exec, stream);
        hipGraphExecDestroy(exec);
        hipGraphDestroy(graph);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Function: graph_lifecycle" in crossgl
    assert "void graph_lifecycle(hipStream_t stream)" in crossgl
    assert "var graph: hipGraph_t;" in crossgl
    assert "var captured_graph: hipGraph_t;" in crossgl
    assert "var exec: hipGraphExec_t;" in crossgl
    assert "var empty_node: hipGraphNode_t;" in crossgl
    assert "var error_node: hipGraphNode_t;" in crossgl
    assert "var log: array<i8, 128>;" in crossgl
    assert (
        "// HIP stream begin capture: stream: stream, "
        "mode: hipStreamCaptureModeGlobal"
    ) in crossgl
    assert (
        "// HIP stream end capture: stream: stream, graph output: captured_graph"
        in crossgl
    )
    assert "// HIP graph destroy: captured_graph" in crossgl
    assert "// HIP graph create: output: graph, flags: 0" in crossgl
    assert (
        "// HIP graph add empty node: output: empty_node, graph: graph, "
        "dependencies: NULL, count: 0"
    ) in crossgl
    assert (
        "// HIP graph instantiate: output: exec, graph: graph, "
        "error node output: error_node, log buffer: log, log bytes: 128"
    ) in crossgl
    assert "// HIP graph launch: exec: exec, stream: stream" in crossgl
    assert "// HIP graph exec destroy: exec" in crossgl
    assert "// HIP graph destroy: graph" in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_graph_exec_update_parses_and_compiles_if_available(tmp_path):
    """Smoke native HIP graph exec update and kernel-node parameter APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    __global__ void graph_update_kernel(float* out, float value) {
        out[threadIdx.x] = value;
    }

    void graph_exec_update(float* device_ptr) {
        hipGraph_t graph;
        hipGraphExec_t exec;
        hipGraphNode_t kernel_node;
        hipGraphNode_t error_node;
        hipGraphExecUpdateResult update_result;
        hipKernelNodeParams kernel_params;
        hipKernelNodeParams fetched_params;
        char log[128];

        hipGraphCreate(&graph, 0);
        hipGraphAddKernelNode(&kernel_node, graph, NULL, 0, &kernel_params);
        hipGraphKernelNodeGetParams(kernel_node, &fetched_params);
        hipGraphKernelNodeSetParams(kernel_node, &kernel_params);
        hipGraphInstantiate(&exec, graph, &error_node, log, 128);
        hipGraphExecKernelNodeSetParams(exec, kernel_node, &kernel_params);
        hipGraphExecUpdate(exec, graph, &error_node, &update_result);
        hipGraphExecDestroy(exec);
        hipGraphDestroy(graph);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Kernel: graph_update_kernel" in crossgl
    assert "void graph_exec_update(ptr<f32> device_ptr)" in crossgl
    assert "var graph: hipGraph_t;" in crossgl
    assert "var exec: hipGraphExec_t;" in crossgl
    assert "var kernel_node: hipGraphNode_t;" in crossgl
    assert "var error_node: hipGraphNode_t;" in crossgl
    assert "var update_result: hipGraphExecUpdateResult;" in crossgl
    assert "var kernel_params: hipKernelNodeParams;" in crossgl
    assert "var fetched_params: hipKernelNodeParams;" in crossgl
    assert "var log: array<i8, 128>;" in crossgl
    assert "// HIP graph create: output: graph, flags: 0" in crossgl
    assert (
        "// HIP graph add kernel node: output: kernel_node, graph: graph, "
        "dependencies: NULL, count: 0, params: (&kernel_params)"
    ) in crossgl
    assert (
        "// HIP graph kernel node get params: node: kernel_node, "
        "params: (&fetched_params)"
    ) in crossgl
    assert (
        "// HIP graph kernel node set params: node: kernel_node, "
        "params: (&kernel_params)"
    ) in crossgl
    assert (
        "// HIP graph instantiate: output: exec, graph: graph, "
        "error node output: error_node, log buffer: log, log bytes: 128"
    ) in crossgl
    assert (
        "// HIP graph exec set kernel node params: exec: exec, "
        "node: kernel_node, params: (&kernel_params)"
    ) in crossgl
    assert (
        "// HIP graph exec update: exec: exec, graph: graph, "
        "error node output: error_node, result output: update_result"
    ) in crossgl
    assert "// HIP graph exec destroy: exec" in crossgl
    assert "// HIP graph destroy: graph" in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_graph_dependency_queries_parse_and_compile_if_available(
    tmp_path,
):
    """Smoke native HIP graph dependency and query APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void graph_dependency_queries() {
        hipGraph_t graph;
        hipGraph_t clone;
        hipGraphNode_t root;
        hipGraphNode_t child;
        hipGraphNode_t cloned_child;
        hipGraphNode_t nodes[4];
        hipGraphNode_t edge_from[4];
        hipGraphNode_t edge_to[4];
        hipGraphNodeType node_type;
        size_t num_nodes = 4;
        size_t num_roots = 4;
        size_t num_edges = 4;
        size_t num_dependencies = 4;
        size_t num_dependents = 4;

        hipGraphCreate(&graph, 0);
        hipGraphAddEmptyNode(&root, graph, NULL, 0);
        hipGraphAddEmptyNode(&child, graph, NULL, 0);
        hipGraphAddDependencies(graph, &root, &child, 1);
        hipGraphGetNodes(graph, nodes, &num_nodes);
        hipGraphGetRootNodes(graph, nodes, &num_roots);
        hipGraphGetEdges(graph, edge_from, edge_to, &num_edges);
        hipGraphNodeGetDependencies(child, nodes, &num_dependencies);
        hipGraphNodeGetDependentNodes(root, nodes, &num_dependents);
        hipGraphNodeGetType(child, &node_type);
        hipGraphClone(&clone, graph);
        hipGraphNodeFindInClone(&cloned_child, child, clone);
        hipGraphRemoveDependencies(graph, &root, &child, 1);
        hipGraphDestroyNode(child);
        hipGraphDestroy(clone);
        hipGraphDestroy(graph);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Function: graph_dependency_queries" in crossgl
    assert "void graph_dependency_queries()" in crossgl
    assert "var graph: hipGraph_t;" in crossgl
    assert "var clone: hipGraph_t;" in crossgl
    assert "var root: hipGraphNode_t;" in crossgl
    assert "var child: hipGraphNode_t;" in crossgl
    assert "var cloned_child: hipGraphNode_t;" in crossgl
    assert "var nodes: array<hipGraphNode_t, 4>;" in crossgl
    assert "var edge_from: array<hipGraphNode_t, 4>;" in crossgl
    assert "var edge_to: array<hipGraphNode_t, 4>;" in crossgl
    assert "var node_type: hipGraphNodeType;" in crossgl
    assert "var num_nodes: u32 = 4;" in crossgl
    assert "var num_roots: u32 = 4;" in crossgl
    assert "var num_edges: u32 = 4;" in crossgl
    assert "var num_dependencies: u32 = 4;" in crossgl
    assert "var num_dependents: u32 = 4;" in crossgl
    assert "// HIP graph create: output: graph, flags: 0" in crossgl
    assert (
        "// HIP graph add empty node: output: root, graph: graph, "
        "dependencies: NULL, count: 0"
    ) in crossgl
    assert (
        "// HIP graph add empty node: output: child, graph: graph, "
        "dependencies: NULL, count: 0"
    ) in crossgl
    assert (
        "// HIP graph add dependencies: graph: graph, from: (&root), "
        "to: (&child), count: 1"
    ) in crossgl
    assert (
        "// HIP graph get nodes: graph: graph, nodes output: nodes, "
        "count output: num_nodes"
    ) in crossgl
    assert (
        "// HIP graph get root nodes: graph: graph, nodes output: nodes, "
        "count output: num_roots"
    ) in crossgl
    assert (
        "// HIP graph get edges: graph: graph, from output: edge_from, "
        "to output: edge_to, count output: num_edges"
    ) in crossgl
    assert (
        "// HIP graph node get dependencies: node: child, nodes output: nodes, "
        "count output: num_dependencies"
    ) in crossgl
    assert (
        "// HIP graph node get dependent nodes: node: root, nodes output: nodes, "
        "count output: num_dependents"
    ) in crossgl
    assert "// HIP graph node get type: node: child, output: node_type" in crossgl
    assert "// HIP graph clone: output: clone, source: graph" in crossgl
    assert (
        "// HIP graph node find in clone: output: cloned_child, "
        "original: child, clone graph: clone"
    ) in crossgl
    assert (
        "// HIP graph remove dependencies: graph: graph, from: (&root), "
        "to: (&child), count: 1"
    ) in crossgl
    assert "// HIP graph destroy node: child" in crossgl
    assert "// HIP graph destroy: clone" in crossgl
    assert "// HIP graph destroy: graph" in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_graph_event_nodes_parse_and_compile_if_available(tmp_path):
    """Smoke native HIP graph event record/wait node APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void graph_event_nodes_smoke(hipStream_t stream) {
        hipGraph_t graph;
        hipGraphExec_t exec;
        hipGraphNode_t record_node;
        hipGraphNode_t wait_node;
        hipGraphNode_t error_node;
        hipEvent_t event;
        char log[128];

        hipEventCreateWithFlags(&event, hipEventDisableTiming);
        hipGraphCreate(&graph, 0);
        hipGraphAddEventRecordNode(&record_node, graph, NULL, 0, event);
        hipGraphAddEventWaitNode(&wait_node, graph, &record_node, 1, event);
        hipGraphEventRecordNodeGetEvent(record_node, &event);
        hipGraphEventRecordNodeSetEvent(record_node, event);
        hipGraphEventWaitNodeGetEvent(wait_node, &event);
        hipGraphEventWaitNodeSetEvent(wait_node, event);
        hipGraphInstantiate(&exec, graph, &error_node, log, 128);
        hipGraphExecEventRecordNodeSetEvent(exec, record_node, event);
        hipGraphExecEventWaitNodeSetEvent(exec, wait_node, event);
        hipGraphExecDestroy(exec);
        hipGraphDestroyNode(wait_node);
        hipGraphDestroyNode(record_node);
        hipGraphDestroy(graph);
        hipEventDestroy(event);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    expected_fragments = (
        "// Function: graph_event_nodes_smoke",
        "void graph_event_nodes_smoke(hipStream_t stream)",
        "var graph: hipGraph_t;",
        "var exec: hipGraphExec_t;",
        "var record_node: hipGraphNode_t;",
        "var wait_node: hipGraphNode_t;",
        "var error_node: hipGraphNode_t;",
        "var event: hipEvent_t;",
        "var log: array<i8, 128>;",
        "// HIP event create: event, flags: hipEventDisableTiming",
        "// HIP graph create: output: graph, flags: 0",
        "// HIP graph add event record node: output: record_node, graph: graph, "
        "dependencies: NULL, count: 0, event: event",
        "// HIP graph add event wait node: output: wait_node, graph: graph, "
        "dependencies: (&record_node), count: 1, event: event",
        "// HIP graph event record node get event: node: record_node, output: event",
        "// HIP graph event record node set event: node: record_node, event: event",
        "// HIP graph event wait node get event: node: wait_node, output: event",
        "// HIP graph event wait node set event: node: wait_node, event: event",
        "// HIP graph instantiate: output: exec, graph: graph, "
        "error node output: error_node, log buffer: log, log bytes: 128",
        "// HIP graph exec event record node set event: exec: exec, "
        "node: record_node, event: event",
        "// HIP graph exec event wait node set event: exec: exec, "
        "node: wait_node, event: event",
        "// HIP graph exec destroy: exec",
        "// HIP graph destroy node: wait_node",
        "// HIP graph destroy node: record_node",
        "// HIP graph destroy: graph",
        "// HIP event destroy: event",
    )
    for expected in expected_fragments:
        assert expected in crossgl

    raw_calls = (
        "hipEventCreateWithFlags",
        "hipGraphCreate",
        "hipGraphAddEventRecordNode",
        "hipGraphAddEventWaitNode",
        "hipGraphEventRecordNodeGetEvent",
        "hipGraphEventRecordNodeSetEvent",
        "hipGraphEventWaitNodeGetEvent",
        "hipGraphEventWaitNodeSetEvent",
        "hipGraphInstantiate",
        "hipGraphExecEventRecordNodeSetEvent",
        "hipGraphExecEventWaitNodeSetEvent",
        "hipGraphExecDestroy",
        "hipGraphDestroyNode",
        "hipGraphDestroy",
        "hipEventDestroy",
    )
    for raw_call in raw_calls:
        assert f"{raw_call}(" not in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_graph_external_semaphore_nodes_parse_and_compile_if_available(
    tmp_path,
):
    """Smoke native HIP graph external semaphore signal/wait node APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void graph_external_semaphore_nodes_smoke(hipStream_t stream) {
        hipGraph_t graph;
        hipGraphExec_t exec;
        hipGraphNode_t signal_node;
        hipGraphNode_t wait_node;
        hipGraphNode_t error_node;
        hipExternalSemaphoreSignalNodeParams signal_params;
        hipExternalSemaphoreWaitNodeParams wait_params;
        char log[128];

        hipGraphCreate(&graph, 0);
        hipGraphAddExternalSemaphoresSignalNode(
            &signal_node, graph, NULL, 0, &signal_params
        );
        hipGraphExternalSemaphoresSignalNodeGetParams(
            signal_node, &signal_params
        );
        hipGraphExternalSemaphoresSignalNodeSetParams(
            signal_node, &signal_params
        );
        hipGraphAddExternalSemaphoresWaitNode(
            &wait_node, graph, &signal_node, 1, &wait_params
        );
        hipGraphExternalSemaphoresWaitNodeGetParams(wait_node, &wait_params);
        hipGraphExternalSemaphoresWaitNodeSetParams(wait_node, &wait_params);
        hipGraphInstantiate(&exec, graph, &error_node, log, 128);
        hipGraphExecExternalSemaphoresSignalNodeSetParams(
            exec, signal_node, &signal_params
        );
        hipGraphExecExternalSemaphoresWaitNodeSetParams(
            exec, wait_node, &wait_params
        );
        hipGraphExecDestroy(exec);
        hipGraphDestroyNode(wait_node);
        hipGraphDestroyNode(signal_node);
        hipGraphDestroy(graph);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    expected_fragments = (
        "// Function: graph_external_semaphore_nodes_smoke",
        "void graph_external_semaphore_nodes_smoke(hipStream_t stream)",
        "var graph: hipGraph_t;",
        "var exec: hipGraphExec_t;",
        "var signal_node: hipGraphNode_t;",
        "var wait_node: hipGraphNode_t;",
        "var error_node: hipGraphNode_t;",
        "var signal_params: hipExternalSemaphoreSignalNodeParams;",
        "var wait_params: hipExternalSemaphoreWaitNodeParams;",
        "var log: array<i8, 128>;",
        "// HIP graph create: output: graph, flags: 0",
        "// HIP graph add external semaphore signal node: "
        "output: signal_node, graph: graph, dependencies: NULL, "
        "count: 0, params: (&signal_params)",
        "// HIP graph external semaphore signal node get params: "
        "node: signal_node, params: (&signal_params)",
        "// HIP graph external semaphore signal node set params: "
        "node: signal_node, params: (&signal_params)",
        "// HIP graph add external semaphore wait node: "
        "output: wait_node, graph: graph, dependencies: (&signal_node), "
        "count: 1, params: (&wait_params)",
        "// HIP graph external semaphore wait node get params: "
        "node: wait_node, params: (&wait_params)",
        "// HIP graph external semaphore wait node set params: "
        "node: wait_node, params: (&wait_params)",
        "// HIP graph instantiate: output: exec, graph: graph, "
        "error node output: error_node, log buffer: log, log bytes: 128",
        "// HIP graph exec external semaphore signal node set params: "
        "exec: exec, node: signal_node, params: (&signal_params)",
        "// HIP graph exec external semaphore wait node set params: "
        "exec: exec, node: wait_node, params: (&wait_params)",
        "// HIP graph exec destroy: exec",
        "// HIP graph destroy node: wait_node",
        "// HIP graph destroy node: signal_node",
        "// HIP graph destroy: graph",
    )
    for expected in expected_fragments:
        assert expected in crossgl

    raw_calls = (
        "hipGraphCreate",
        "hipGraphAddExternalSemaphoresSignalNode",
        "hipGraphExternalSemaphoresSignalNodeGetParams",
        "hipGraphExternalSemaphoresSignalNodeSetParams",
        "hipGraphAddExternalSemaphoresWaitNode",
        "hipGraphExternalSemaphoresWaitNodeGetParams",
        "hipGraphExternalSemaphoresWaitNodeSetParams",
        "hipGraphInstantiate",
        "hipGraphExecExternalSemaphoresSignalNodeSetParams",
        "hipGraphExecExternalSemaphoresWaitNodeSetParams",
        "hipGraphExecDestroy",
        "hipGraphDestroyNode",
        "hipGraphDestroy",
    )
    for raw_call in raw_calls:
        assert f"{raw_call}(" not in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_graph_memory_user_objects_parse_and_compile_if_available(
    tmp_path,
):
    """Smoke native HIP graph memory attributes and user-object APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void user_object_destructor(void* resource) {
    }

    void graph_memory_user_object_smoke() {
        hipGraph_t graph;
        hipUserObject_t user_object;
        hipHostFn_t destructor = user_object_destructor;
        void* resource = NULL;
        size_t bytes = 0;
        int device = 0;

        hipGraphCreate(&graph, 0);
        hipDeviceGetGraphMemAttribute(
            device, hipGraphMemAttrUsedMemCurrent, &bytes
        );
        hipDeviceSetGraphMemAttribute(
            device, hipGraphMemAttrUsedMemCurrent, &bytes
        );
        hipDeviceGraphMemTrim(device);
        hipGraphDebugDotPrint(graph, "graph.dot", 0);
        hipUserObjectCreate(&user_object, resource, destructor, 1, 0);
        hipUserObjectRetain(user_object, 1);
        hipGraphRetainUserObject(graph, user_object, 1, 0);
        hipGraphReleaseUserObject(graph, user_object, 1);
        hipUserObjectRelease(user_object, 1);
        hipGraphDestroy(graph);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    expected_fragments = (
        "// Function: user_object_destructor",
        "void user_object_destructor(ptr<void> resource)",
        "// Function: graph_memory_user_object_smoke",
        "void graph_memory_user_object_smoke()",
        "var graph: hipGraph_t;",
        "var user_object: hipUserObject_t;",
        "var destructor: hipHostFn_t = user_object_destructor;",
        "var resource: ptr<void> = NULL;",
        "var bytes: u32 = 0;",
        "var device: i32 = 0;",
        "// HIP graph create: output: graph, flags: 0",
        "// HIP device graph memory get attribute: device: device, "
        "attribute: hipGraphMemAttrUsedMemCurrent, output: bytes",
        "// HIP device graph memory set attribute: device: device, "
        "attribute: hipGraphMemAttrUsedMemCurrent, value: bytes",
        "// HIP device graph memory trim: device: device",
        '// HIP graph debug dot print: graph: graph, path: "graph.dot", flags: 0',
        "// HIP user object create: output: user_object, resource: resource, "
        "destructor: destructor, initial refcount: 1, flags: 0",
        "// HIP user object retain: object: user_object, count: 1",
        "// HIP graph retain user object: graph: graph, object: user_object, "
        "count: 1, flags: 0",
        "// HIP graph release user object: graph: graph, object: user_object, "
        "count: 1",
        "// HIP user object release: object: user_object, count: 1",
        "// HIP graph destroy: graph",
    )
    for expected in expected_fragments:
        assert expected in crossgl

    raw_calls = (
        "hipGraphCreate",
        "hipDeviceGetGraphMemAttribute",
        "hipDeviceSetGraphMemAttribute",
        "hipDeviceGraphMemTrim",
        "hipGraphDebugDotPrint",
        "hipUserObjectCreate",
        "hipUserObjectRetain",
        "hipGraphRetainUserObject",
        "hipGraphReleaseUserObject",
        "hipUserObjectRelease",
        "hipGraphDestroy",
    )
    for raw_call in raw_calls:
        assert f"{raw_call}(" not in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_graph_memory_nodes_parse_and_compile_if_available(
    tmp_path,
):
    """Smoke native HIP graph memory allocation and free node APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void graph_memory_nodes() {
        hipGraph_t graph;
        hipGraphNode_t alloc_node;
        hipGraphNode_t free_node;
        hipMemAllocNodeParams alloc_params;
        hipMemAllocNodeParams fetched_alloc_params;
        void* device_ptr = NULL;

        hipGraphCreate(&graph, 0);
        hipGraphAddMemAllocNode(&alloc_node, graph, NULL, 0, &alloc_params);
        hipGraphMemAllocNodeGetParams(alloc_node, &fetched_alloc_params);
        hipGraphAddMemFreeNode(&free_node, graph, NULL, 0, device_ptr);
        hipGraphMemFreeNodeGetParams(free_node, &device_ptr);
        hipGraphDestroy(graph);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Function: graph_memory_nodes" in crossgl
    assert "void graph_memory_nodes()" in crossgl
    assert "var graph: hipGraph_t;" in crossgl
    assert "var alloc_node: hipGraphNode_t;" in crossgl
    assert "var free_node: hipGraphNode_t;" in crossgl
    assert "var alloc_params: hipMemAllocNodeParams;" in crossgl
    assert "var fetched_alloc_params: hipMemAllocNodeParams;" in crossgl
    assert "var device_ptr: ptr<void> = NULL;" in crossgl
    assert "// HIP graph create: output: graph, flags: 0" in crossgl
    assert (
        "// HIP graph add memory alloc node: output: alloc_node, "
        "graph: graph, dependencies: NULL, count: 0, params: (&alloc_params)"
    ) in crossgl
    assert (
        "// HIP graph memory alloc node get params: node: alloc_node, "
        "params output: fetched_alloc_params"
    ) in crossgl
    assert (
        "// HIP graph add memory free node: output: free_node, graph: graph, "
        "dependencies: NULL, count: 0, pointer: device_ptr"
    ) in crossgl
    assert (
        "// HIP graph memory free node get params: node: free_node, "
        "pointer output: device_ptr"
    ) in crossgl
    assert "// HIP graph destroy: graph" in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_graph_memcpy_nodes_parse_and_compile_if_available(
    tmp_path,
):
    """Smoke native HIP graph memcpy node add, query, set, and exec patch APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void graph_memcpy_nodes() {
        hipGraph_t graph;
        hipGraphExec_t exec;
        hipGraphNode_t generic_node;
        hipGraphNode_t copy_1d_node;
        hipGraphNode_t error_node;
        hipMemcpy3DParms copy_params;
        hipMemcpy3DParms fetched_copy_params;
        char log[128];
        void* dst = NULL;
        void* src = NULL;
        size_t bytes = 64;

        hipGraphCreate(&graph, 0);
        hipGraphAddMemcpyNode(&generic_node, graph, NULL, 0, &copy_params);
        hipGraphMemcpyNodeGetParams(generic_node, &fetched_copy_params);
        hipGraphMemcpyNodeSetParams(generic_node, &copy_params);
        hipGraphAddMemcpyNode1D(
            &copy_1d_node, graph, NULL, 0, dst, src, bytes,
            hipMemcpyDeviceToDevice
        );
        hipGraphMemcpyNodeSetParams1D(
            copy_1d_node, dst, src, bytes, hipMemcpyDeviceToDevice
        );
        hipGraphInstantiate(&exec, graph, &error_node, log, 128);
        hipGraphExecMemcpyNodeSetParams(exec, generic_node, &copy_params);
        hipGraphExecMemcpyNodeSetParams1D(
            exec, copy_1d_node, dst, src, bytes, hipMemcpyDeviceToDevice
        );
        hipGraphExecDestroy(exec);
        hipGraphDestroy(graph);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Function: graph_memcpy_nodes" in crossgl
    assert "void graph_memcpy_nodes()" in crossgl
    assert "var graph: hipGraph_t;" in crossgl
    assert "var exec: hipGraphExec_t;" in crossgl
    assert "var generic_node: hipGraphNode_t;" in crossgl
    assert "var copy_1d_node: hipGraphNode_t;" in crossgl
    assert "var error_node: hipGraphNode_t;" in crossgl
    assert "var copy_params: hipMemcpy3DParms;" in crossgl
    assert "var fetched_copy_params: hipMemcpy3DParms;" in crossgl
    assert "var log: array<i8, 128>;" in crossgl
    assert "var dst: ptr<void> = NULL;" in crossgl
    assert "var src: ptr<void> = NULL;" in crossgl
    assert "var bytes: u32 = 64;" in crossgl
    assert "// HIP graph create: output: graph, flags: 0" in crossgl
    assert (
        "// HIP graph add memcpy node: output: generic_node, graph: graph, "
        "dependencies: NULL, count: 0, params: (&copy_params)"
    ) in crossgl
    assert (
        "// HIP graph memcpy node get params: node: generic_node, "
        "params: (&fetched_copy_params)"
    ) in crossgl
    assert (
        "// HIP graph memcpy node set params: node: generic_node, "
        "params: (&copy_params)"
    ) in crossgl
    assert (
        "// HIP graph add memcpy 1D node: output: copy_1d_node, graph: graph, "
        "dependencies: NULL, count: 0, destination: dst, source: src, "
        "bytes: bytes, kind: hipMemcpyDeviceToDevice"
    ) in crossgl
    assert (
        "// HIP graph memcpy 1D node set params: node: copy_1d_node, "
        "destination: dst, source: src, bytes: bytes, kind: hipMemcpyDeviceToDevice"
    ) in crossgl
    assert (
        "// HIP graph instantiate: output: exec, graph: graph, "
        "error node output: error_node, log buffer: log, log bytes: 128"
    ) in crossgl
    assert (
        "// HIP graph exec set memcpy node params: exec: exec, "
        "node: generic_node, params: (&copy_params)"
    ) in crossgl
    assert (
        "// HIP graph exec memcpy 1D node set params: exec: exec, "
        "node: copy_1d_node, destination: dst, source: src, bytes: bytes, "
        "kind: hipMemcpyDeviceToDevice"
    ) in crossgl
    assert "// HIP graph exec destroy: exec" in crossgl
    assert "// HIP graph destroy: graph" in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_graph_symbol_memcpy_nodes_parse_and_compile_if_available(
    tmp_path,
):
    """Smoke native HIP graph symbol memcpy node add, set, and exec patch APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void graph_symbol_memcpy_nodes() {
        hipGraph_t graph;
        hipGraphExec_t exec;
        hipGraphNode_t from_symbol_node;
        hipGraphNode_t to_symbol_node;
        hipGraphNode_t error_node;
        void* dst = NULL;
        const void* symbol = NULL;
        const void* src = NULL;
        size_t bytes = 64;
        size_t offset = 4;
        char log[128];

        hipGraphCreate(&graph, 0);
        hipGraphAddMemcpyNodeFromSymbol(
            &from_symbol_node, graph, NULL, 0, dst, symbol, bytes, offset,
            hipMemcpyDeviceToDevice
        );
        hipGraphMemcpyNodeSetParamsFromSymbol(
            from_symbol_node, dst, symbol, bytes, offset, hipMemcpyDeviceToDevice
        );
        hipGraphAddMemcpyNodeToSymbol(
            &to_symbol_node, graph, NULL, 0, symbol, src, bytes, offset,
            hipMemcpyDeviceToDevice
        );
        hipGraphMemcpyNodeSetParamsToSymbol(
            to_symbol_node, symbol, src, bytes, offset, hipMemcpyDeviceToDevice
        );
        hipGraphInstantiate(&exec, graph, &error_node, log, 128);
        hipGraphExecMemcpyNodeSetParamsFromSymbol(
            exec, from_symbol_node, dst, symbol, bytes, offset,
            hipMemcpyDeviceToDevice
        );
        hipGraphExecMemcpyNodeSetParamsToSymbol(
            exec, to_symbol_node, symbol, src, bytes, offset,
            hipMemcpyDeviceToDevice
        );
        hipGraphExecDestroy(exec);
        hipGraphDestroy(graph);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Function: graph_symbol_memcpy_nodes" in crossgl
    assert "void graph_symbol_memcpy_nodes()" in crossgl
    assert "var graph: hipGraph_t;" in crossgl
    assert "var exec: hipGraphExec_t;" in crossgl
    assert "var from_symbol_node: hipGraphNode_t;" in crossgl
    assert "var to_symbol_node: hipGraphNode_t;" in crossgl
    assert "var error_node: hipGraphNode_t;" in crossgl
    assert "var dst: ptr<void> = NULL;" in crossgl
    assert "var symbol: ptr<void> = NULL;" in crossgl
    assert "var src: ptr<void> = NULL;" in crossgl
    assert "var bytes: u32 = 64;" in crossgl
    assert "var offset: u32 = 4;" in crossgl
    assert "var log: array<i8, 128>;" in crossgl
    assert "// HIP graph create: output: graph, flags: 0" in crossgl
    assert (
        "// HIP graph add memcpy from symbol node: output: from_symbol_node, "
        "graph: graph, dependencies: NULL, count: 0, destination: dst, "
        "source: symbol, bytes: bytes, offset: offset, "
        "kind: hipMemcpyDeviceToDevice"
    ) in crossgl
    assert (
        "// HIP graph memcpy from symbol node set params: "
        "node: from_symbol_node, destination: dst, source: symbol, "
        "bytes: bytes, offset: offset, kind: hipMemcpyDeviceToDevice"
    ) in crossgl
    assert (
        "// HIP graph add memcpy to symbol node: output: to_symbol_node, "
        "graph: graph, dependencies: NULL, count: 0, destination: symbol, "
        "source: src, bytes: bytes, offset: offset, kind: hipMemcpyDeviceToDevice"
    ) in crossgl
    assert (
        "// HIP graph memcpy to symbol node set params: node: to_symbol_node, "
        "destination: symbol, source: src, bytes: bytes, offset: offset, "
        "kind: hipMemcpyDeviceToDevice"
    ) in crossgl
    assert (
        "// HIP graph instantiate: output: exec, graph: graph, "
        "error node output: error_node, log buffer: log, log bytes: 128"
    ) in crossgl
    assert (
        "// HIP graph exec memcpy from symbol node set params: exec: exec, "
        "node: from_symbol_node, destination: dst, source: symbol, "
        "bytes: bytes, offset: offset, kind: hipMemcpyDeviceToDevice"
    ) in crossgl
    assert (
        "// HIP graph exec memcpy to symbol node set params: exec: exec, "
        "node: to_symbol_node, destination: symbol, source: src, bytes: bytes, "
        "offset: offset, kind: hipMemcpyDeviceToDevice"
    ) in crossgl
    assert "// HIP graph exec destroy: exec" in crossgl
    assert "// HIP graph destroy: graph" in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_graph_child_kernel_attributes_parse_and_compile_if_available(
    tmp_path,
):
    """Smoke native HIP graph child, kernel attribute, and enabled-state APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void graph_child_kernel_attributes(float* out) {
        hipGraph_t graph;
        hipGraph_t child_graph;
        hipGraph_t fetched_child_graph;
        hipGraphExec_t exec;
        hipGraphNode_t child_node;
        hipGraphNode_t source_kernel_node;
        hipGraphNode_t kernel_node;
        hipGraphNode_t error_node;
        hipKernelNodeParams source_kernel_params;
        hipKernelNodeParams kernel_params;
        hipKernelNodeAttrValue attr_value;
        unsigned int enabled = 0;
        char log[128];

        hipGraphCreate(&graph, 0);
        hipGraphCreate(&child_graph, 0);
        hipGraphAddChildGraphNode(&child_node, graph, NULL, 0, child_graph);
        hipGraphChildGraphNodeGetGraph(child_node, &fetched_child_graph);
        hipGraphAddKernelNode(
            &source_kernel_node, graph, NULL, 0, &source_kernel_params
        );
        hipGraphAddKernelNode(&kernel_node, graph, NULL, 0, &kernel_params);
        hipGraphKernelNodeCopyAttributes(kernel_node, source_kernel_node);
        hipGraphKernelNodeSetAttribute(
            kernel_node, hipKernelNodeAttributeCooperative, &attr_value
        );
        hipGraphKernelNodeGetAttribute(
            kernel_node, hipKernelNodeAttributeCooperative, &attr_value
        );
        hipGraphInstantiate(&exec, graph, &error_node, log, 128);
        hipGraphNodeSetEnabled(exec, kernel_node, 1);
        hipGraphNodeGetEnabled(exec, kernel_node, &enabled);
        hipGraphExecDestroy(exec);
        hipGraphDestroy(child_graph);
        hipGraphDestroy(graph);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Function: graph_child_kernel_attributes" in crossgl
    assert "void graph_child_kernel_attributes(ptr<f32> out)" in crossgl
    assert "var graph: hipGraph_t;" in crossgl
    assert "var child_graph: hipGraph_t;" in crossgl
    assert "var fetched_child_graph: hipGraph_t;" in crossgl
    assert "var exec: hipGraphExec_t;" in crossgl
    assert "var child_node: hipGraphNode_t;" in crossgl
    assert "var source_kernel_node: hipGraphNode_t;" in crossgl
    assert "var kernel_node: hipGraphNode_t;" in crossgl
    assert "var error_node: hipGraphNode_t;" in crossgl
    assert "var source_kernel_params: hipKernelNodeParams;" in crossgl
    assert "var kernel_params: hipKernelNodeParams;" in crossgl
    assert "var attr_value: hipKernelNodeAttrValue;" in crossgl
    assert "var enabled: u32 = 0;" in crossgl
    assert "var log: array<i8, 128>;" in crossgl
    assert crossgl.count("// HIP graph create: output:") == 2
    assert (
        "// HIP graph add child graph node: output: child_node, "
        "graph: graph, dependencies: NULL, count: 0, child graph: child_graph"
    ) in crossgl
    assert (
        "// HIP graph child node get graph: node: child_node, "
        "output: fetched_child_graph"
    ) in crossgl
    assert (
        "// HIP graph add kernel node: output: source_kernel_node, "
        "graph: graph, dependencies: NULL, count: 0, "
        "params: (&source_kernel_params)"
    ) in crossgl
    assert (
        "// HIP graph add kernel node: output: kernel_node, graph: graph, "
        "dependencies: NULL, count: 0, params: (&kernel_params)"
    ) in crossgl
    assert (
        "// HIP graph kernel node copy attributes: source: kernel_node, "
        "destination: source_kernel_node"
    ) in crossgl
    assert (
        "// HIP graph kernel node set attribute: node: kernel_node, "
        "attribute: hipKernelNodeAttributeCooperative, value: attr_value"
    ) in crossgl
    assert (
        "// HIP graph kernel node get attribute: node: kernel_node, "
        "attribute: hipKernelNodeAttributeCooperative, output: attr_value"
    ) in crossgl
    assert (
        "// HIP graph instantiate: output: exec, graph: graph, "
        "error node output: error_node, log buffer: log, log bytes: 128"
    ) in crossgl
    assert (
        "// HIP graph node set enabled: exec: exec, node: kernel_node, value: 1"
        in crossgl
    )
    assert (
        "// HIP graph node get enabled: exec: exec, node: kernel_node, "
        "output: enabled"
    ) in crossgl
    assert "// HIP graph exec destroy: exec" in crossgl
    assert "// HIP graph destroy: child_graph" in crossgl
    assert "// HIP graph destroy: graph" in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_device_error_runtime_parses_and_compiles_if_available(
    tmp_path,
):
    """Smoke native HIP device selection and error query APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void device_error_smoke(int* count_out) {
        int device = 0;
        int count = 0;
        hipDeviceProp_t props;
        hipGetDevice(&device);
        hipGetDeviceCount(&count);
        hipSetDevice(device);
        hipGetDeviceProperties(&props, device);
        hipDeviceSynchronize();
        hipError_t last = hipGetLastError();
        hipError_t peek = hipPeekAtLastError();
        const char* name = hipGetErrorName(last);
        const char* message = hipGetErrorString(peek);
        count_out[0] = count;
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Function: device_error_smoke" in crossgl
    assert "void device_error_smoke(ptr<i32> count_out)" in crossgl
    assert "var device: i32 = 0;" in crossgl
    assert "var count: i32 = 0;" in crossgl
    assert "var props: hipDeviceProp_t;" in crossgl
    assert "// HIP get current device: output: device" in crossgl
    assert "// HIP get device count: output: count" in crossgl
    assert "// HIP set device: device" in crossgl
    assert "// HIP get device properties: props, device: device" in crossgl
    assert "// HIP device synchronize" in crossgl
    assert "// HIP get last error" in crossgl
    assert "var last: hipError_t = hipSuccess;" in crossgl
    assert "// HIP peek at last error" in crossgl
    assert "var peek: hipError_t = hipSuccess;" in crossgl
    assert 'var name: ptr<i8> = /* HIP error name: last */ "";' in crossgl
    assert 'var message: ptr<i8> = /* HIP error string: peek */ "";' in crossgl
    assert "count_out[0] = count;" in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_peer_access_copy_parses_and_compiles_if_available(tmp_path):
    """Smoke native HIP peer access and peer copy APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void peer_access_copy(
        float* dst,
        float* src,
        hipStream_t stream,
        size_t n
    ) {
        int can_access = 0;
        int p2p_attribute = 0;
        int device = 0;
        int peer_device = 1;
        unsigned int link_type = 0;
        unsigned int hop_count = 0;
        size_t bytes = n * sizeof(float);

        hipDeviceCanAccessPeer(&can_access, device, peer_device);
        hipDeviceGetP2PAttribute(
            &p2p_attribute,
            hipDevP2PAttrPerformanceRank,
            device,
            peer_device
        );
        hipDeviceEnablePeerAccess(peer_device, 0);
        hipMemcpyPeer(dst, peer_device, src, device, bytes);
        hipMemcpyPeerAsync(dst, peer_device, src, device, bytes, stream);
        hipExtGetLinkTypeAndHopCount(
            device, peer_device, &link_type, &hop_count
        );
        hipDeviceDisablePeerAccess(peer_device);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    expected_fragments = (
        "// Function: peer_access_copy",
        "void peer_access_copy("
        "ptr<f32> dst, ptr<f32> src, hipStream_t stream, u32 n)",
        "var can_access: i32 = 0;",
        "var p2p_attribute: i32 = 0;",
        "var device: i32 = 0;",
        "var peer_device: i32 = 1;",
        "var link_type: u32 = 0;",
        "var hop_count: u32 = 0;",
        "var bytes: u32 = (n * sizeof(float));",
        "// HIP device can access peer: output: can_access, "
        "device: device, peer device: peer_device",
        "// HIP get P2P attribute: output: p2p_attribute, "
        "attribute: hipDevP2PAttrPerformanceRank, source device: device, "
        "destination device: peer_device",
        "// HIP enable peer access: peer device: peer_device, flags: 0",
        "// HIP peer memory copy: source: src, source device: device, "
        "destination: dst, destination device: peer_device, bytes: bytes",
        "// HIP peer memory copy: source: src, source device: device, "
        "destination: dst, destination device: peer_device, bytes: bytes, "
        "stream: stream",
        "// HIP get link type and hop count: device 1: device, "
        "device 2: peer_device, link type output: link_type, "
        "hop count output: hop_count",
        "// HIP disable peer access: peer device: peer_device",
    )
    for expected in expected_fragments:
        assert expected in crossgl

    raw_calls = (
        "hipDeviceCanAccessPeer",
        "hipDeviceGetP2PAttribute",
        "hipDeviceEnablePeerAccess",
        "hipMemcpyPeer",
        "hipMemcpyPeerAsync",
        "hipExtGetLinkTypeAndHopCount",
        "hipDeviceDisablePeerAccess",
    )
    for raw_call in raw_calls:
        assert f"{raw_call}(" not in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_host_pinned_memory_parses_and_compiles_if_available(
    tmp_path,
):
    """Smoke native HIP host-pinned allocation and registration APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void host_pinned_memory(float* registered, size_t n) {
        float* host = NULL;
        float* device = NULL;
        unsigned int flags = 0;
        hipHostMalloc((void**)&host, n * sizeof(float), hipHostMallocMapped);
        hipHostGetDevicePointer((void**)&device, host, 0);
        hipHostGetFlags(&flags, host);
        hipHostRegister(registered, n * sizeof(float), hipHostRegisterMapped);
        hipHostUnregister(registered);
        hipHostFree(host);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Function: host_pinned_memory" in crossgl
    assert "void host_pinned_memory(ptr<f32> registered, u32 n)" in crossgl
    assert "var host: ptr<f32> = NULL;" in crossgl
    assert "var device: ptr<f32> = NULL;" in crossgl
    assert "var flags: u32 = 0;" in crossgl
    assert (
        "// HIP host memory allocate: host, bytes: (n * sizeof(float)), "
        "flags: hipHostMallocMapped"
    ) in crossgl
    assert "// HIP host device pointer: output: device, host: host, flags: 0" in crossgl
    assert "// HIP host memory flags: output: flags, host: host" in crossgl
    assert (
        "// HIP host memory register: registered, bytes: (n * sizeof(float)), "
        "flags: hipHostRegisterMapped"
    ) in crossgl
    assert "// HIP host memory unregister: registered" in crossgl
    assert "// HIP memory free: host" in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_ipc_handle_lifecycle_parses_and_compiles_if_available(
    tmp_path,
):
    """Smoke native HIP IPC memory and event handle APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void ipc_handle_lifecycle(float* device_ptr) {
        hipIpcMemHandle_t mem_handle;
        hipIpcEventHandle_t event_handle;
        void* opened = NULL;
        hipEvent_t event;
        hipEvent_t opened_event;
        hipIpcGetMemHandle(&mem_handle, device_ptr);
        hipIpcOpenMemHandle(&opened, mem_handle, hipIpcMemLazyEnablePeerAccess);
        hipIpcCloseMemHandle(opened);
        hipEventCreateWithFlags(&event, hipEventInterprocess);
        hipIpcGetEventHandle(&event_handle, event);
        hipIpcOpenEventHandle(&opened_event, event_handle);
        hipEventDestroy(event);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Function: ipc_handle_lifecycle" in crossgl
    assert "void ipc_handle_lifecycle(ptr<f32> device_ptr)" in crossgl
    assert "var mem_handle: hipIpcMemHandle_t;" in crossgl
    assert "var event_handle: hipIpcEventHandle_t;" in crossgl
    assert "var opened: ptr<void> = NULL;" in crossgl
    assert "var event: hipEvent_t;" in crossgl
    assert "var opened_event: hipEvent_t;" in crossgl
    assert (
        "// HIP IPC get memory handle: output: mem_handle, pointer: device_ptr"
        in crossgl
    )
    assert (
        "// HIP IPC open memory handle: output: opened, handle: mem_handle, "
        "flags: hipIpcMemLazyEnablePeerAccess"
    ) in crossgl
    assert "// HIP IPC close memory handle: pointer: opened" in crossgl
    assert "// HIP event create: event, flags: hipEventInterprocess" in crossgl
    assert "// HIP IPC get event handle: output: event_handle, event: event" in crossgl
    assert (
        "// HIP IPC open event handle: output: opened_event, handle: event_handle"
        in crossgl
    )
    assert "// HIP event destroy: event" in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_module_occupancy_launch_parses_and_compiles_if_available(
    tmp_path,
):
    """Smoke native HIP module, function, occupancy, and cooperative launch APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    __global__ void occupancy_kernel(float* out) {
        out[threadIdx.x] = 0.0f;
    }

    void module_occupancy_launch_smoke(hipStream_t stream, float* out) {
        hipModule_t module;
        hipFunction_t function;
        hipFunction_t symbol_function;
        hipFuncAttributes attrs;
        hipDeviceptr_t global_ptr;
        size_t global_bytes = 0;
        unsigned int function_count = 0;
        int min_grid = 0;
        int block_size = 0;
        int active_blocks = 0;
        int attr_value = 0;
        void* kernel_params[] = { &out };

        hipModuleLoad(&module, "kernel.hsaco");
        hipModuleGetFunction(&function, module, "occupancy_kernel");
        hipModuleGetFunctionCount(&function_count, module);
        hipModuleGetGlobal(&global_ptr, &global_bytes, module, "device_symbol");
        hipGetFuncBySymbol(&symbol_function, (const void*)occupancy_kernel);
        hipFuncGetAttribute(
            &attr_value,
            hipFuncAttributeMaxThreadsPerBlock,
            symbol_function
        );
        hipFuncGetAttributes(&attrs, symbol_function);
        hipFuncSetAttribute(
            symbol_function,
            hipFuncAttributeMaxDynamicSharedMemorySize,
            0
        );
        hipFuncSetCacheConfig(symbol_function, hipFuncCachePreferL1);
        hipFuncSetSharedMemConfig(
            symbol_function, hipSharedMemBankSizeFourByte
        );
        hipOccupancyMaxPotentialBlockSize(
            &min_grid, &block_size, occupancy_kernel, 0, 0
        );
        hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks, occupancy_kernel, block_size, 0
        );
        hipModuleLaunchKernel(
            function,
            1, 1, 1,
            block_size, 1, 1,
            0,
            stream,
            kernel_params,
            NULL
        );
        hipLaunchCooperativeKernel(
            (const void*)occupancy_kernel,
            dim3(1),
            dim3(block_size),
            kernel_params,
            0,
            stream
        );
        hipModuleUnload(module);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Kernel: occupancy_kernel" in crossgl
    assert "// Function: module_occupancy_launch_smoke" in crossgl
    assert (
        "void module_occupancy_launch_smoke(hipStream_t stream, ptr<f32> out)"
        in crossgl
    )
    assert "var module: hipModule_t;" in crossgl
    assert "var function: hipFunction_t;" in crossgl
    assert "var symbol_function: hipFunction_t;" in crossgl
    assert "var attrs: hipFuncAttributes;" in crossgl
    assert "var global_bytes: u32 = 0;" in crossgl
    assert "var function_count: u32 = 0;" in crossgl
    assert "var min_grid: i32 = 0;" in crossgl
    assert "var block_size: i32 = 0;" in crossgl
    assert "var active_blocks: i32 = 0;" in crossgl
    assert "var attr_value: i32 = 0;" in crossgl
    assert '// HIP module load: output: module, file: "kernel.hsaco"' in crossgl
    assert (
        "// HIP module get function: output: function, module: module, "
        'name: "occupancy_kernel"'
    ) in crossgl
    assert (
        "// HIP module get function count: output: function_count, module: module"
        in crossgl
    )
    assert (
        "// HIP module get global: pointer output: global_ptr, "
        'size output: global_bytes, module: module, name: "device_symbol"'
    ) in crossgl
    assert (
        "// HIP get function by symbol: output: symbol_function, "
        "symbol: ptr<void>(occupancy_kernel)"
    ) in crossgl
    assert (
        "// HIP function get attribute: output: attr_value, "
        "attribute: hipFuncAttributeMaxThreadsPerBlock, "
        "function: symbol_function"
    ) in crossgl
    assert (
        "// HIP function get attributes: output: attrs, " "function: symbol_function"
    ) in crossgl
    assert (
        "// HIP function set attribute: function: symbol_function, "
        "attribute: hipFuncAttributeMaxDynamicSharedMemorySize, value: 0"
    ) in crossgl
    assert (
        "// HIP function set cache config: function: symbol_function, "
        "config: hipFuncCachePreferL1"
    ) in crossgl
    assert (
        "// HIP function set shared memory config: function: symbol_function, "
        "config: hipSharedMemBankSizeFourByte"
    ) in crossgl
    assert (
        "// HIP occupancy max potential block size: grid output: min_grid, "
        "block output: block_size, kernel: occupancy_kernel, "
        "dynamic shared memory: 0, block size limit: 0"
    ) in crossgl
    assert (
        "// HIP occupancy active blocks per multiprocessor: "
        "output: active_blocks, kernel: occupancy_kernel, "
        "block size: block_size, dynamic shared memory: 0"
    ) in crossgl
    assert (
        "// HIP module launch kernel: function: function, grid: (1, 1, 1), "
        "block: (block_size, 1, 1), shared memory: 0, stream: stream, "
        "params: kernel_params, extra: NULL"
    ) in crossgl
    assert (
        "// HIP cooperative kernel launch: "
        "function: ptr<void>(occupancy_kernel), grid: vec3<u32>(1), "
        "block: vec3<u32>(block_size), params: kernel_params, "
        "shared memory: 0, stream: stream"
    ) in crossgl
    assert "// HIP module unload: module" in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_pitched_array_memory_copy_parses_and_compiles_if_available(
    tmp_path,
):
    """Smoke native HIP pitched allocation, arrays, and structured copy APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void pitched_array_memory_copy(
        float* host,
        hipStream_t stream,
        size_t width_elems
    ) {
        float* pitched_device = NULL;
        size_t pitch = 0;
        size_t width_bytes = width_elems * sizeof(float);
        hipPitchedPtr pitched_3d;
        hipArray_t array;
        hipArray_t array_3d;
        HIP_ARRAY_DESCRIPTOR array_desc;
        HIP_ARRAY3D_DESCRIPTOR array3d_desc;
        hipChannelFormatDesc channel_desc;
        hipChannelFormatDesc queried_channel_desc;
        hipExtent extent = make_hipExtent(width_bytes, 4, 2);
        hipMemcpy3DParms copy_params = {0};
        HIP_MEMCPY3D driver_copy_params = {0};
        unsigned int flags = 0;
        hipDeviceptr_t device_ptr = 0;

        hipMallocPitch((void**)&pitched_device, &pitch, width_bytes, 4);
        hipMalloc3D(&pitched_3d, extent);
        hipMallocArray(&array, &channel_desc, width_elems, 4, hipArrayDefault);
        hipMalloc3DArray(&array_3d, &channel_desc, extent, hipArrayDefault);
        hipArrayCreate(&array, &array_desc);
        hipArray3DCreate(&array_3d, &array3d_desc);
        hipArrayGetDescriptor(&array_desc, array);
        hipArray3DGetDescriptor(&array3d_desc, array_3d);
        hipArrayGetInfo(&channel_desc, &extent, &flags, array);
        hipGetChannelDesc(&queried_channel_desc, array);
        hipMemcpy2D(
            pitched_device, pitch, host, width_bytes, width_bytes, 4,
            hipMemcpyHostToDevice
        );
        hipMemcpy2DAsync(
            pitched_device, pitch, host, width_bytes, width_bytes, 4,
            hipMemcpyHostToDevice, stream
        );
        hipMemcpyToArray(array, 0, 0, host, width_bytes, hipMemcpyHostToDevice);
        hipMemcpyToArrayAsync(
            array, 0, 0, host, width_bytes, hipMemcpyHostToDevice, stream
        );
        hipMemcpyFromArray(host, array, 0, 0, width_bytes, hipMemcpyDeviceToHost);
        hipMemcpyFromArrayAsync(
            host, array, 0, 0, width_bytes, hipMemcpyDeviceToHost, stream
        );
        hipMemcpy2DToArray(
            array, 0, 0, host, pitch, width_bytes, 4, hipMemcpyHostToDevice
        );
        hipMemcpy2DToArrayAsync(
            array, 0, 0, host, pitch, width_bytes, 4,
            hipMemcpyHostToDevice, stream
        );
        hipMemcpy2DFromArray(
            host, pitch, array, 0, 0, width_bytes, 4, hipMemcpyDeviceToHost
        );
        hipMemcpy2DFromArrayAsync(
            host, pitch, array, 0, 0, width_bytes, 4,
            hipMemcpyDeviceToHost, stream
        );
        hipMemcpyArrayToArray(
            array_3d, 0, 0, array, 0, 0, width_bytes,
            hipMemcpyDeviceToDevice
        );
        hipMemcpy2DArrayToArray(
            array_3d, 0, 0, array, 0, 0, width_bytes, 4,
            hipMemcpyDeviceToDevice
        );
        hipMemcpy3D(&copy_params);
        hipMemcpy3DAsync(&copy_params, stream);
        hipDrvMemcpy3D(&driver_copy_params);
        hipDrvMemcpy3DAsync(&driver_copy_params, stream);
        hipMemcpyAtoH(host, array, 0, width_bytes);
        hipMemcpyAtoHAsync(host, array, 0, width_bytes, stream);
        hipMemcpyHtoA(array, 4, host, width_bytes);
        hipMemcpyHtoAAsync(array, 4, host, width_bytes, stream);
        hipMemcpyAtoD(device_ptr, array, 8, width_bytes);
        hipMemcpyDtoA(array, 12, device_ptr, width_bytes);
        hipMemcpyAtoA(array_3d, 16, array, 20, width_bytes);
        hipArrayDestroy(array_3d);
        hipFreeArray(array);
        hipFree(pitched_device);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Function: pitched_array_memory_copy" in crossgl
    assert (
        "void pitched_array_memory_copy("
        "ptr<f32> host, hipStream_t stream, u32 width_elems)"
    ) in crossgl
    assert "var pitched_device: ptr<f32> = NULL;" in crossgl
    assert "var pitch: u32 = 0;" in crossgl
    assert "var width_bytes: u32 = (width_elems * sizeof(float));" in crossgl
    assert "var pitched_3d: hipPitchedPtr;" in crossgl
    assert "var array: ptr<void>;" in crossgl
    assert "var array_3d: ptr<void>;" in crossgl
    assert "var array_desc: HIP_ARRAY_DESCRIPTOR;" in crossgl
    assert "var array3d_desc: HIP_ARRAY3D_DESCRIPTOR;" in crossgl
    assert "var channel_desc: hipChannelFormatDesc;" in crossgl
    assert "var queried_channel_desc: hipChannelFormatDesc;" in crossgl
    assert "var extent: hipExtent = make_hipExtent(width_bytes, 4, 2);" in crossgl
    assert "var copy_params: hipMemcpy3DParms = {0};" in crossgl
    assert "var driver_copy_params: HIP_MEMCPY3D = {0};" in crossgl
    assert "var flags: u32 = 0;" in crossgl
    assert "var device_ptr: hipDeviceptr_t = 0;" in crossgl
    assert (
        "// HIP pitched memory allocate: pitched_device, pitch: pitch, "
        "width: width_bytes, height: 4"
    ) in crossgl
    assert "// HIP 3D memory allocate: pitched_3d, extent: extent" in crossgl
    assert (
        "// HIP array allocate: array, desc: channel_desc, width: width_elems, "
        "height: 4, flags: hipArrayDefault"
    ) in crossgl
    assert (
        "// HIP 3D array allocate: array_3d, desc: channel_desc, "
        "extent: extent, flags: hipArrayDefault"
    ) in crossgl
    assert "// HIP array create: output: array, descriptor: array_desc" in crossgl
    assert (
        "// HIP 3D array create: output: array_3d, descriptor: array3d_desc" in crossgl
    )
    assert "// HIP array get descriptor: output: array_desc, array: array" in crossgl
    assert (
        "// HIP array get 3D descriptor: output: array3d_desc, array: array_3d"
        in crossgl
    )
    assert (
        "// HIP array get info: desc output: channel_desc, "
        "extent output: extent, flags output: flags, array: array"
    ) in crossgl
    assert (
        "// HIP get channel desc: output: queried_channel_desc, array: array" in crossgl
    )
    assert (
        "// HIP 2D memory copy: host -> pitched_device, dst pitch: pitch, "
        "src pitch: width_bytes, width: width_bytes, height: 4, "
        "kind: hipMemcpyHostToDevice"
    ) in crossgl
    assert (
        "// HIP 2D memory copy: host -> pitched_device, dst pitch: pitch, "
        "src pitch: width_bytes, width: width_bytes, height: 4, "
        "kind: hipMemcpyHostToDevice, stream: stream"
    ) in crossgl
    assert (
        "// HIP memory copy to array: source: host, destination array: array, "
        "w offset: 0, h offset: 0, bytes: width_bytes, "
        "kind: hipMemcpyHostToDevice"
    ) in crossgl
    assert (
        "// HIP memory copy to array: source: host, destination array: array, "
        "w offset: 0, h offset: 0, bytes: width_bytes, "
        "kind: hipMemcpyHostToDevice, stream: stream"
    ) in crossgl
    assert (
        "// HIP memory copy from array: source array: array, w offset: 0, "
        "h offset: 0, destination: host, bytes: width_bytes, "
        "kind: hipMemcpyDeviceToHost"
    ) in crossgl
    assert (
        "// HIP memory copy from array: source array: array, w offset: 0, "
        "h offset: 0, destination: host, bytes: width_bytes, "
        "kind: hipMemcpyDeviceToHost, stream: stream"
    ) in crossgl
    assert (
        "// HIP 2D memory copy to array: source: host, source pitch: pitch, "
        "destination array: array, w offset: 0, h offset: 0, "
        "width: width_bytes, height: 4, kind: hipMemcpyHostToDevice"
    ) in crossgl
    assert (
        "// HIP 2D memory copy to array: source: host, source pitch: pitch, "
        "destination array: array, w offset: 0, h offset: 0, "
        "width: width_bytes, height: 4, kind: hipMemcpyHostToDevice, "
        "stream: stream"
    ) in crossgl
    assert (
        "// HIP 2D memory copy from array: source array: array, "
        "w offset: 0, h offset: 0, destination: host, "
        "destination pitch: pitch, width: width_bytes, height: 4, "
        "kind: hipMemcpyDeviceToHost"
    ) in crossgl
    assert (
        "// HIP 2D memory copy from array: source array: array, "
        "w offset: 0, h offset: 0, destination: host, "
        "destination pitch: pitch, width: width_bytes, height: 4, "
        "kind: hipMemcpyDeviceToHost, stream: stream"
    ) in crossgl
    assert (
        "// HIP memory copy array to array: source array: array, "
        "source w offset: 0, source h offset: 0, destination array: array_3d, "
        "destination w offset: 0, destination h offset: 0, "
        "bytes: width_bytes, kind: hipMemcpyDeviceToDevice"
    ) in crossgl
    assert (
        "// HIP 2D memory copy array to array: source array: array, "
        "source w offset: 0, source h offset: 0, destination array: array_3d, "
        "destination w offset: 0, destination h offset: 0, "
        "width: width_bytes, height: 4, kind: hipMemcpyDeviceToDevice"
    ) in crossgl
    assert "// HIP 3D memory copy: params: copy_params" in crossgl
    assert "// HIP 3D memory copy: params: copy_params, stream: stream" in crossgl
    assert "// HIP driver 3D memory copy: params: driver_copy_params" in crossgl
    assert (
        "// HIP driver 3D memory copy: params: driver_copy_params, stream: stream"
        in crossgl
    )
    assert (
        "// HIP driver memory copy array to host: source array: array, "
        "source offset: 0, destination host: host, bytes: width_bytes"
    ) in crossgl
    assert (
        "// HIP driver memory copy array to host: source array: array, "
        "source offset: 0, destination host: host, bytes: width_bytes, "
        "stream: stream"
    ) in crossgl
    assert (
        "// HIP driver memory copy host to array: source host: host, "
        "destination array: array, destination offset: 4, bytes: width_bytes"
    ) in crossgl
    assert (
        "// HIP driver memory copy host to array: source host: host, "
        "destination array: array, destination offset: 4, bytes: width_bytes, "
        "stream: stream"
    ) in crossgl
    assert (
        "// HIP driver memory copy array to device: source array: array, "
        "source offset: 8, destination device: device_ptr, bytes: width_bytes"
    ) in crossgl
    assert (
        "// HIP driver memory copy device to array: source device: device_ptr, "
        "destination array: array, destination offset: 12, bytes: width_bytes"
    ) in crossgl
    assert (
        "// HIP driver memory copy array to array: source array: array, "
        "source offset: 20, destination array: array_3d, "
        "destination offset: 16, bytes: width_bytes"
    ) in crossgl
    assert "// HIP array free: array_3d" in crossgl
    assert "// HIP array free: array" in crossgl
    assert "// HIP memory free: pitched_device" in crossgl

    raw_calls = (
        "hipMallocPitch",
        "hipMalloc3D",
        "hipMallocArray",
        "hipMalloc3DArray",
        "hipArrayCreate",
        "hipArray3DCreate",
        "hipArrayGetDescriptor",
        "hipArray3DGetDescriptor",
        "hipArrayGetInfo",
        "hipGetChannelDesc",
        "hipMemcpy2D",
        "hipMemcpy2DAsync",
        "hipMemcpyToArray",
        "hipMemcpyToArrayAsync",
        "hipMemcpyFromArray",
        "hipMemcpyFromArrayAsync",
        "hipMemcpy2DToArray",
        "hipMemcpy2DToArrayAsync",
        "hipMemcpy2DFromArray",
        "hipMemcpy2DFromArrayAsync",
        "hipMemcpyArrayToArray",
        "hipMemcpy2DArrayToArray",
        "hipMemcpy3D",
        "hipMemcpy3DAsync",
        "hipDrvMemcpy3D",
        "hipDrvMemcpy3DAsync",
        "hipMemcpyAtoH",
        "hipMemcpyAtoHAsync",
        "hipMemcpyHtoA",
        "hipMemcpyHtoAAsync",
        "hipMemcpyAtoD",
        "hipMemcpyDtoA",
        "hipMemcpyAtoA",
        "hipArrayDestroy",
        "hipFreeArray",
    )
    for raw_call in raw_calls:
        assert f"{raw_call}(" not in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_mipmapped_array_lifecycle_parses_and_compiles_if_available(
    tmp_path,
):
    """Smoke native HIP mipmapped-array allocation, level lookup, and cleanup."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void mipmapped_array_lifecycle(size_t width_elems) {
        hipMipmappedArray_t allocated_mipmapped_array;
        hipMipmappedArray_t created_mipmapped_array;
        hipArray_t allocated_level;
        hipArray_t created_level;
        hipChannelFormatDesc channel_desc;
        HIP_ARRAY3D_DESCRIPTOR array3d_desc;
        hipExtent extent = make_hipExtent(width_elems, 4, 1);

        hipMallocMipmappedArray(
            &allocated_mipmapped_array, &channel_desc, extent, 4,
            hipArrayDefault
        );
        hipMipmappedArrayCreate(&created_mipmapped_array, &array3d_desc, 3);
        hipGetMipmappedArrayLevel(
            &allocated_level, allocated_mipmapped_array, 1
        );
        hipMipmappedArrayGetLevel(&created_level, created_mipmapped_array, 2);
        hipMipmappedArrayDestroy(created_mipmapped_array);
        hipFreeMipmappedArray(allocated_mipmapped_array);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    expected_fragments = (
        "// Function: mipmapped_array_lifecycle",
        "void mipmapped_array_lifecycle(u32 width_elems)",
        "var allocated_mipmapped_array: hipMipmappedArray_t;",
        "var created_mipmapped_array: hipMipmappedArray_t;",
        "var allocated_level: ptr<void>;",
        "var created_level: ptr<void>;",
        "var channel_desc: hipChannelFormatDesc;",
        "var array3d_desc: HIP_ARRAY3D_DESCRIPTOR;",
        "var extent: hipExtent = make_hipExtent(width_elems, 4, 1);",
        "// HIP mipmapped array allocate: output: allocated_mipmapped_array, "
        "desc: channel_desc, extent: extent, levels: 4, "
        "flags: hipArrayDefault",
        "// HIP mipmapped array create: output: created_mipmapped_array, "
        "descriptor: array3d_desc, levels: 3",
        "// HIP mipmapped array get level: output: allocated_level, "
        "mipmapped array: allocated_mipmapped_array, level: 1",
        "// HIP mipmapped array get level: output: created_level, "
        "mipmapped array: created_mipmapped_array, level: 2",
        "// HIP free mipmapped array: created_mipmapped_array",
        "// HIP free mipmapped array: allocated_mipmapped_array",
    )
    for expected in expected_fragments:
        assert expected in crossgl

    raw_calls = (
        "hipMallocMipmappedArray",
        "hipMipmappedArrayCreate",
        "hipGetMipmappedArrayLevel",
        "hipMipmappedArrayGetLevel",
        "hipMipmappedArrayDestroy",
        "hipFreeMipmappedArray",
    )
    for raw_call in raw_calls:
        assert f"{raw_call}(" not in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_driver_memory_memset_parses_and_compiles_if_available(
    tmp_path,
):
    """Smoke native HIP driver memory, pointer metadata, and memset APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void driver_memory_memset_smoke(
        float* host,
        hipStream_t stream,
        size_t n
    ) {
        hipDeviceptr_t device_ptr;
        hipDeviceptr_t pitched_device_ptr;
        hipDeviceptr_t mapped_device_ptr;
        hipDeviceptr_t base_ptr;
        void* driver_host = NULL;
        void* device_void = (void*)device_ptr;
        void* attribute_data = NULL;
        size_t pitch = 0;
        size_t range_bytes = 0;
        size_t bytes = n * sizeof(float);
        hipPointerAttribute_t pointer_attrs;
        hipPointer_attribute pointer_attribute = hipPointerAttributeMemoryType;
        int pointer_attr_value = 0;

        hipMemAlloc(&device_ptr, bytes);
        hipMemAllocPitch(&pitched_device_ptr, &pitch, bytes, 4, 4);
        hipMemAllocHost(&driver_host, bytes);
        hipMemHostAlloc(&driver_host, bytes, hipHostMallocDefault);
        hipMemHostGetDevicePointer(&mapped_device_ptr, driver_host, 0);
        hipMemGetAddressRange(&base_ptr, &range_bytes, device_ptr);
        hipMemcpyHtoD(device_ptr, host, bytes);
        hipMemcpyHtoDAsync(device_ptr, host, bytes, stream);
        hipMemcpyDtoH(host, device_ptr, bytes);
        hipMemcpyDtoHAsync(host, device_ptr, bytes, stream);
        hipMemcpyDtoD(pitched_device_ptr, device_ptr, bytes);
        hipMemcpyDtoDAsync(pitched_device_ptr, device_ptr, bytes, stream);
        hipPointerGetAttributes(&pointer_attrs, device_void);
        hipDrvPointerGetAttributes(
            1, &pointer_attribute, &attribute_data, device_ptr
        );
        hipPointerGetAttribute(
            &pointer_attr_value, hipPointerAttributeMemoryType, device_void
        );
        hipPointerSetAttribute(
            &pointer_attr_value, hipPointerAttributeMemoryType, device_void
        );
        hipMemPtrGetInfo(device_void, &range_bytes);
        hipMemsetD8(device_ptr, 0, n);
        hipMemsetD8Async(device_ptr, 1, n, stream);
        hipMemsetD16(device_ptr, 0, n);
        hipMemsetD16Async(device_ptr, 1, n, stream);
        hipMemsetD32(device_ptr, 0, n);
        hipMemsetD32Async(device_ptr, 1, n, stream);
        hipMemsetD2D8(device_ptr, pitch, 0, n, 4);
        hipMemsetD2D8Async(device_ptr, pitch, 1, n, 4, stream);
        hipMemsetD2D16(device_ptr, pitch, 0, n, 4);
        hipMemsetD2D16Async(device_ptr, pitch, 1, n, 4, stream);
        hipMemsetD2D32(device_ptr, pitch, 0, n, 4);
        hipMemsetD2D32Async(device_ptr, pitch, 1, n, 4, stream);
        hipMemFreeHost(driver_host);
        hipMemFree(pitched_device_ptr);
        hipMemFree(device_ptr);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    expected_fragments = (
        "// Function: driver_memory_memset_smoke",
        "void driver_memory_memset_smoke(" "ptr<f32> host, hipStream_t stream, u32 n)",
        "var device_ptr: hipDeviceptr_t;",
        "var pitched_device_ptr: hipDeviceptr_t;",
        "var mapped_device_ptr: hipDeviceptr_t;",
        "var base_ptr: hipDeviceptr_t;",
        "var driver_host: ptr<void> = NULL;",
        "var device_void: ptr<void> = ptr<void>(device_ptr);",
        "var attribute_data: ptr<void> = NULL;",
        "var pitch: u32 = 0;",
        "var range_bytes: u32 = 0;",
        "var bytes: u32 = (n * sizeof(float));",
        "var pointer_attrs: hipPointerAttribute_t;",
        "var pointer_attribute: hipPointer_attribute = "
        "hipPointerAttributeMemoryType;",
        "var pointer_attr_value: i32 = 0;",
        "// HIP driver memory allocate: output: device_ptr, bytes: bytes",
        "// HIP driver pitched memory allocate: output: pitched_device_ptr, "
        "pitch output: pitch, width: bytes, height: 4, element bytes: 4",
        "// HIP driver host memory allocate: output: driver_host, bytes: bytes",
        "// HIP driver host memory allocate: output: driver_host, bytes: bytes, "
        "flags: hipHostMallocDefault",
        "// HIP driver host device pointer: output: mapped_device_ptr, "
        "host: driver_host, flags: 0",
        "// HIP driver memory address range: base output: base_ptr, "
        "size output: range_bytes, pointer: device_ptr",
        "// HIP driver memory copy host to device: source: host, "
        "destination: device_ptr, bytes: bytes",
        "// HIP driver memory copy host to device: source: host, "
        "destination: device_ptr, bytes: bytes, stream: stream",
        "// HIP driver memory copy device to host: source: device_ptr, "
        "destination: host, bytes: bytes",
        "// HIP driver memory copy device to host: source: device_ptr, "
        "destination: host, bytes: bytes, stream: stream",
        "// HIP driver memory copy device to device: source: device_ptr, "
        "destination: pitched_device_ptr, bytes: bytes",
        "// HIP driver memory copy device to device: source: device_ptr, "
        "destination: pitched_device_ptr, bytes: bytes, stream: stream",
        "// HIP pointer attributes: output: pointer_attrs, pointer: device_void",
        "// HIP driver pointer attributes: count: 1, "
        "attributes: (&pointer_attribute), data: (&attribute_data), "
        "pointer: device_ptr",
        "// HIP pointer attribute: output: pointer_attr_value, "
        "attribute: hipPointerAttributeMemoryType, pointer: device_void",
        "// HIP pointer set attribute: value: pointer_attr_value, "
        "attribute: hipPointerAttributeMemoryType, pointer: device_void",
        "// HIP memory pointer info: pointer: device_void, " "size output: range_bytes",
        "// HIP driver memory set 8-bit: pointer: device_ptr, value: 0, count: n",
        "// HIP driver memory set 8-bit: pointer: device_ptr, value: 1, "
        "count: n, stream: stream",
        "// HIP driver memory set 16-bit: pointer: device_ptr, value: 0, count: n",
        "// HIP driver memory set 16-bit: pointer: device_ptr, value: 1, "
        "count: n, stream: stream",
        "// HIP driver memory set 32-bit: pointer: device_ptr, value: 0, count: n",
        "// HIP driver memory set 32-bit: pointer: device_ptr, value: 1, "
        "count: n, stream: stream",
        "// HIP driver 2D memory set 8-bit: pointer: device_ptr, "
        "pitch: pitch, value: 0, width: n, height: 4",
        "// HIP driver 2D memory set 8-bit: pointer: device_ptr, "
        "pitch: pitch, value: 1, width: n, height: 4, stream: stream",
        "// HIP driver 2D memory set 16-bit: pointer: device_ptr, "
        "pitch: pitch, value: 0, width: n, height: 4",
        "// HIP driver 2D memory set 16-bit: pointer: device_ptr, "
        "pitch: pitch, value: 1, width: n, height: 4, stream: stream",
        "// HIP driver 2D memory set 32-bit: pointer: device_ptr, "
        "pitch: pitch, value: 0, width: n, height: 4",
        "// HIP driver 2D memory set 32-bit: pointer: device_ptr, "
        "pitch: pitch, value: 1, width: n, height: 4, stream: stream",
        "// HIP driver host memory free: driver_host",
        "// HIP driver memory free: pitched_device_ptr",
        "// HIP driver memory free: device_ptr",
    )
    for expected in expected_fragments:
        assert expected in crossgl

    raw_calls = (
        "hipMemAlloc",
        "hipMemAllocPitch",
        "hipMemAllocHost",
        "hipMemHostAlloc",
        "hipMemHostGetDevicePointer",
        "hipMemGetAddressRange",
        "hipMemcpyHtoD",
        "hipMemcpyHtoDAsync",
        "hipMemcpyDtoH",
        "hipMemcpyDtoHAsync",
        "hipMemcpyDtoD",
        "hipMemcpyDtoDAsync",
        "hipPointerGetAttributes",
        "hipDrvPointerGetAttributes",
        "hipPointerGetAttribute",
        "hipPointerSetAttribute",
        "hipMemPtrGetInfo",
        "hipMemsetD8",
        "hipMemsetD8Async",
        "hipMemsetD16",
        "hipMemsetD16Async",
        "hipMemsetD32",
        "hipMemsetD32Async",
        "hipMemsetD2D8",
        "hipMemsetD2D8Async",
        "hipMemsetD2D16",
        "hipMemsetD2D16Async",
        "hipMemsetD2D32",
        "hipMemsetD2D32Async",
        "hipMemFreeHost",
        "hipMemFree",
    )
    for raw_call in raw_calls:
        assert f"{raw_call}(" not in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_stream_callback_launch_config_parses_and_compiles_if_available(
    tmp_path,
):
    """Smoke native HIP stream callbacks and launch-configuration APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    __global__ void launch_config_kernel(float* out) {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        out[idx] = 1.0f;
    }

    void stream_callback(
        hipStream_t stream,
        hipError_t status,
        void* user_data
    ) {
    }

    void host_function(void* user_data) {
    }

    void stream_callback_launch_config_smoke(
        hipStream_t stream,
        float* out
    ) {
        dim3 grid(1, 1, 1);
        dim3 block(32, 1, 1);
        dim3 out_grid;
        dim3 out_block;
        size_t shared_mem = sizeof(float) * 32;
        hipStream_t out_stream;
        int value = 7;
        size_t offset = 0;
        hipStreamCallback_t callback = stream_callback;
        hipHostFn_t host_fn = host_function;
        void* packed_args[] = {&out};

        hipStreamAddCallback(stream, callback, out, 0);
        hipLaunchHostFunc(stream, host_fn, out);
        hipConfigureCall(grid, block, shared_mem, stream);
        __hipPushCallConfiguration(grid, block, 0, stream);
        __hipPopCallConfiguration(
            &out_grid, &out_block, &shared_mem, &out_stream
        );
        hipSetupArgument(&value, sizeof(value), offset);
        launch_config_kernel<<<grid, block, shared_mem, stream>>>(out);
        hipLaunchKernel(
            (const void*)launch_config_kernel,
            grid,
            block,
            packed_args,
            0,
            stream
        );
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    expected_fragments = (
        "// Kernel: launch_config_kernel",
        "// Function: stream_callback",
        "void stream_callback("
        "hipStream_t stream, hipError_t status, ptr<void> user_data)",
        "// Function: host_function",
        "void host_function(ptr<void> user_data)",
        "// Function: stream_callback_launch_config_smoke",
        "void stream_callback_launch_config_smoke(" "hipStream_t stream, ptr<f32> out)",
        "var grid: vec3<u32> = vec3<u32>(1, 1, 1);",
        "var block: vec3<u32> = vec3<u32>(32, 1, 1);",
        "var out_grid: vec3<u32>;",
        "var out_block: vec3<u32>;",
        "var shared_mem: u32 = (sizeof(float) * 32);",
        "var out_stream: hipStream_t;",
        "var value: i32 = 7;",
        "var offset: u32 = 0;",
        "var callback: hipStreamCallback_t = stream_callback;",
        "var host_fn: hipHostFn_t = host_function;",
        "// HIP stream add callback: stream: stream, callback: callback, "
        "user data: out, flags: 0",
        "// HIP launch host function: stream: stream, function: host_fn, "
        "user data: out",
        "// HIP configure call: grid: grid, block: block, "
        "shared memory: shared_mem, stream: stream",
        "// HIP push call configuration: grid: grid, block: block, "
        "shared memory: 0, stream: stream",
        "// HIP pop call configuration: grid output: out_grid, "
        "block output: out_block, shared memory output: shared_mem, "
        "stream output: out_stream",
        "// HIP setup kernel argument: value: (&value), "
        "bytes: sizeof(value), offset: offset",
        "// Kernel launch: launch_config_kernel<<<grid, block, shared_mem, "
        "stream>>>()",
        "// Kernel launch: launch_config_kernel<<<grid, block, 0, stream>>>()",
    )
    for expected in expected_fragments:
        assert expected in crossgl

    assert crossgl.count("// Arguments: out") == 2

    raw_calls = (
        "hipStreamAddCallback",
        "hipLaunchHostFunc",
        "hipConfigureCall",
        "__hipPushCallConfiguration",
        "__hipPopCallConfiguration",
        "hipSetupArgument",
        "hipLaunchKernel",
    )
    for raw_call in raw_calls:
        assert f"{raw_call}(" not in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_stream_priority_capture_info_parses_and_compiles_if_available(
    tmp_path,
):
    """Smoke native HIP priority streams and capture-info edge APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void stream_priority_capture_info_smoke() {
        hipStream_t stream;
        hipStream_t captured_stream;
        hipEvent_t event;
        hipGraph_t graph;
        hipGraph_t captured_graph;
        hipStreamCaptureStatus capture_status;
        hipStreamCaptureMode capture_mode = hipStreamCaptureModeGlobal;
        unsigned long long capture_id = 0;
        size_t num_dependencies = 0;
        int least_priority = 0;
        int greatest_priority = 0;
        int priority = 0;
        unsigned int flags = 0;

        hipDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
        hipStreamCreateWithPriority(
            &stream, hipStreamNonBlocking, greatest_priority
        );
        hipStreamGetFlags(stream, &flags);
        hipStreamGetPriority(stream, &priority);
        hipEventCreateWithFlags(&event, hipEventDisableTiming);
        hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
        hipStreamIsCapturing(stream, &capture_status);
        hipStreamGetCaptureInfo(stream, &capture_status, &capture_id);
        hipStreamGetCaptureInfo_v2(
            stream,
            &capture_status,
            &capture_id,
            &captured_graph,
            NULL,
            &num_dependencies
        );
        hipStreamUpdateCaptureDependencies(stream, NULL, 0, 0);
        hipEventRecord(event, stream);
        hipStreamWaitEvent(stream, event, 0);
        hipStreamEndCapture(stream, &graph);
        hipThreadExchangeStreamCaptureMode(&capture_mode);
        hipStreamCreateWithFlags(&captured_stream, hipStreamNonBlocking);
        hipStreamDestroy(captured_stream);
        hipGraphDestroy(graph);
        hipEventDestroy(event);
        hipStreamDestroy(stream);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    expected_fragments = (
        "// Function: stream_priority_capture_info_smoke",
        "void stream_priority_capture_info_smoke()",
        "var stream: hipStream_t;",
        "var captured_stream: hipStream_t;",
        "var event: hipEvent_t;",
        "var graph: hipGraph_t;",
        "var captured_graph: hipGraph_t;",
        "var capture_status: hipStreamCaptureStatus;",
        "var capture_mode: hipStreamCaptureMode = hipStreamCaptureModeGlobal;",
        "var capture_id: u64 = 0;",
        "var num_dependencies: u32 = 0;",
        "var least_priority: i32 = 0;",
        "var greatest_priority: i32 = 0;",
        "var priority: i32 = 0;",
        "var flags: u32 = 0;",
        "// HIP get stream priority range: "
        "least output: least_priority, greatest output: greatest_priority",
        "// HIP stream create: stream, flags: hipStreamNonBlocking, "
        "priority: greatest_priority",
        "// HIP get stream flags: stream: stream, output: flags",
        "// HIP get stream priority: stream: stream, output: priority",
        "// HIP event create: event, flags: hipEventDisableTiming",
        "// HIP stream begin capture: stream: stream, "
        "mode: hipStreamCaptureModeGlobal",
        "// HIP stream is capturing: stream: stream, output: capture_status",
        "// HIP stream capture info: stream: stream, "
        "status output: capture_status, id output: capture_id",
        "// HIP stream capture info: stream: stream, "
        "status output: capture_status, id output: capture_id, "
        "graph output: captured_graph, dependencies output: NULL, "
        "dependency count output: num_dependencies",
        "// HIP stream update capture dependencies: stream: stream, "
        "dependencies: NULL, count: 0, flags: 0",
        "// HIP event record: event, stream: stream",
        "// HIP stream wait event: stream waits for event, flags: 0",
        "// HIP stream end capture: stream: stream, graph output: graph",
        "// HIP exchange stream capture mode: output: capture_mode",
        "// HIP stream create: captured_stream, flags: hipStreamNonBlocking",
        "// HIP stream destroy: captured_stream",
        "// HIP graph destroy: graph",
        "// HIP event destroy: event",
        "// HIP stream destroy: stream",
    )
    for expected in expected_fragments:
        assert expected in crossgl

    raw_calls = (
        "hipDeviceGetStreamPriorityRange",
        "hipStreamCreateWithPriority",
        "hipStreamGetFlags",
        "hipStreamGetPriority",
        "hipEventCreateWithFlags",
        "hipStreamBeginCapture",
        "hipStreamIsCapturing",
        "hipStreamGetCaptureInfo",
        "hipStreamGetCaptureInfo_v2",
        "hipStreamUpdateCaptureDependencies",
        "hipEventRecord",
        "hipStreamWaitEvent",
        "hipStreamEndCapture",
        "hipThreadExchangeStreamCaptureMode",
        "hipStreamCreateWithFlags",
        "hipStreamDestroy",
        "hipGraphDestroy",
        "hipEventDestroy",
    )
    for raw_call in raw_calls:
        assert f"{raw_call}(" not in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_capture_to_graph_generic_node_parses_and_compiles_if_available(
    tmp_path,
):
    """Smoke native HIP capture-to-graph and generic graph-node APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void capture_to_graph_generic_node_smoke(hipStream_t stream) {
        hipGraph_t graph;
        hipGraph_t captured_graph;
        hipGraphExec_t exec;
        hipGraphNode_t generic_node;
        hipGraphNodeParams node_params;
        hipGraphInstantiateParams instantiate_params;
        unsigned long long exec_flags = 0;

        hipGraphCreate(&graph, 0);
        hipStreamBeginCaptureToGraph(
            stream,
            graph,
            NULL,
            NULL,
            0,
            hipStreamCaptureModeGlobal
        );
        hipStreamEndCapture(stream, &captured_graph);
        hipGraphAddNode(&generic_node, graph, NULL, 0, &node_params);
        hipGraphNodeSetParams(generic_node, &node_params);
        hipGraphInstantiateWithParams(&exec, graph, &instantiate_params);
        hipGraphUpload(exec, stream);
        hipGraphExecGetFlags(exec, &exec_flags);
        hipGraphExecNodeSetParams(exec, generic_node, &node_params);
        hipGraphLaunch(exec, stream);
        hipGraphDestroyNode(generic_node);
        hipGraphExecDestroy(exec);
        hipGraphDestroy(captured_graph);
        hipGraphDestroy(graph);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    expected_fragments = (
        "// Function: capture_to_graph_generic_node_smoke",
        "void capture_to_graph_generic_node_smoke(hipStream_t stream)",
        "var graph: hipGraph_t;",
        "var captured_graph: hipGraph_t;",
        "var exec: hipGraphExec_t;",
        "var generic_node: hipGraphNode_t;",
        "var node_params: hipGraphNodeParams;",
        "var instantiate_params: hipGraphInstantiateParams;",
        "var exec_flags: u64 = 0;",
        "// HIP graph create: output: graph, flags: 0",
        "// HIP stream begin capture to graph: stream: stream, graph: graph, "
        "dependencies: NULL, dependency data: NULL, count: 0, "
        "mode: hipStreamCaptureModeGlobal",
        "// HIP stream end capture: stream: stream, graph output: captured_graph",
        "// HIP graph add generic node: output: generic_node, graph: graph, "
        "dependencies: NULL, count: 0, params: (&node_params)",
        "// HIP graph generic node set params: node: generic_node, "
        "params: (&node_params)",
        "// HIP graph instantiate with params: output: exec, graph: graph, "
        "params: (&instantiate_params)",
        "// HIP graph upload: exec: exec, stream: stream",
        "// HIP graph exec get flags: exec: exec, output: exec_flags",
        "// HIP graph exec generic node set params: exec: exec, "
        "node: generic_node, params: (&node_params)",
        "// HIP graph launch: exec: exec, stream: stream",
        "// HIP graph destroy node: generic_node",
        "// HIP graph exec destroy: exec",
        "// HIP graph destroy: captured_graph",
        "// HIP graph destroy: graph",
    )
    for expected in expected_fragments:
        assert expected in crossgl

    raw_calls = (
        "hipGraphCreate",
        "hipStreamBeginCaptureToGraph",
        "hipStreamEndCapture",
        "hipGraphAddNode",
        "hipGraphNodeSetParams",
        "hipGraphInstantiateWithParams",
        "hipGraphUpload",
        "hipGraphExecGetFlags",
        "hipGraphExecNodeSetParams",
        "hipGraphLaunch",
        "hipGraphDestroyNode",
        "hipGraphExecDestroy",
        "hipGraphDestroy",
    )
    for raw_call in raw_calls:
        assert f"{raw_call}(" not in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)
