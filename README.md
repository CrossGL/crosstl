<div style="display: block;" align="center">
    <img class="only-dark" width="10%" height="10%" src="https://github.com/CrossGL/crossgl-docs/blob/main/docs/assets/logo.png#gh-dark-mode-only"/>
</div>

---

<div style="display: block;" align="center">
    <img class="dark-light" width="5%" >
    <a href="https://crossgl.net/">
        <img class="dark-light" height="5%" width="5%" src="https://github.com/CrossGL/crossgl-docs/blob/main/docs/assets/web_icon.png">
    </a>
    <img class="dark-light" width="5%" >
    <a href="https://docs.crossgl.net/products/crossgl-translator/index.html">
        <img class="dark-light" height="5%" width="5%" src="https://github.com/CrossGL/crossgl-docs/blob/main/docs/assets/docs.png">
    </a>
    <img class="dark-light" width="5%" >
    <a href="https://github.com/CrossGL/demos">
        <img class="dark-light" height="5%" width="5%" src="https://github.com/CrossGL/crossgl-docs/blob/main/docs/assets/written.png">
    </a>
    <img class="dark-light" width="5%" >
    <a href="https://crossgl.github.io/crossgl-docs/pages/graphica/design">
        <img class="dark-light" height="5%" width="5%" src="https://github.com/CrossGL/crossgl-docs/blob/main/docs/assets/strategic-plan.png">
    </a>
</div>

<br>

<div style="margin-top: 10px; margin-bottom: 10px; display: block;" align="center">
    <a href="https://github.com/CrossGL/crosstl/issues">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/github/issues/CrossGL/crosstl">
    </a>
    <a href="https://github.com/CrossGL/crosstl/network/members">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/github/forks/CrossGL/crosstl">
    </a>
    <a href="https://github.com/CrossGL/crosstl/stargazers">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/github/stars/CrossGL/crosstl">
    </a>
    <a href="https://github.com/CrossGL/crosstl/pulls">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg">
    </a>
    <a href="https://pypi.org/project/crosstl/">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://badge.fury.io/py/crosstl.svg">
    </a>
    <a href="https://discord.com/invite/uyRQKXhcyW">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/discord/1240998239206113330?color=blue&label=%20&logo=discord&logoColor=white">
    </a>
    <a href="https://doi.org/10.5281/zenodo.15826974">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://zenodo.org/badge/DOI/10.5281/zenodo.15826974.svg">
    </a>
</div>
<br clear="all" />

# CrossTL - Universal Shader Translator

CrossTL is a shader and compute program translator built around **CrossGL** — an intermediate representation (IR) language that bridges graphics APIs, GPU-compute platforms, and systems-language targets.

## Supported Translation Targets

CrossTL provides translation from CrossGL to:

- **Metal** - Apple's graphics and compute API
- **DirectX (HLSL)** - Microsoft's graphics API
- **OpenGL (GLSL)** - Cross-platform graphics
- **WebGL (GLSL ES)** - Browser graphics through WebGL 2.0 compatible GLSL ES
- **WebGPU (WGSL)** - Browser and native WebGPU shader output
- **Vulkan (SPIRV)** - High-performance graphics and compute
- **CUDA** - NVIDIA parallel computing
- **HIP** - AMD GPU computing
- **Rust** - GPU-oriented Rust shader code
- **Mojo** - Mojo shader/compute modules
- **Slang** - Real-time shading language
- **CrossGL (.cgl)** - The IR/interchange format itself

Native source import is available for the bidirectional backends listed in the
support matrix. WebGL and WGSL are currently target-only outputs; `.webgl.glsl`,
`.wgsl`, and `.wesl` inputs are rejected until dedicated source frontends
land.

## Backend Readiness: DirectX / Metal / OpenGL

We maintain first-class, bidirectional support for the three cornerstone graphics APIs. Each backend is implemented as both a **source** (parse/import) and **codegen** (export) target, so you can round‑trip between native shaders and CrossGL without lossy hops.

- **DirectX / HLSL**
  - Pipeline coverage: vertex, fragment/pixel, compute, geometry, hull/domain (tessellation), mesh/task, full ray‑tracing stages.
  - Target profile aliases: `dx11`, `dx12`, `d3d11`, and `d3d12` resolve to the HLSL emitter for Direct3D deployment planning; final DXBC/DXIL packaging remains a toolchain step.
  - Resource model: cbuffers, register/space bindings, UAV/RW textures & buffers, structured buffers, Interlocked atomics, wave ops, texture/buffer dimension queries.
  - Semantics map to `SV_*` and user semantics, preserved through CrossGL attributes.
- **Metal**
  - Stages: vertex, fragment, compute, mesh/object, and ray‑tracing qualifiers.
  - Resource/binding support: argument buffers, `[[buffer]]`, `[[texture]]`, `[[sampler]]`, function constants, packed/simd types, indirect command buffers, payload/hit attributes.
  - Texture methods (sample/load/store/gather/compare) and threadgroup builtins are translated to CrossGL intrinsics.
- **OpenGL / GLSL**
  - Stages: vertex, fragment, compute with version inference (defaults to `#version 450 core` when absent).
  - Handles interface blocks, layouts/bindings, sampler & image operations, control flow, structs/arrays, discard, builtins (`gl_*`) and preprocessor directives.
- **Round‑trips verified** via converters in `crosstl.backend.{DirectX,Metal,GLSL}` and registered in `translator.source_registry` and `translator.codegen.registry`.

### How we validate backend parity

- Targeted unit suites exercise the full feature surface for these three backends (shader stages, bindings, intrinsics, control‑flow, resources, and preprocessor handling).
- Quick check: run `pytest tests/test_backend/test_directx tests/test_backend/test_metal tests/test_backend/test_GLSL -q` (currently 321 passing tests).
- End‑to‑end translation: translate native HLSL/Metal/GLSL → CrossGL → HLSL/Metal/GLSL/Vulkan to ensure attributes, layouts, and semantics survive round‑trips.

## Translation Architecture

CrossTL uses a multi-stage translation pipeline:

1. **Lexical Analysis**: Tokenization
2. **Syntax Analysis**: AST generation
3. **Semantic Analysis**: Type checking and scope resolution
4. **IR Generation**: Conversion to CrossGL intermediate representation
5. **Target Generation**: Backend-specific code generation

## CrossGL Programming Language Examples

### PBR Shader Example

The README demo uses the same current CrossGL source file as the showcase
notebook: [`demos/readme/showcase_pbr.cgl`](demos/readme/showcase_pbr.cgl) and
[`demos/readme/demo.ipynb`](demos/readme/demo.ipynb). The shader demonstrates a
real PBR-style material path: tangent-space normal mapping, GGX normal
distribution, Smith geometry, Schlick Fresnel, multiple point lights, texture
sampling, tone mapping, and gamma correction.

It uses the current struct-return stage style so backends such as Metal can
emit proper stage input/output structs and `vertex`/`fragment` entry points.

```cpp
shader ShowcasePBRShader {
    const float PI = 3.14159265359;
    const float EPSILON = 0.0001;
    const int MAX_LIGHTS = 4;

    struct VertexInput {
        vec3 position;
        vec3 normal;
        vec3 tangent;
        vec2 texCoord;
    }

    struct VertexOutput {
        vec3 worldPosition;
        vec3 worldNormal;
        vec3 worldTangent;
        vec3 worldBitangent;
        vec2 uv;
        vec4 position;
    }

    struct FragmentInput {
        vec3 worldPosition;
        vec3 worldNormal;
        vec3 worldTangent;
        vec3 worldBitangent;
        vec2 uv;
    }

    struct FragmentOutput {
        vec4 color;
    }

    float distributionGGX(vec3 normal, vec3 halfVector, float roughness) {
        float alpha = roughness * roughness;
        float alpha2 = alpha * alpha;
        float ndoth = max(dot(normal, halfVector), 0.0);
        float denom = ndoth * ndoth * (alpha2 - 1.0) + 1.0;
        return alpha2 / max(PI * denom * denom, EPSILON);
    }

    float geometrySchlickGGX(float ndotv, float roughness) {
        float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
        return ndotv / max(ndotv * (1.0 - k) + k, EPSILON);
    }

    float geometrySmith(vec3 normal, vec3 viewDir, vec3 lightDir, float roughness) {
        float ndotv = max(dot(normal, viewDir), 0.0);
        float ndotl = max(dot(normal, lightDir), 0.0);
        return geometrySchlickGGX(ndotv, roughness) * geometrySchlickGGX(ndotl, roughness);
    }

    vec3 fresnelSchlick(float cosTheta, vec3 baseReflectance) {
        return baseReflectance + (vec3(1.0) - baseReflectance) * pow(max(1.0 - cosTheta, 0.0), 5.0);
    }

    vertex {
        uniform mat4 modelMatrix;
        uniform mat4 viewProjectionMatrix;
        uniform mat3 normalMatrix;

        VertexOutput main(VertexInput input) {
            VertexOutput output;
            vec4 worldPosition = modelMatrix * vec4(input.position, 1.0);
            output.worldPosition = worldPosition.xyz;
            output.worldNormal = normalize(normalMatrix * input.normal);
            output.worldTangent = normalize(normalMatrix * input.tangent);
            output.worldBitangent = normalize(cross(output.worldNormal, output.worldTangent));
            output.uv = input.texCoord;
            output.position = viewProjectionMatrix * worldPosition;
            return output;
        }
    }

    fragment {
        uniform sampler2D albedoMap;
        uniform sampler2D normalMap;
        uniform sampler2D metallicRoughnessMap;
        uniform vec3 cameraPosition;
        uniform vec3 lightPositions[MAX_LIGHTS];
        uniform vec3 lightColors[MAX_LIGHTS];
        uniform int activeLightCount;
        uniform vec3 baseColor;
        uniform float metallicFactor;
        uniform float roughnessFactor;

        vec3 decodeNormal(vec3 encodedNormal, mat3 tangentFrame) {
            vec3 tangentNormal = encodedNormal * 2.0 - 1.0;
            return normalize(tangentFrame * tangentNormal);
        }

        FragmentOutput main(FragmentInput input) {
            FragmentOutput output;
            mat3 tangentFrame = mat3(
                normalize(input.worldTangent),
                normalize(input.worldBitangent),
                normalize(input.worldNormal)
            );
            vec3 normal = decodeNormal(texture(normalMap, input.uv).rgb, tangentFrame);
            vec3 viewDir = normalize(cameraPosition - input.worldPosition);
            vec3 albedo = texture(albedoMap, input.uv).rgb * baseColor;
            vec3 materialSample = texture(metallicRoughnessMap, input.uv).rgb;
            float metallic = clamp(materialSample.b * metallicFactor, 0.0, 1.0);
            float roughness = clamp(materialSample.g * roughnessFactor, 0.04, 1.0);
            vec3 baseReflectance = mix(vec3(0.04), albedo, metallic);
            vec3 radianceSum = vec3(0.0);

            for (int i = 0; i < MAX_LIGHTS; i++) {
                if (i >= activeLightCount) {
                    break;
                }
                vec3 lightVector = lightPositions[i] - input.worldPosition;
                float distanceSq = max(dot(lightVector, lightVector), EPSILON);
                vec3 lightDir = normalize(lightVector);
                vec3 halfVector = normalize(viewDir + lightDir);
                float attenuation = 1.0 / distanceSq;
                vec3 radiance = lightColors[i] * attenuation;

                float normalDistribution = distributionGGX(normal, halfVector, roughness);
                float geometry = geometrySmith(normal, viewDir, lightDir, roughness);
                vec3 fresnel = fresnelSchlick(max(dot(halfVector, viewDir), 0.0), baseReflectance);
                vec3 diffuseShare = (vec3(1.0) - fresnel) * (1.0 - metallic);
                vec3 numerator = normalDistribution * geometry * fresnel;
                float denominator = max(4.0 * max(dot(normal, viewDir), 0.0) * max(dot(normal, lightDir), 0.0), EPSILON);
                vec3 specular = numerator / denominator;
                float ndotl = max(dot(normal, lightDir), 0.0);
                radianceSum += (diffuseShare * albedo / PI + specular) * radiance * ndotl;
            }

            vec3 ambient = albedo * 0.03;
            vec3 color = ambient + radianceSum;
            color = color / (color + vec3(1.0));
            color = pow(color, vec3(1.0 / 2.2));
            output.color = vec4(color, 1.0);
            return output;
        }
    }
}
```

Run the showcase notebook or translate the source directly:

```bash
python -m crosstl translate demos/readme/showcase_pbr.cgl --backend metal --output -
python -m crosstl translate demos/readme/showcase_pbr.cgl --backend directx --output -
python -m crosstl translate demos/readme/showcase_pbr.cgl --backend opengl --output -
```

## Backend Compatibility

All backends support the core CrossGL language (structs, arrays, functions, control flow, texture sampling, shader stages). Some advanced language features have partial backend coverage:

| Feature | Not yet supported in |
|---------|---------------------|
| Generic functions | SPIR-V, CUDA, HIP, Mojo, Slang (pending monomorphization) |
| Geometry stage | Metal, CUDA, HIP, Mojo, Rust (diagnostic fallback) |
| Tessellation stage | OpenGL, Metal, CUDA, HIP, Mojo, Rust (diagnostic fallback) |
| Mesh/Task stage | CUDA, HIP, Mojo, Rust, Slang (diagnostic fallback) |

Translation calls targeting unsupported feature/backend combinations raise a clear error with a reason string.

## Getting Started

Install CrossTL:

```bash
pip install crosstl
```

## Basic Usage

### 1. Create a CrossGL shader

```cpp
shader SimpleShader {
    struct VertexInput {
        vec3 position;
        vec2 texCoord;
    }

    struct VertexOutput {
        vec2 uv;
        vec4 position;
    }

    struct FragmentInput {
        vec2 uv;
    }

    struct FragmentOutput {
        vec4 color;
    }

    vertex {
        VertexOutput main(VertexInput input) {
            VertexOutput output;
            output.uv = input.texCoord;
            output.position = vec4(input.position, 1.0);
            return output;
        }
    }

    fragment {
        FragmentOutput main(FragmentInput input) {
            FragmentOutput output;
            output.color = vec4(input.uv, 0.5, 1.0);
            return output;
        }
    }
}
```

### 2. Translate to a target

```python
import crosstl

# Translate to any supported backend
metal_code = crosstl.translate('shader.cgl', backend='metal', save_shader='shader.metal')
hlsl_code = crosstl.translate('shader.cgl', backend='directx', save_shader='shader.hlsl')
dx12_hlsl = crosstl.translate('shader.cgl', backend='dx12', save_shader='shader.hlsl')
glsl_code = crosstl.translate('shader.cgl', backend='opengl', save_shader='shader.glsl')
```

### Project audit workflow

For repositories with many shader or GPU source files, start with a scan-only
report before writing translated artifacts:

```bash
python -m crosstl scan /path/to/repo \
  --target metal \
  --output scan-report.json

python -m crosstl translate-project /path/to/repo \
  --target metal \
  --output-dir crosstl-out \
  --report crosstl-out/portability-report.json

python -m crosstl validate-project \
  crosstl-out/portability-report.json \
  --format text

python -m crosstl inspect-report \
  crosstl-out/portability-report.json \
  --format text
```

Project reports cover shader and kernel source translation, diagnostics,
artifact provenance, optional toolchain checks, and manual migration actions
for host/runtime integration. See the [project porting guide](docs/source/project-porting.rst).
The repository also includes pinned open-source porting demos with checked
artifacts and platform validator coverage in
[`demos/open-source-porting`](demos/open-source-porting/README.md).

## Reverse Translation - Import Existing Code

```python
# Import existing shaders into CrossGL
conversions = [
    ('existing_shader.hlsl', 'unified.cgl'),     # DirectX to CrossGL
    ('gpu_kernel.cu', 'unified.cgl'),            # CUDA to CrossGL
    ('graphics.metal', 'unified.cgl'),           # Metal to CrossGL
]

for source, target in conversions:
    unified_code = crosstl.translate(source, backend='cgl', save_shader=target)
    print(f"Unified {source} into CrossGL: {target}")
```

## Deploying to All Backends

```python
import crosstl
from pathlib import Path

program = 'universal_shader.cgl'

deployment_targets = {
    'metal': '.metal',
    'directx': '.hlsl',
    'dx12': '.hlsl',
    'opengl': '.glsl',
    'vulkan': '.spvasm',
    'rust': '.rs',
    'mojo': '.mojo',
    'cuda': '.cu',
    'hip': '.hip',
    'slang': '.slang',
}

stem = Path(program).stem
for backend, extension in deployment_targets.items():
    output_file = f'deployments/{stem}_{backend}{extension}'
    try:
        crosstl.translate(program, backend=backend, save_shader=output_file)
        print(f"{backend}: {output_file}")
    except Exception as e:
        print(f"{backend}: {str(e)}")
```

For comprehensive language documentation, visit our [Language Reference](https://crossgl.github.io/crossgl-docs/pages/graphica/language/).

# Contributing

CrossGL is a community-driven project. Find out more in our [Contributing Guide](CONTRIBUTING.md).

<a href="https://github.com/CrossGL/crosstl/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=CrossGL/crosstl" />
</a>

# Community

- [Twitter](https://x.com/crossGL_)
- [LinkedIn](https://www.linkedin.com/company/crossgl/?viewAsMember=true)
- [Discord](https://discord.com/invite/uyRQKXhcyW)
- [YouTube](https://www.youtube.com/channel/UCxv7_flRCHp7p0fjMxVSuVQ)

## License

CrossTL is open-source and licensed under the [Apache License 2.0](https://github.com/CrossGL/crosstl/blob/main/LICENSE).

---

The CrossGL Team
