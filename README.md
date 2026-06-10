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

CrossTL provides bidirectional translation between CrossGL and:

- **Metal** - Apple's graphics and compute API
- **DirectX (HLSL)** - Microsoft's graphics API
- **OpenGL (GLSL)** - Cross-platform graphics
- **Vulkan (SPIRV)** - High-performance graphics and compute
- **CUDA** - NVIDIA parallel computing
- **HIP** - AMD GPU computing
- **Rust** - GPU-oriented Rust shader code
- **Mojo** - Mojo shader/compute modules
- **Slang** - Real-time shading language
- **CrossGL (.cgl)** - The IR/interchange format itself

## Backend Readiness: DirectX / Metal / OpenGL

We maintain first-class, bidirectional support for the three cornerstone graphics APIs. Each backend is implemented as both a **source** (parse/import) and **codegen** (export) target, so you can round‑trip between native shaders and CrossGL without lossy hops.

- **DirectX / HLSL**
  - Pipeline coverage: vertex, fragment/pixel, compute, geometry, hull/domain (tessellation), mesh/task, full ray‑tracing stages.
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

```cpp
// Physically-based rendering shader
struct Material {
    albedo: vec3,
    metallic: float,
    roughness: float,
    normal_map: texture2d,
    displacement: texture2d
}

struct Lighting {
    position: vec3,
    color: vec3,
    intensity: float,
    attenuation: vec3
}

shader PBRShader {
    vertex {
        input vec3 position;
        input vec3 normal;
        input vec2 texCoord;
        input vec4 tangent;

        uniform mat4 modelMatrix;
        uniform mat4 viewMatrix;
        uniform mat4 projectionMatrix;

        output vec3 worldPos;
        output vec3 worldNormal;
        output vec2 uv;
        output mat3 TBN;

        void main() {
            vec4 worldPosition = modelMatrix * vec4(position, 1.0);
            worldPos = worldPosition.xyz;

            vec3 T = normalize(vec3(modelMatrix * vec4(tangent.xyz, 0.0)));
            vec3 N = normalize(vec3(modelMatrix * vec4(normal, 0.0)));
            vec3 B = cross(N, T) * tangent.w;
            TBN = mat3(T, B, N);

            worldNormal = N;
            uv = texCoord;

            gl_Position = projectionMatrix * viewMatrix * worldPosition;
        }
    }

    fragment {
        input vec3 worldPos;
        input vec3 worldNormal;
        input vec2 uv;
        input mat3 TBN;

        uniform Material material;
        uniform Lighting lights[8];
        uniform int lightCount;
        uniform vec3 cameraPos;

        output vec4 fragColor;

        vec3 calculatePBR(vec3 albedo, float metallic, float roughness,
                         vec3 normal, vec3 viewDir, vec3 lightDir, vec3 lightColor) {
            vec3 halfVector = normalize(viewDir + lightDir);
            float NdotV = max(dot(normal, viewDir), 0.0);
            float NdotL = max(dot(normal, lightDir), 0.0);
            float NdotH = max(dot(normal, halfVector), 0.0);
            float VdotH = max(dot(viewDir, halfVector), 0.0);

            vec3 F0 = mix(vec3(0.04), albedo, metallic);
            vec3 F = F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);

            float alpha = roughness * roughness;
            float alpha2 = alpha * alpha;
            float denom = NdotH * NdotH * (alpha2 - 1.0) + 1.0;
            float D = alpha2 / (3.14159 * denom * denom);

            float G = geometrySmith(NdotV, NdotL, roughness);

            vec3 numerator = D * G * F;
            float denominator = 4.0 * NdotV * NdotL + 0.001;
            vec3 specular = numerator / denominator;

            vec3 kS = F;
            vec3 kD = vec3(1.0) - kS;
            kD *= 1.0 - metallic;

            return (kD * albedo / 3.14159 + specular) * lightColor * NdotL;
        }

        void main() {
            vec3 normal = normalize(TBN * (texture(material.normal_map, uv).rgb * 2.0 - 1.0));
            vec3 viewDir = normalize(cameraPos - worldPos);

            vec3 color = vec3(0.0);

            for (int i = 0; i < lightCount; ++i) {
                vec3 lightDir = normalize(lights[i].position - worldPos);
                float distance = length(lights[i].position - worldPos);
                float attenuation = 1.0 / (lights[i].attenuation.x +
                                          lights[i].attenuation.y * distance +
                                          lights[i].attenuation.z * distance * distance);

                vec3 lightColor = lights[i].color * lights[i].intensity * attenuation;
                color += calculatePBR(material.albedo, material.metallic,
                                     material.roughness, normal, viewDir, lightDir, lightColor);
            }

            color += material.albedo * 0.03;

            color = color / (color + vec3(1.0));
            color = pow(color, vec3(1.0/2.2));

            fragColor = vec4(color, 1.0);
        }
    }
}
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
    vertex {
        input vec3 position;
        uniform mat4 modelViewProjection;
        output vec4 fragPosition;

        void main() {
            fragPosition = modelViewProjection * vec4(position, 1.0);
        }
    }

    fragment {
        input vec4 fragPosition;
        output vec4 fragColor;

        void main() {
            fragColor = vec4(1.0, 0.0, 0.0, 1.0);
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
