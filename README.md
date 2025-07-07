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
    <a href="https://crossgl.github.io/crossgl-docs/pages/graphica/language">
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
</div>
<br clear="all" />

# CrossTL - Universal Programming Language & Translator

CrossTL is a revolutionary universal programming language translator built around **CrossGL** - a powerful intermediate representation (IR) language that serves as the bridge between diverse programming languages, platforms, and computing paradigms. More than just a translation tool, CrossGL represents a complete programming language designed for universal code portability and cross-platform development.

## 🌍 CrossGL: The Universal Programming Language

### Beyond Shader Translation - A Complete Programming Ecosystem

**CrossGL** has evolved far beyond its origins as a shader language into a comprehensive programming language with full support for:

- **Advanced Control Flow**: Complex conditionals, loops, switch statements, and pattern matching
- **Rich Data Structures**: Arrays, structs, enums, and custom types
- **Memory Management**: Buffer handling, pointer operations, and memory layout control
- **Function Systems**: First-class functions, generics, and polymorphism
- **Compute Paradigms**: Parallel processing, GPU computing, and heterogeneous execution
- **Modern Language Features**: Type inference, pattern matching, and algebraic data types

#### 🚀 Write Once, Deploy Everywhere

CrossGL enables you to write sophisticated programs **once** and deploy across:

- **Graphics APIs**: Metal, DirectX (HLSL), OpenGL (GLSL), Vulkan (SPIRV)
- **Systems Languages**: Rust, Mojo
- **GPU Computing**: CUDA, HIP
- **Specialized Domains**: Slang (real-time graphics), compute shaders

## 🎯 Supported Translation Targets

CrossTL provides comprehensive bidirectional translation between CrossGL and major programming ecosystems:

### Graphics & Compute APIs

- **Metal** - Apple's unified graphics and compute API
- **DirectX (HLSL)** - Microsoft's graphics framework
- **OpenGL (GLSL)** - Cross-platform graphics standard
- **Vulkan (SPIRV)** - High-performance graphics and compute

### Systems Programming Languages

- **Rust** - Memory-safe systems programming with GPU support
- **Mojo** - AI-first systems language with Python compatibility

### Parallel Computing Platforms

- **CUDA** - NVIDIA's parallel computing platform
- **HIP** - AMD's GPU computing interface

### Specialized Languages

- **Slang** - Real-time shading and compute language

### Universal Intermediate Representation

- **CrossGL (.cgl)** - Our universal programming language and IR

## 💡 Revolutionary Advantages

1. **🔄 Universal Portability**: Write complex algorithms once, run on any platform
2. **⚡ Performance Preservation**: Maintain optimization opportunities across translations
3. **🧠 Simplified Development**: Master one language instead of platform-specific variants
4. **🔍 Advanced Debugging**: Universal tooling for analysis and optimization
5. **🔮 Future-Proof Architecture**: Easily adapt to emerging programming paradigms
6. **🌐 Bidirectional Translation**: Migrate existing codebases to CrossGL or translate to any target
7. **📈 Scalable Complexity**: From simple shaders to complex distributed algorithms
8. **🎯 Domain Flexibility**: Graphics, AI, HPC, systems programming, and beyond

## ⚙️ Translation Architecture

CrossTL employs a sophisticated multi-stage translation pipeline:

1. **Lexical Analysis**: Advanced tokenization with context-aware parsing
2. **Syntax Analysis**: Robust AST generation with error recovery
3. **Semantic Analysis**: Type checking, scope resolution, and semantic validation
4. **IR Generation**: Conversion to CrossGL intermediate representation
5. **Optimization Passes**: Platform-agnostic code optimization and analysis
6. **Target Generation**: Backend-specific code generation with optimization
7. **Post-Processing**: Platform-specific optimizations and formatting

## 🔄 Bidirectional Translation Capabilities

CrossGL supports seamless translation in both directions - import existing code from any supported language or export CrossGL to any target platform.

## 🌈 CrossGL Programming Language Examples

### Complex Algorithm Implementation

```cpp
// Advanced pattern matching and algebraic data types
enum Result<T, E> {
    Ok(T),
    Error(E)
}

struct Matrix<T> {
    data: T[],
    rows: u32,
    cols: u32
}

function matrixMultiply<T>(a: Matrix<T>, b: Matrix<T>) -> Result<Matrix<T>, String> {
    if (a.cols != b.rows) {
        return Result::Error("Matrix dimensions incompatible");
    }

    let result = Matrix<T> {
        data: new T[a.rows * b.cols],
        rows: a.rows,
        cols: b.cols
    };

    parallel for i in 0..a.rows {
        for j in 0..b.cols {
            let mut sum = T::default();
            for k in 0..a.cols {
                sum += a.data[i * a.cols + k] * b.data[k * b.cols + j];
            }
            result.data[i * result.cols + j] = sum;
        }
    }

    return Result::Ok(result);
}

// GPU compute shader with advanced memory management
compute spawn matrixCompute {
    buffer float* matrix_A;
    buffer float* matrix_B;
    buffer float* result;

    local float shared_memory[256];

    void main() {
        let idx = workgroup_id() * workgroup_size() + local_id();

        // Complex parallel reduction with shared memory
        shared_memory[local_id()] = matrix_A[idx] * matrix_B[idx];
        workgroup_barrier();

        // Tree reduction
        for stride in [128, 64, 32, 16, 8, 4, 2, 1] {
            if (local_id() < stride) {
                shared_memory[local_id()] += shared_memory[local_id() + stride];
            }
            workgroup_barrier();
        }

        if (local_id() == 0) {
            result[workgroup_id()] = shared_memory[0];
        }
    }
}
```

### Advanced Graphics Pipeline

```cpp
// Physically-based rendering with advanced material system
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

    // Advanced fragment shader with PBR lighting
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
            // Advanced PBR calculation with microfacet model
            vec3 halfVector = normalize(viewDir + lightDir);
            float NdotV = max(dot(normal, viewDir), 0.0);
            float NdotL = max(dot(normal, lightDir), 0.0);
            float NdotH = max(dot(normal, halfVector), 0.0);
            float VdotH = max(dot(viewDir, halfVector), 0.0);

            // Fresnel term
            vec3 F0 = mix(vec3(0.04), albedo, metallic);
            vec3 F = F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);

            // Distribution term (GGX)
            float alpha = roughness * roughness;
            float alpha2 = alpha * alpha;
            float denom = NdotH * NdotH * (alpha2 - 1.0) + 1.0;
            float D = alpha2 / (3.14159 * denom * denom);

            // Geometry term
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

            // Add ambient lighting
            color += material.albedo * 0.03;

            // Tone mapping and gamma correction
            color = color / (color + vec3(1.0));
            color = pow(color, vec3(1.0/2.2));

            fragColor = vec4(color, 1.0);
        }
    }
}
```

# Getting Started

Install CrossTL's universal translator:

```bash
pip install crosstl
```

## Basic Usage

### 1. Create CrossGL Program

```cpp
// algorithm.cgl - Universal algorithm implementation
function quicksort<T>(arr: T[], low: i32, high: i32) -> void {
    if (low < high) {
        let pivot = partition(arr, low, high);
        quicksort(arr, low, pivot - 1);
        quicksort(arr, pivot + 1, high);
    }
}

function partition<T>(arr: T[], low: i32, high: i32) -> i32 {
    let pivot = arr[high];
    let i = low - 1;

    for j in low..high {
        if (arr[j] <= pivot) {
            i++;
            swap(arr[i], arr[j]);
        }
    }

    swap(arr[i + 1], arr[high]);
    return i + 1;
}

compute parallel_sort {
    buffer float* data;
    uniform int size;

    void main() {
        let idx = global_id();
        if (idx < size) {
            // Parallel bitonic sort implementation
            bitonicSort(data, size, idx);
        }
    }
}
```

### 2. Universal Translation

```python
import crosstl

# Translate to any target language/platform
rust_code = crosstl.translate('algorithm.cgl', backend='rust', save_shader='algorithm.rs')
cuda_code = crosstl.translate('algorithm.cgl', backend='cuda', save_shader='algorithm.cu')
metal_code = crosstl.translate('algorithm.cgl', backend='metal', save_shader='algorithm.metal')
mojo_code = crosstl.translate('algorithm.cgl', backend='mojo', save_shader='algorithm.mojo')
```

## Advanced Translation Examples

### Cross-Platform AI Kernels

```python
import crosstl

# Translate neural network kernels across AI platforms
ai_kernel = """
compute neuralNetwork {
    buffer float* weights;
    buffer float* inputs;
    buffer float* outputs;
    buffer float* biases;

    uniform int input_size;
    uniform int output_size;

    void main() {
        let neuron_id = global_id();
        if (neuron_id >= output_size) return;

        float sum = 0.0;
        for i in 0..input_size {
            sum += weights[neuron_id * input_size + i] * inputs[i];
        }

        outputs[neuron_id] = relu(sum + biases[neuron_id]);
    }
}
"""

# Deploy across AI hardware platforms
cuda_ai = crosstl.translate(ai_kernel, backend='cuda')    # NVIDIA GPUs
hip_ai = crosstl.translate(ai_kernel, backend='hip')      # AMD GPUs
metal_ai = crosstl.translate(ai_kernel, backend='metal')  # Apple Silicon
mojo_ai = crosstl.translate(ai_kernel, backend='mojo')    # Mojo AI runtime
```

### Systems Programming Translation

```python
# Translate systems-level code across platforms
systems_code = """
struct MemoryPool {
    buffer u8* memory;
    size_t capacity;
    size_t used;
    mutex lock;
}

function allocate(pool: MemoryPool*, size: size_t) -> void* {
    lock_guard guard(pool.lock);

    if (pool.used + size > pool.capacity) {
        return null;
    }

    void* ptr = pool.memory + pool.used;
    pool.used += size;
    return ptr;
}
"""

rust_systems = crosstl.translate(systems_code, backend='rust')  # Memory-safe systems code
cpp_systems = crosstl.translate(systems_code, backend='cpp')    # High-performance C++
c_systems = crosstl.translate(systems_code, backend='c')        # Portable C code
```

## Reverse Translation - Import Existing Code

```python
# Import and unify existing codebases
conversions = [
    ('existing_shader.hlsl', 'unified.cgl'),     # DirectX to CrossGL
    ('gpu_kernel.cu', 'unified.cgl'),            # CUDA to CrossGL
    ('graphics.metal', 'unified.cgl'),           # Metal to CrossGL
    ('algorithm.rs', 'unified.cgl'),             # Rust to CrossGL
    ('compute.mojo', 'unified.cgl'),             # Mojo to CrossGL
]

for source, target in conversions:
    unified_code = crosstl.translate(source, backend='cgl', save_shader=target)
    print(f"✅ Unified {source} into CrossGL: {target}")
```

## Comprehensive Platform Deployment

```python
import crosstl

# One CrossGL program, unlimited platforms
program = 'universal_algorithm.cgl'

# Deploy across all supported platforms
deployment_targets = {
    # Graphics APIs
    'metal': '.metal',      # Apple ecosystems
    'directx': '.hlsl',     # Windows/Xbox
    'opengl': '.glsl',      # Cross-platform
    'vulkan': '.spirv',     # High-performance

    # Systems languages
    'rust': '.rs',          # Memory safety
    'mojo': '.mojo',        # AI-optimized
    'cpp': '.cpp',          # Performance
    'c': '.c',              # Portability

    # Parallel computing
    'cuda': '.cu',          # NVIDIA
    'hip': '.hip',          # AMD
    'opencl': '.cl',        # Cross-vendor

    # Specialized
    'slang': '.slang',      # Real-time graphics
    'wgsl': '.wgsl',        # Web platforms
}

for backend, extension in deployment_targets.items():
    output_file = f'deployments/{program.stem}_{backend}{extension}'
    try:
        code = crosstl.translate(program, backend=backend, save_shader=output_file)
        print(f"✅ {backend.upper()}: {output_file}")
    except Exception as e:
        print(f"⚠️ {backend.upper()}: {str(e)}")

print(f"\n🚀 Universal deployment complete!")
print(f"📁 Check deployments/ directory for platform-specific implementations")
```

## Language Features Deep Dive

CrossGL provides comprehensive programming language features:

### Type System

- **Strong static typing** with type inference
- **Generic programming** with type parameters
- **Algebraic data types** (enums with associated data)
- **Trait/interface system** for polymorphism
- **Memory layout control** for performance

### Control Flow

- **Pattern matching** for complex conditional logic
- **Advanced loops** (for, while, loop with break/continue)
- **Exception handling** with Result types
- **Async/await** for concurrent programming

### Memory Management

- **Explicit lifetime management** when needed
- **Buffer abstractions** for GPU programming
- **Pointer safety** with compile-time checks
- **Reference semantics** for efficient data sharing

### Parallelism

- **Compute shaders** for GPU programming
- **Parallel loops** for CPU vectorization
- **Workgroup operations** for GPU synchronization
- **Memory barriers** for consistency

For comprehensive language documentation, visit our [Language Reference](https://crossgl.github.io/crossgl-docs/pages/graphica/language/).

# Contributing

CrossGL is a community-driven project. Whether you're contributing language features, backend implementations, optimizations, or documentation, your contributions shape the future of universal programming! 🌟

Find out more in our [Contributing Guide](CONTRIBUTING.md)

<a href="https://github.com/CrossGL/crosstl/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=CrossGL/crosstl" />
</a>

# Community

**Join the universal programming revolution**

- [Twitter](https://x.com/crossGL_) - Latest updates and announcements
- [LinkedIn](https://www.linkedin.com/company/crossgl/?viewAsMember=true) - Professional community
- [Discord](https://discord.com/invite/uyRQKXhcyW) - Real-time discussions and support
- [YouTube](https://www.youtube.com/channel/UCxv7_flRCHp7p0fjMxVSuVQ) - Tutorials and deep dives

**Shape the future of programming languages!**

## License

CrossTL is open-source and licensed under the [MIT License](https://github.com/CrossGL/crosstl/blob/main/LICENSE).

---

**CrossGL: One Language, Infinite Possibilities** 🌍

_Building the universal foundation for the next generation of programming_

The CrossGL Team
