<div style="display: block;" align="center">
    <img class="only-dark" width="10%" height="10%" src="https://github.com/CrossGL/crossgl-docs/blob/main/docs/assets/logo.png#gh-dark-mode-only"/>
</div>


------------------------------------------------------------------------

<div style="display: block;" align="center">
    <img class="dark-light" width="5%" >
    <a href="https://crossgl.net/">
        <img class="dark-light" height="5%" width="5%" src="https://github.com/CrossGL/crossgl-docs/blob/main/docs/assets/web_icon.png">
    </a>
    <img class="dark-light" width="5%" >
    <a href="https://crossgl.github.io/crossgl-docs/language.html">
        <img class="dark-light" height="5%" width="5%" src="https://github.com/CrossGL/crossgl-docs/blob/main/docs/assets/docs.png">
    </a>
    <img class="dark-light" width="5%" >
    <a href="https://github.com/CrossGL/demos">
        <img class="dark-light" height="5%" width="5%" src="https://github.com/CrossGL/crossgl-docs/blob/main/docs/assets/written.png">
    </a>
    <img class="dark-light" width="5%" >
    <a href="https://crossgl.github.io/crossgl-docs/design.html">
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

# CrossTL

The CrossTL is a core component of our platform, enabling the conversion of CrossGL shader code directly into various graphics APIs, such as DirectX, Metal, Vulkan, and OpenGL and vice-versa. This translator simplifies shader development by allowing a single, unified shader language to be used across multiple platforms.

## 🌟 CrossGL-Graphica: Revolutionizing Shader Development

### The Universal Shader Language

In the ever-evolving world of graphics programming, **CrossGL** emerges as a solution to bridge the gap between diverse graphics APIs.

#### 🚀 Write Once, Run Everywhere

Imagine writing a shader _once_ and deploying it across:

- 🍎 **Metal**
- 🎮 **DirectX**
- 🖥️ **OpenGL**
- 🖥️ **Vulkan**
- ⚙️  **Slang** 
- 🔥 **Mojo**

...all without changing a single line of code!

## Supported Backends

- Metal
- DirectX
- OpenGL
- Slang


## 💡 Key Benefits

1. **⏱️ Time-Saving**: Slash development time by eliminating the need for multiple shader versions.
2. **🛠️ Consistency**: Ensure uniform behavior across all platforms.
3. **🧠 Simplified Learning Curve**: Master one language instead of many.
4. **🔍 Enhanced Debugging**: Develop universal tools for shader analysis.
5. **🔮 Future-Proof**: Easily adapt to new graphics APIs as they emerge.

## How It Works

The translator takes CrossGL shader code and processes it through several stages:

1. **Parsing**: The code is parsed into an abstract syntax tree (AST).
2. **Intermediate Representation**: The AST is converted into an intermediate representation (IR) for optimization.
3. **Code Generation**: The IR is translated into the target backend code.
4. **Optimization**: Various optimization passes are applied to ensure maximum performance.
5. **Source Output**: The final output is produced and ready for use.

## 🔄 Two-Way Translation: From Platform-Specific to CrossGL

CrossGL doesn't just translate from a universal language to platform-specific shaders - it also works in reverse! This powerful feature allows developers to convert existing shaders from various platforms into CrossGL.

## 🌈 CrossGL Shader

```cpp
shader main {
    vertex {
        input vec3 position;
        output vec2 vUV;

        void main() {
            vUV = position.xy * 10.0;
            gl_Position = vec4(position, 1.0);
        }
    }

    float perlinNoise(vec2 p) {
        return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
    }

    fragment {
        input vec2 vUV;
        output vec4 fragColor;

        void main() {
            float noise = perlinNoise(vUV);
            float height = noise * 10.0;
            vec3 color = vec3(height / 10.0, 1.0 - height / 10.0, 0.0);
            fragColor = vec4(color, 1.0);
        }
    }
}

```

# Getting started


First, install CrossGL's translation library using pip:

```bash
pip install crosstl
```

#### Using CrossGL

1. Create a CrossGL shader file (e.g., `shader.cgl`):

```glsl
shader main {
    vertex {
        input vec3 position;
        output vec2 vUV;

        void main() {
            vUV = position.xy * 10.0;
            gl_Position = vec4(position, 1.0);
        }
    }

    fragment {
        input vec2 vUV;
        output vec4 fragColor;

        void main() {
            fragColor = vec4(vUV, 0.0, 1.0);
        }
    }
}
```

2. Translate to your desired backend:

```python
import crosstl

# Translate to Metal
metal_code = crosstl.translate('shader.cgl', backend='metal', save_shader= 'shader.metal')

# Translate to DirectX (HLSL)
hlsl_code = crosstl.translate('shader.cgl', backend='directx', save_shader= 'shader.hlsl')

# Translate to OpenGL
opengl_code = crosstl.translate('shader.cgl', backend='opengl', save_shader= 'shader.glsl')

```

#### Converting from HLSL to CrossGL

1. write your HLSL shader (e.g., `shader.hlsl`):

```hlsl
struct VS_INPUT {
    float3 position : POSITION;
};

struct PS_INPUT {
    float4 position : SV_POSITION;
    float2 uv : TEXCOORD0;
};

PS_INPUT VSMain(VS_INPUT input) {
    PS_INPUT output;
    output.position = float4(input.position, 1.0);
    output.uv = input.position.xy * 10.0;
    return output;
}

float4 PSMain(PS_INPUT input) : SV_TARGET {
    return float4(input.uv, 0.0, 1.0);
}
```

2. Convert to CrossGL:

```python
import crosstl

crossgl_code = crosstl.translate('shader.hlsl', backend='cgl', save_shader= 'shader.cgl')
print(crossgl_code)
```

#### Converting from Metal to CrossGL

1. write your Metal shader (e.g., `shader.metal`):

```metal
#include <metal_stdlib>
using namespace metal;

struct VertexInput {
    float3 position [[attribute(0)]];
};

struct VertexOutput {
    float4 position [[position]];
    float2 uv;
};

vertex VertexOutput vertexShader(VertexInput in [[stage_in]]) {
    VertexOutput out;
    out.position = float4(in.position, 1.0);
    out.uv = in.position.xy * 10.0;
    return out;
}

fragment float4 fragmentShader(VertexOutput in [[stage_in]]) {
    return float4(in.uv, 0.0, 1.0);
}
```

2. Convert to CrossGL:

```python
import crosstl

crossgl_code = crosstl.translate('shader.metal', backend='cgl')
print(crossgl_code)
```

With these examples, you can easily get started with CrossGL, translating between different shader languages, and integrating existing shaders into your CrossGL workflow. Happy shader coding! 🚀✨

For more deep dive into crosstl , check out our [Getting Started Notebook](https://colab.research.google.com/drive/1reF8usj2CA5R6M5JSrBKOQBtU4WW-si2?usp=sharing#scrollTo=D7qkQrpcQ7zF).


# Contributing


We believe that everyone can contribute and make a difference. Whether
it\'s writing code, fixing bugs, or simply sharing feedback,
your contributions are definitely welcome and appreciated 🙌

find out more info in our [Contributing guide](CONTRIBUTING.md)

<a href="https://github.com/CrossGL/crosstl/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=CrossGL/crosstl" />
</a>



# Community

<b> Stay connected and follow our latest updates and announcements </b>

- [Twitter](https://x.com/crossGL_)
- [LinkedIn](https://www.linkedin.com/company/crossgl/?viewAsMember=true)
- [Discord Channel](https://discord.com/invite/uyRQKXhcyW)
- [YouTube](https://www.youtube.com/channel/UCxv7_flRCHp7p0fjMxVSuVQ)

<b> See you there! </b>
<br>

<br>


## License

The CrossGL Translator is open-source and licensed under the [License](https://github.com/CrossGL/crosstl/blob/main/LICENSE).

<br>

Thank you for using the CrossGL Translator!

The CrossGL Team
