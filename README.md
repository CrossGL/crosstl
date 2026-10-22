# CrossTL

[![Tests](https://github.com/CrossGL/crosstl/actions/workflows/full-tests.yml/badge.svg?branch=main)](https://github.com/CrossGL/crosstl/actions/workflows/full-tests.yml)
[![PyPI](https://img.shields.io/pypi/v/crosstl)](https://pypi.org/project/crosstl/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/CrossGL/crosstl/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15826974.svg)](https://doi.org/10.5281/zenodo.15826974)

CrossTL translates GPU shaders and compute kernels between languages. Every translation goes through CrossGL, a
small intermediate language: a frontend imports GLSL, HLSL, Metal or another source language into CrossGL, and a
code generator writes CrossGL out as GLSL, HLSL, Metal, SPIR-V, WGSL or another target. You can also write CrossGL
directly and generate every target from one source file.

![CrossTL coverage per language and direction](https://raw.githubusercontent.com/CrossGL/crosstl/coverage/coverage.svg)

CI measures this chart on every push to `main` and nightly. Every feature in a pinned upstream feature list gets a
generated probe, CrossTL translates it, and the language's reference compiler or validator checks the result. The
score is supported / (supported + refused + wrong). [COVERAGE.md](https://github.com/CrossGL/crosstl/blob/coverage/COVERAGE.md)
lists every gap behind each bar.

## Install

```bash
pip install crosstl
```

CrossTL needs Python 3.11 or later. Validating the generated code needs the target's own compiler or validator
(glslang, DXC, spirv-tools, naga, the Metal compiler, nvcc, hipcc, slangc, rust-gpu or the Mojo compiler), which
CrossTL does not install.

## Quickstart

Save this compute kernel as `saxpy.cgl`:

```glsl
shader Saxpy {
    compute {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        buffer float* x;
        buffer float* y;
        uniform float a;

        void main() {
            uint i = gl_GlobalInvocationID.x;
            y[i] = a * x[i] + y[i];
        }
    }
}
```

Then translate it from Python, where the source language comes from the file extension:

```python
import crosstl

metal = crosstl.translate("saxpy.cgl", backend="metal")
crosstl.translate("saxpy.cgl", backend="wgsl", save_shader="saxpy.wgsl")

# A native shader is imported into CrossGL first, then generated for the target.
glsl = crosstl.translate("blur.hlsl", backend="opengl")
crossgl = crosstl.translate("blur.hlsl", backend="cgl")
```

or from the command line:

```bash
crosstl translate saxpy.cgl --backend directx --output saxpy.hlsl
crosstl translate saxpy.cgl --backend vulkan --output -
```

`python -m crosstl` is the same command. The `examples/` directory has larger CrossGL programs.

## Languages

| Language | `backend` | Output | Input |
|---|---|---|---|
| GLSL | `opengl` | measured | measured |
| HLSL | `directx` | measured | measured |
| Metal | `metal` | measured | measured |
| SPIR-V assembly | `vulkan` | measured | measured |
| WGSL | `wgsl` | measured | not offered |
| GLSL ES 3.00 (WebGL 2) | `webgl` | measured | not offered |
| CUDA | `cuda` | measured | measured |
| HIP | `hip` | measured | measured |
| Slang | `slang` | measured | measured |
| Rust (rust-gpu) | `rust` | measured | measured |
| Mojo | `mojo` | measured | measured |
| OpenCL C | | not offered | measured |

A measured direction has a row in [COVERAGE.md](https://github.com/CrossGL/crosstl/blob/coverage/COVERAGE.md).
CrossTL has no OpenCL C generator, so the OpenCL C input row checks the HLSL that CrossTL writes from each OpenCL C
probe. The language names `glsl`, `hlsl`, `msl` and `spirv` work as backend names too, and `cgl` writes the CrossGL a
native source imports to. CrossGL has its own coverage row, CrossGL IR, which measures how many of the features of
all the measured output languages CrossGL can express.

CrossTL is meant to fail closed: a construct it cannot translate raises a diagnostic that names the construct and
the target, instead of emitting a placeholder. Output that the reference tool rejects counts as wrong in the coverage
report, and the Coverage workflow files it as a bug.

## How it works

```text
 .hlsl .metal .glsl .cu ...                                        .glsl .hlsl .metal .spvasm .wgsl ...
 native source --> frontend --> CrossGL source --> CrossGL parser --> code generator --> target source
                   (parse and                      (AST and types)     (one per target)
                    import)
```

Each source language has a frontend in `crosstl/backend/<Language>/` that parses it and writes CrossGL. The CrossGL
parser in `crosstl/translator/` builds the typed AST that every code generator in `crosstl/translator/codegen/`
reads, so a feature added to CrossGL is available to every target that can express it.

## Porting a repository

For a project with many shader or kernel sources, scan it before writing anything, then translate it and check the
report:

```bash
python -m crosstl scan path/to/repo --target metal --output scan-report.json
python -m crosstl translate-project path/to/repo --target metal --output-dir crosstl-out \
  --report crosstl-out/report.json
python -m crosstl validate-project crosstl-out/report.json --format text
python -m crosstl inspect-report crosstl-out/report.json --format text
```

A relative `--output-dir` is resolved against the project, so the translated files land in
`path/to/repo/crosstl-out/<target>/`. The report records each translated file, its diagnostics and provenance, the
reference toolchain checks that ran, and the host-side changes that remain manual. The [project porting guide](https://github.com/CrossGL/crosstl/blob/main/docs/source/project-porting.rst) covers
the options, and [`demos/`](https://github.com/CrossGL/crosstl/blob/main/demos/README.md) has pinned open-source ports with checked artifacts.

## Documentation

The user documentation is at [docs.crossgl.net](https://docs.crossgl.net/products/crossgl-translator/index.html).
The Sphinx sources, including the architecture notes and decision records, are in [`docs/source`](https://github.com/CrossGL/crosstl/tree/main/docs/source).

## Contributing

Read [CONTRIBUTING.md](https://github.com/CrossGL/crosstl/blob/main/CONTRIBUTING.md) for setup, the test tiers and the pull request flow. Issues labelled
[good first issue](https://github.com/CrossGL/crosstl/labels/good%20first%20issue) are a good place to start, and
questions belong in [Discussions](https://github.com/CrossGL/crosstl/discussions) or on
[Discord](https://discord.com/invite/uyRQKXhcyW). Report security problems as described in [SECURITY.md](https://github.com/CrossGL/crosstl/blob/main/SECURITY.md).

## Citation

If you use CrossTL in your work, cite it with the metadata in [CITATION.cff](https://github.com/CrossGL/crosstl/blob/main/CITATION.cff); GitHub's "Cite this
repository" button produces BibTeX and APA from it. Each release is archived on Zenodo under
[10.5281/zenodo.15826974](https://doi.org/10.5281/zenodo.15826974).

## License

CrossTL is licensed under the [Apache License 2.0](https://github.com/CrossGL/crosstl/blob/main/LICENSE).
