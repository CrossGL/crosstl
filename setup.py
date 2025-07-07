from setuptools import setup

setup(
    name="crosstl",
    packages=[
        "crosstl",
        "crosstl/translator/",
        "crosstl/translator/codegen/",
        "crosstl/backend",
        "crosstl/backend/DirectX",
        "crosstl/backend/Metal",
        "crosstl/backend/GLSL",
        "crosstl/backend/SPIRV",
        "crosstl/backend/Mojo",
        "crosstl/backend/Rust",
        "crosstl/backend/CUDA",
        "crosstl/backend/HIP",
    ],
    version="1.0.0",
    author="CrossGL team",
    author_email="nripesh@crossgl.net",
    description="CrossTL: Universal Programming Language and Translator",
    long_description=open("README.md", "r", -1, "UTF8").read(),
    long_description_content_type="text/markdown",
    url="https://crossgl.net/",
    project_urls={"Documentation": "https://crossgl.github.io/crossgl-docs/"},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=["gast", "pytest"],
)
