from setuptools import setup, find_packages
import os

# Ensure we have all necessary __init__.py files
backend_dirs = [
    "crosstl/backend",
    "crosstl/backend/DirectX",
    "crosstl/backend/Metal",
    "crosstl/backend/OpenGL",
    "crosstl/backend/Slang",
    "crosstl/backend/Vulkan",
    "crosstl/backend/Mojo",
    "crosstl/translator",
    "crosstl/translator/codegen",
]

for dir_path in backend_dirs:
    init_file = os.path.join(dir_path, "__init__.py")
    if not os.path.exists(init_file):
        # Create an empty __init__.py file if it doesn't exist
        with open(init_file, "w") as f:
            f.write('"""Package initialization for {}"""\n'.format(dir_path))

setup(
    name="crosstl",
    packages=find_packages(),
    version="0.0.1.3",
    author="CrossGL team",
    author_email="nripesh@crossgl.net",
    description="CrossTL: Revolutionizing Shader Development",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://crossgl.net/",
    project_urls={"Documentation": "https://crossgl.github.io/index.html"},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=["gast", "pytest"],
)
