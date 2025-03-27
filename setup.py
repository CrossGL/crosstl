from setuptools import setup, find_packages
import os
import sys

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

# Verify directories exist with correct case on case-sensitive filesystems
if sys.platform.startswith("linux"):  # Linux is case-sensitive
    for dir_path in backend_dirs:
        if not os.path.isdir(dir_path):
            print(
                f"Warning: Directory '{dir_path}' does not exist or has incorrect case."
            )
            parent_dir = os.path.dirname(dir_path)
            if os.path.isdir(parent_dir):
                dir_name = os.path.basename(dir_path)
                # Check if the directory exists with a different case
                for item in os.listdir(parent_dir):
                    if item.lower() == dir_name.lower() and item != dir_name:
                        print(
                            f"  Found directory with different case: '{os.path.join(parent_dir, item)}'"
                        )
                        print(
                            f"  Please rename directories to match the expected case."
                        )
                        break
            else:
                print(f"  Parent directory '{parent_dir}' does not exist.")

# Create __init__.py files where needed
for dir_path in backend_dirs:
    # Ensure directory exists first
    os.makedirs(dir_path, exist_ok=True)

    init_file = os.path.join(dir_path, "__init__.py")
    if not os.path.exists(init_file):
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
