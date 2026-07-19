from setuptools import find_namespace_packages, setup

setup(
    name="crosstl",
    packages=find_namespace_packages(include=["crosstl*"]),
    version="3.1.0",
    author="CrossGL team",
    author_email="nripesh@crossgl.net",
    description="CrossGL shader and compute translator",
    long_description=open("README.md", "r", -1, "UTF8").read(),
    long_description_content_type="text/markdown",
    url="https://crossgl.net/",
    project_urls={
        "Documentation": "https://crossgl.github.io/crossgl-docs/",
        "Source": "https://github.com/CrossGL/crosstl",
        "Issues": "https://github.com/CrossGL/crosstl/issues",
        "Changelog": "https://github.com/CrossGL/crosstl/blob/main/CHANGELOG.md",
    },
    license="Apache-2.0",
    include_package_data=True,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14"
        "Topic :: Software Development :: Compilers",
    ],
    python_requires=">=3.8",
    install_requires=["gast", "tomli; python_version<'3.11'"],
    extras_require={
        "directx-runtime": [
            "compushady>=0.17.5,<0.18; platform_system=='Windows'",
        ],
    },
    entry_points={
        "console_scripts": [
            "crosstl=crosstl._crosstl:main",
        ]
    },
)
