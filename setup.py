from setuptools import setup, find_packages

setup(
    name="crosstl",
    packages=find_packages(),
    version="0.0.1.3",
    author="CrossGL team",
    author_email="nripesh@crossgl.net",
    description="CrossTL: Revolutionizing Shader Development",
    long_description=open("README.md", "r", -1, "UTF8").read(),
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
