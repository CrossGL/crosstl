from setuptools import setup, find_packages

def prep(line):
    return line.split(" ")[0].split("#")[0].split(",")[0]

setup(
    name="crosstl",
    packages= ["crosstl","crosstl/src","crosstl/src/translator/","crosstl/src/translator/codegen/","crosstl/src/backend","crosstl/src/backend/DirectX","crosstl/src/backend/Metal/","crosstl/src/backend/Opengl/"],
    version="0.0.0.15",
    author="CrossGL team",
    author_email="vaatsalya@crossgl.net",
    description="CrossGL: Revolutionizing Shader Development",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://crossgl.net/",
    project_urls={
        "Documentation": "https://crossgl.github.io/index.html"
    },
    include_package_dats=True,
    #packages=find_packages(),

    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires = ["gast","pytest"],
)

