# Mesa software drivers for the conformance harness: llvmpipe (OpenGL 4.5 core through surfaceless EGL) and
# lavapipe (Vulkan), plus vkrun, spirv-as and moderngl. Runs on linux/amd64 and linux/arm64.
#
# Build from the repository root:
#   docker build -t crosstl-conformance-mesa:1 -f tests/conformance/docker/mesa/Dockerfile tests/conformance
#
# The base image is pinned by digest and the packages that implement the drivers under test are pinned to exact
# versions (Mesa 25.2.8, LLVM 20.1.2, Vulkan loader 1.3.275). Ubuntu keeps only the newest version of a package in
# noble-updates, so once a pin is superseded the build fails instead of silently testing another driver. Then move
# MESA and LLVM to the new versions, rebuild, re-run the suite and update the scoreboard in the same change.
FROM ubuntu:24.04@sha256:4fbb8e6a8395de5a7550b33509421a2bafbc0aab6c06ba2cef9ebffbc7092d90
ARG MESA=25.2.8-0ubuntu0.24.04.4
ARG LLVM=1:20.1.2-0ubuntu1~24.04.3
ARG VULKAN=1.3.275.0-1build1
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates build-essential python3 python3-venv python3-dev \
       libegl1 libegl-dev libgl-dev libgles2 libx11-dev spirv-tools \
       "mesa-vulkan-drivers=${MESA}" "libgl1-mesa-dri=${MESA}" "libegl-mesa0=${MESA}" "libgbm1=${MESA}" \
       "libglx-mesa0=${MESA}" "mesa-libgallium=${MESA}" "libllvm20=${LLVM}" \
       "libvulkan1=${VULKAN}" "libvulkan-dev=${VULKAN}" \
    && ln -s "$(ls /usr/share/vulkan/icd.d/lvp_icd*.json | head -n 1)" /opt/lvp_icd.json \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir moderngl==5.13.0 glcontext==3.1.0
COPY vkrun.c /opt/vkrun/vkrun.c
RUN cc -O2 -std=c11 -Wall -o /usr/local/bin/vkrun /opt/vkrun/vkrun.c -lvulkan
ENV PATH=/opt/venv/bin:$PATH \
    LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe EGL_PLATFORM=surfaceless MESA_SHADER_CACHE_DISABLE=true \
    VK_DRIVER_FILES=/opt/lvp_icd.json
WORKDIR /tmp
