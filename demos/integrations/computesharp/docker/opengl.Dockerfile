# Linux OpenGL execution environment for the ComputeSharp port.
#
# Mesa llvmpipe provides OpenGL 4.5 core through EGL without a display
# (EGL_MESA_platform_surfaceless). The image mirrors the Linux CI job: the
# pinned .NET SDK, a C compiler for the native backend and Python for the
# CrossTL translation and result accounting.
#
#   docker build -f demos/integrations/computesharp/docker/opengl.Dockerfile \
#       -t crosstl-computesharp-opengl demos/integrations/computesharp/docker
FROM ubuntu:24.04

ARG DOTNET_VERSION=10.0.400

ENV DEBIAN_FRONTEND=noninteractive \
    DOTNET_ROOT=/opt/dotnet \
    PATH=/opt/dotnet:$PATH \
    DOTNET_CLI_TELEMETRY_OPTOUT=1 \
    DOTNET_NOLOGO=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git gcc libc6-dev \
        libegl-dev libgl-dev libegl1 libegl-mesa0 libgl1-mesa-dri libgbm1 \
        libicu74 libssl3 python3 python3-gast \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSfL https://dot.net/v1/dotnet-install.sh -o /tmp/dotnet-install.sh \
    && bash /tmp/dotnet-install.sh --version "$DOTNET_VERSION" --install-dir /opt/dotnet \
    && rm /tmp/dotnet-install.sh \
    && dotnet --version
