# Installs the pinned DirectX Shader Compiler and WARP 1.0.21 for the MLX
# workflow's Windows legs. Every MLX job that compiles or runs translated HLSL
# calls this one script, so the pins and checksums live in one place and the
# workflow file stays below the size GitHub accepts.
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$archive = Join-Path $env:RUNNER_TEMP "dxc.zip"
$root = Join-Path $env:RUNNER_TEMP "dxc"
Invoke-WebRequest -Uri "https://github.com/microsoft/DirectXShaderCompiler/releases/download/v1.9.2602.24/dxc_2026_05_27.zip" -OutFile $archive
if ((Get-FileHash $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne "cf658aacf070d3045e31b8f1f8a696c2945f37c1095019481ef7c513368db3b4") {
  throw "DXC archive checksum mismatch."
}
Expand-Archive $archive -DestinationPath $root
$bin = Join-Path $root "bin/x64"
if (-not (Test-Path "$bin/dxc.exe")) { throw "DXC executable missing." }
$bin | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append
$warpArchive = Join-Path $env:RUNNER_TEMP "warp.zip"
$warpRoot = Join-Path $env:RUNNER_TEMP "warp"
Invoke-WebRequest -Uri "https://api.nuget.org/v3-flatcontainer/microsoft.direct3d.warp/1.0.21/microsoft.direct3d.warp.1.0.21.nupkg" -OutFile $warpArchive
if ((Get-FileHash $warpArchive -Algorithm SHA256).Hash.ToLowerInvariant() -ne "ea44b77eb30eec14427193e20f40cdb2e4c31ed11ab39cb880538fdf4bac2681") {
  throw "WARP archive checksum mismatch."
}
Expand-Archive $warpArchive -DestinationPath $warpRoot
Copy-Item "$warpRoot/build/native/bin/x64/d3d10warp.dll" "$env:pythonLocation/d3d10warp.dll"
