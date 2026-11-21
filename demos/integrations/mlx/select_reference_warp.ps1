# Runs the steps that follow on WARP 1.0.19, the reference library of the WARP
# conformance matrix, and restores the required WARP 1.0.21 library with
# -Restore. WARP 1.0.21 returns wrong WaveActiveSum results for partially
# active waves in 512-invocation [WaveSize(32)] groups (#2120); the matrix's
# fixed HLSL controls show it and WARP 1.0.19 passes them. Kernels that reduce
# through such a wave therefore execute on WARP 1.0.19. D3D12 loads
# d3d10warp.dll from the application directory first, so the selected library
# is the one beside python.exe. Both directions check the installed digest.
param([switch]$Restore)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$required = "e79c10550449365adf0a9393d97a0df69941e671ab6e952a78d92da066517ca3"
$reference = "1fd2f9d991f00d890d06c00f791a756446f5f29da508b12fdeb8ffd1b5696add"
$installed = Join-Path $env:pythonLocation "d3d10warp.dll"
$saved = Join-Path $env:RUNNER_TEMP "required-d3d10warp.dll"
function Get-Digest($path) { (Get-FileHash $path -Algorithm SHA256).Hash.ToLowerInvariant() }
if ($Restore) {
  if (-not (Test-Path $saved -PathType Leaf)) { throw "The required WARP 1.0.21 library was not saved." }
  Copy-Item $saved $installed -Force
  if ((Get-Digest $installed) -ne $required) { throw "Required WARP 1.0.21 library was not restored." }
  Write-Host "Restored WARP 1.0.21 ($required) beside python.exe."
  exit 0
}
if ((Get-Digest $installed) -ne $required) { throw "Required WARP 1.0.21 installation identity does not match." }
Copy-Item $installed $saved -Force
$archive = Join-Path $env:RUNNER_TEMP "warp-1.0.19.zip"
$root = Join-Path $env:RUNNER_TEMP "warp-1.0.19"
Invoke-WebRequest -Uri "https://api.nuget.org/v3-flatcontainer/microsoft.direct3d.warp/1.0.19/microsoft.direct3d.warp.1.0.19.nupkg" -OutFile $archive -TimeoutSec 120
if ((Get-Digest $archive) -ne "332b1b70602e6ceec0531bb352508e42ec4948acd3f576bfb46434c165d073f7") {
  throw "WARP 1.0.19 archive checksum mismatch."
}
Expand-Archive $archive -DestinationPath $root -Force
Copy-Item (Join-Path $root "build/native/bin/x64/d3d10warp.dll") $installed -Force
if ((Get-Digest $installed) -ne $reference) { throw "Reference WARP 1.0.19 library identity does not match." }
Write-Host "Selected reference WARP 1.0.19 ($reference) beside python.exe."
