#!/usr/bin/env pwsh
#
# Build the `dagnam` wheel + sdist into dag-lib/dist/.
#
# Two uses:
#   1. Release prep — produces the exact artifacts you twine-upload to PyPI
#      (see RELEASE.md).
#   2. Local end-to-end tests — a downstream consumer can install dagnam from
#      this dist/ via `uv pip install --find-links <dag-lib>/dist`, exercising
#      the same artifact you'll publish, before you publish it.
#
# Usage:
#   ./scripts/build-wheel.ps1            # clean build into dist/ (default)
#   ./scripts/build-wheel.ps1 -NoClean   # keep existing dist/ contents
#
[CmdletBinding()]
param(
    [switch]$NoClean
)

$ErrorActionPreference = 'Stop'

# scripts/ lives directly under the dag-lib root.
$DagLibRoot = Split-Path -Parent $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error 'uv is not on PATH. Install it: https://docs.astral.sh/uv/'
    exit 1
}

$DistDir = Join-Path $DagLibRoot 'dist'
if (-not $NoClean -and (Test-Path $DistDir)) {
    Write-Host "Cleaning $DistDir"
    Remove-Item -Recurse -Force $DistDir
}

Push-Location $DagLibRoot
try {
    Write-Host 'Building dagnam (wheel + sdist) into dist/ ...'
    & uv build
    if ($LASTEXITCODE -ne 0) { throw "uv build failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}

$wheel = Get-ChildItem -Path $DistDir -Filter 'dagnam-*.whl' | Select-Object -First 1
Write-Host ''
Write-Host "Built: $($wheel.FullName)"
Write-Host ''
Write-Host 'To exercise it from a downstream consumer, install from this wheelhouse:'
Write-Host "  uv pip install --find-links $DistDir dagnam"
