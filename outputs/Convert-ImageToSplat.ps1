param(
    [Parameter(Mandatory=$true)][string]$InputPath,
    [string]$OutputPath = (Join-Path $PSScriptRoot 'my-splats')
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path $PSScriptRoot -Parent
$sharpExe = Join-Path $taskRoot 'work\sharp-env\Scripts\sharp.exe'
$checkpoint = Join-Path $taskRoot 'work\models\sharp_2572gikvuh.pt'
& $sharpExe predict -i $InputPath -o $OutputPath -c $checkpoint --device cuda
if ($LASTEXITCODE -ne 0) { throw "SHARP failed with exit code $LASTEXITCODE" }
Write-Host "Splats saved to $OutputPath"
