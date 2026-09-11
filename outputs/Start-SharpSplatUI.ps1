$ErrorActionPreference = 'Stop'
$appDirectory = $PSScriptRoot
$taskDirectory = Split-Path $appDirectory -Parent
$python = Join-Path $taskDirectory 'work\sharp-env\Scripts\python.exe'

# Make sure logs exist; clear any stale 0-byte placeholders so we can see real output.
$logFile = Join-Path $appDirectory 'ui-server.log'
$errFile = Join-Path $appDirectory 'ui-server-error.log'
if (Test-Path $logFile) { Remove-Item $logFile -Force }
if (Test-Path $errFile) { Remove-Item $errFile -Force }

# PowerShell tee equivalent: redirect both streams into files AND the host console.
& $python (Join-Path $appDirectory 'SharpSplatUI.py') *>&1 | Tee-Object -FilePath $logFile | Write-Host
if ($LASTEXITCODE -ne 0) {
    Add-Content -Path $errFile -Value "Gradio UI exited with code $LASTEXITCODE at $(Get-Date -Format o)."
    exit $LASTEXITCODE
}
