$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$pythonExe = Join-Path $repoRoot ".venv311fa\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-Error "Missing .venv311fa. Please recreate the flash-attn environment first."
}

$cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
if (Test-Path $cudaRoot) {
    $env:CUDA_HOME = $cudaRoot
    $env:CUDA_PATH = $cudaRoot
    $env:PATH = (Join-Path $cudaRoot "bin") + ";" + $env:PATH
}

Write-Host "Launching VibeVoice WebUI with flash-attn environment..."
Write-Host "Python: $pythonExe"
if ($env:CUDA_HOME) {
    Write-Host "CUDA_HOME: $env:CUDA_HOME"
}

& $pythonExe ".\webui.py"
