param([Parameter(Mandatory=$true)][string]$PySpinWheel)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path -LiteralPath $PySpinWheel -PathType Leaf)) {
    throw "PySpin wheel not found: $PySpinWheel"
}
uv sync --locked
if ($LASTEXITCODE -ne 0) { throw 'uv sync failed' }
uv pip install --python .venv\Scripts\python.exe $PySpinWheel
if ($LASTEXITCODE -ne 0) { throw 'PySpin installation failed' }
& .\.venv\Scripts\python.exe -c 'import torch, PySpin; print(torch.__version__, "CUDA", torch.cuda.is_available())'
if ($LASTEXITCODE -ne 0) { throw 'Runtime validation failed' }
