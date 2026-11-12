param([switch]$Demo)
$ErrorActionPreference = 'Stop'
$viewerPython = Join-Path $PSScriptRoot '.venv\Scripts\pythonw.exe'
if (-not (Test-Path -LiteralPath $viewerPython)) {
    throw "Python environment not found: $viewerPython"
}
$viewerArgs = @('-m', 'benchtop.camera_viewer', '--start')
if ($Demo) { $viewerArgs += '--demo' }
Start-Process -FilePath $viewerPython -ArgumentList $viewerArgs -WorkingDirectory $PSScriptRoot -WindowStyle Hidden
