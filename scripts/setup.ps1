$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot

python -m pip install -r (Join-Path $repoRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed with exit code $LASTEXITCODE"
}

foreach ($directory in @('database', 'vector_store', 'project_metadata', 'exports', 'cache', 'logs')) {
    New-Item -ItemType Directory -Force -Path (Join-Path $repoRoot "local_data\$directory") | Out-Null
}
Write-Host "LocalDev AI setup complete."
