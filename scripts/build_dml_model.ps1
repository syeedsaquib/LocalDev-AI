$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$modelsRoot = Join-Path (Split-Path -Parent $projectRoot) 'models'
$sourceModel = Join-Path $modelsRoot 'Qwen2.5-Coder-7B-Instruct'
$outputModel = Join-Path $modelsRoot 'Qwen2.5-Coder-7B-Instruct-ONNX-DML'
$cacheDir = Join-Path $projectRoot 'local_data\cache\genai-build'
$python = (Get-Command python -ErrorAction Stop).Source

if (-not (Test-Path (Join-Path $sourceModel 'model.safetensors.index.json'))) {
    throw "Hugging Face model checkpoint not found at $sourceModel"
}

New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null
& $python -m onnxruntime_genai.models.builder -i $sourceModel -o $outputModel -p int4 -e dml -c $cacheDir
if ($LASTEXITCODE -ne 0) {
    throw "AMD GenAI DirectML conversion failed with exit code $LASTEXITCODE"
}

Write-Host "DirectML model created at $outputModel"
