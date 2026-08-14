param([switch]$Start)

if (-not $Start) {
    Write-Host "candidate-v2-0014 is prepared but not started. Re-run with -Start to train."
    exit 0
}

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = "C:\Users\user\poker-nn-training\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "CUDA Python environment not found at $python"
}
Set-Location $projectRoot
$env:PYTHONUNBUFFERED = "1"

& $python -m tools.self_play_cycle `
    --model-version candidate-v2-0014 `
    --on-policy artifacts\candidates\candidate-v2-0013.manifest.json `
    --foreign-csv "foreign play data\last 5 seasons top 15\combined_top15_decisions_s9-s13.csv" `
    --lineup-hands 3000 `
    --shover-hands 3000 `
    --station-hands 3000 `
    --nit-hands 1000 `
    --sparring artifacts\candidates\candidate-v2-0006.manifest.json `
    --sparring-hands 2000 `
    --seed 79 `
    --equity-trials 80 `
    --starting-stack 6000 `
    --epochs 8 `
    --baseline-warmup-epochs 1 `
    --behavior-warmup-epochs 1 `
    --return-scale-pct 20 `
    --reinforcement-multiplier 1.5 `
    --gradient-clip 5 `
    --device cuda `
    --batch-size 1024

exit $LASTEXITCODE
