param(
    [switch]$DryRun,
    [switch]$Start
)

if (-not $DryRun -and -not $Start) {
    Write-Host "candidate-v2-0015 is prepared but not started. Use -DryRun or -Start."
    exit 0
}

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = "C:\Users\user\poker-nn-training\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "CUDA Python environment not found at $python"
}
Set-Location $projectRoot
$env:PYTHONUNBUFFERED = "1"

$foreignCsvs = @(
    "foreign play data\last 5 seasons top 15\20260812T082657Z_poker-playground_s9_top15\top15_decisions.csv",
    "foreign play data\last 5 seasons top 15\20260812T082859Z_poker-playground_s10_top15\top15_decisions.csv",
    "foreign play data\last 5 seasons top 15\20260812T083023Z_poker-playground_s11_top15\top15_decisions.csv",
    "foreign play data\last 5 seasons top 15\20260812T083204Z_poker-playground_s12_top15\top15_decisions.csv",
    "foreign play data\last 5 seasons top 15\20260812T083313Z_poker-playground_s13_top15\top15_decisions.csv"
)
$trainingArgs = @(
    "-m", "tools.self_play_cycle",
    "--model-version", "candidate-v2-0015",
    "--lineup-hands", "3000",
    "--shover-hands", "3000",
    "--station-hands", "3000",
    "--nit-hands", "1000",
    "--sparring", "artifacts\candidates\candidate-v2-0006.manifest.json",
    "--sparring-hands", "2000",
    "--seed", "83",
    "--equity-trials", "80",
    "--starting-stack", "6000",
    "--epochs", "8",
    "--behavior-warmup-epochs", "1",
    "--return-scale-pct", "20",
    "--reinforcement-multiplier", "1.5",
    "--gradient-clip", "5",
    "--device", "cuda",
    "--batch-size", "1024"
)
foreach ($foreignCsv in $foreignCsvs) {
    $trainingArgs += @("--foreign-csv", $foreignCsv)
}
if ($DryRun) {
    $trainingArgs += "--dry-run"
}

& $python @trainingArgs
exit $LASTEXITCODE
