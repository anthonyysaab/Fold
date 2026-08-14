param(
    [switch]$DryRun,
    [switch]$Start,
    [switch]$RetrainOnly
)

# Candidate v7-0001: first format-2 candidate on the repaired label generator.
# One bundle per V7_DESIGN.md Stage 3 + owner-approved compression (2026-08-14):
# all-family forcing through the acting policy, two aggress sizes, salted
# chance re-deal, champion sparring with both seats recorded, and the measured
# AdamW recipe. Three init seeds train from one persisted corpus; only the
# best-validation seed goes to the gauntlet. Invalidates comparison with
# candidates 0013-0016 by design.

if (-not $DryRun -and -not $Start -and -not $RetrainOnly) {
    Write-Host "candidate-v7-0001 is prepared but not started. Use -DryRun, -Start, or -RetrainOnly."
    exit 0
}

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = "C:\Users\user\poker-nn-training\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "CUDA Python environment not found at $python"
}
Set-Location $projectRoot
$env:PYTHONUNBUFFERED = "1"

$corpus = "artifacts\corpora\candidate-v7-0001.corpus"
$foreignCsvs = @(
    "foreign play data\last 5 seasons top 15\20260812T082657Z_poker-playground_s9_top15\top15_decisions.csv",
    "foreign play data\last 5 seasons top 15\20260812T082657Z_poker-playground_s9_top15\top15_decisions.delta.csv",
    "foreign play data\last 5 seasons top 15\20260812T082859Z_poker-playground_s10_top15\top15_decisions.csv",
    "foreign play data\last 5 seasons top 15\20260812T083023Z_poker-playground_s11_top15\top15_decisions.csv",
    "foreign play data\last 5 seasons top 15\20260812T083023Z_poker-playground_s11_top15\top15_decisions.delta.csv",
    "foreign play data\last 5 seasons top 15\20260812T083204Z_poker-playground_s12_top15\top15_decisions.csv",
    "foreign play data\last 5 seasons top 15\20260812T083313Z_poker-playground_s13_top15\top15_decisions.csv"
)

$trainerArgs = @(
    "--architecture", "v7",
    "--device", "cuda",
    "--batch-size", "256",
    "--epochs", "40",
    "--learning-rate", "0.001",
    "--reinforcement-multiplier", "1.0",
    "--behavior-warmup-epochs", "0",
    "--return-scale-pct", "50",
    "--split-seed", "17",
    "--equity-trials", "80",
    "--starting-stack", "6000"
)
$harvestArgs = @(
    "--lineup-hands", "12000",
    "--shover-hands", "12000",
    "--station-hands", "12000",
    "--nit-hands", "4000",
    "--sparring", "champion",
    "--sparring-hands", "8000",
    "--sparring-record-both",
    "--counterfactual-rollouts", "2",
    "--seed", "89",
    "--harvest-workers", "0"
)

function Invoke-Cycle {
    param([string[]]$ExtraArgs)
    & $python -m tools.self_play_cycle @ExtraArgs
    if ($LASTEXITCODE -ne 0) { throw "self_play_cycle failed with $LASTEXITCODE" }
}

if ($DryRun) {
    $callArgs = @("--model-version", "candidate-v7-0001a", "--init-seed", "89", "--dry-run") + $trainerArgs + $harvestArgs
    foreach ($csv in $foreignCsvs) { $callArgs += @("--foreign-csv", $csv) }
    Invoke-Cycle $callArgs
    exit 0
}

if ($Start) {
    $callArgs = @("--model-version", "candidate-v7-0001a", "--init-seed", "89", "--examples-out", $corpus) + $trainerArgs + $harvestArgs
    foreach ($csv in $foreignCsvs) { $callArgs += @("--foreign-csv", $csv) }
    Invoke-Cycle $callArgs
}

# Seeds b and c retrain from the persisted corpus: byte-identical data,
# different initialization, per the measured 3-seed protocol.
foreach ($seedSpec in @(@("b", "90"), @("c", "91"))) {
    $suffix = $seedSpec[0]
    $seed = $seedSpec[1]
    $callArgs = @(
        "--model-version", "candidate-v7-0001$suffix",
        "--init-seed", $seed,
        "--examples-in", $corpus
    ) + $trainerArgs
    Invoke-Cycle $callArgs
}
exit 0
