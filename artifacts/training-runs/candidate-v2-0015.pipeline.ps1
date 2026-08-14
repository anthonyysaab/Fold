$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = "C:\Users\user\poker-nn-training\.venv\Scripts\python.exe"
$powershell = Join-Path $PSHOME "powershell.exe"
$trainingScript = Join-Path $PSScriptRoot "candidate-v2-0015.ps1"
$manifest = Join-Path $projectRoot "artifacts\candidates\candidate-v2-0015.manifest.json"
$statusPath = Join-Path $PSScriptRoot "candidate-v2-0015.pipeline.status.txt"
$trainLog = Join-Path $PSScriptRoot "candidate-v2-0015.train.log"
$trainError = Join-Path $PSScriptRoot "candidate-v2-0015.train.err.txt"
$evaluationDir = Join-Path $projectRoot "artifacts\evaluations"

function Set-PipelineStatus([string]$status) {
    [System.IO.File]::WriteAllText($statusPath, "$status`n")
}

function Run-Evaluation([string]$threshold, [string]$label) {
    $report = Join-Path $evaluationDir "candidate-v2-0015-hybrid-$label-gauntlet.json"
    $errorLog = Join-Path $evaluationDir "candidate-v2-0015-hybrid-$label-gauntlet.err.txt"
    if (Test-Path -LiteralPath $report) {
        throw "evaluation report already exists: $report"
    }
    Set-PipelineStatus "evaluating-hybrid-$threshold"
    & $python -m tools.evaluate_policies `
        --include-heuristic `
        --candidate $manifest `
        --hybrid-min-advantage $threshold `
        --seeds 2 `
        --json 1> $report 2> $errorLog
    if ($LASTEXITCODE -ne 0) {
        throw "hybrid $threshold evaluation failed with exit code $LASTEXITCODE"
    }
}

try {
    Set-Location $projectRoot
    Set-PipelineStatus "training"
    & $powershell -NoProfile -ExecutionPolicy Bypass `
        -File $trainingScript -Start 1> $trainLog 2> $trainError
    if ($LASTEXITCODE -ne 0) {
        throw "training failed with exit code $LASTEXITCODE"
    }
    if (-not (Test-Path -LiteralPath $manifest)) {
        throw "training returned success without writing the candidate manifest"
    }

    Run-Evaluation "0.10" "010"
    Run-Evaluation "0.20" "020"
    Set-PipelineStatus "complete"
}
catch {
    Set-PipelineStatus "failed: $($_.Exception.Message)"
    throw
}
