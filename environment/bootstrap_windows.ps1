<#
bootstrap_windows.ps1 - create the pinned `omnipose` analysis environment on a Windows PC.

    powershell -ExecutionPolicy Bypass -File environment\bootstrap_windows.ps1 [-EnvName omnipose] [-CondaRoot <dir>] [-SkipCheck]

Steps
  1. find conda (an existing Anaconda / Miniconda), otherwise install Miniconda per-user into -CondaRoot
     (no PATH / registry change, so the CPython 3.11 used by the Micro-Manager BeanShell scripts is untouched)
  2. conda create -n <EnvName> --file environment\omnipose_win64_explicit.txt      (exact package URLs)
  3. <env>\python.exe -m pip install -r environment\omnipose_pip_requirements.txt
  4. <env>\python.exe environment\check_env.py --model models\<newest checkpoint>   (CUDA + model + one inference)

Re-running is safe: an existing conda or env is reused. Requires internet access (repo.anaconda.com,
conda.anaconda.org, pypi.org). Windows PowerShell 5.1 is enough.
#>
param(
    [string]$EnvName = "omnipose",
    [string]$CondaRoot = "$env:USERPROFILE\miniconda3",
    [switch]$SkipCheck
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$Explicit = Join-Path $PSScriptRoot "omnipose_win64_explicit.txt"
$PipReq = Join-Path $PSScriptRoot "omnipose_pip_requirements.txt"
if (-not (Test-Path $Explicit)) { throw "missing $Explicit" }
if (-not (Test-Path $PipReq)) { throw "missing $PipReq" }

function Find-Conda {
    $cands = @(
        (Join-Path $CondaRoot "condabin\conda.bat"),
        "$env:USERPROFILE\miniconda3\condabin\conda.bat",
        "$env:USERPROFILE\anaconda3\condabin\conda.bat",
        "$env:LOCALAPPDATA\miniconda3\condabin\conda.bat",
        "C:\ProgramData\miniconda3\condabin\conda.bat",
        "C:\ProgramData\anaconda3\condabin\conda.bat"
    )
    foreach ($c in $cands) { if (Test-Path $c) { return $c } }
    $cmd = Get-Command conda.bat -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $cmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

# ---- 1. conda -------------------------------------------------------------
$conda = Find-Conda
if (-not $conda) {
    Write-Host "[1/4] conda not found: installing Miniconda (per-user) into $CondaRoot"
    $inst = Join-Path $env:TEMP "Miniconda3-latest-Windows-x86_64.exe"
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -OutFile $inst
    $p = Start-Process -FilePath $inst -ArgumentList @("/S", "/InstallationType=JustMe", "/RegisterPython=0", "/AddToPath=0", "/D=$CondaRoot") -Wait -PassThru
    if ($p.ExitCode -ne 0) { throw "Miniconda installer exit code $($p.ExitCode)" }
    $conda = Join-Path $CondaRoot "condabin\conda.bat"
    if (-not (Test-Path $conda)) { throw "conda.bat not found after install: $conda" }
} else {
    Write-Host "[1/4] using conda: $conda"
}
& $conda --version
if ($LASTEXITCODE -ne 0) { throw "conda does not run ($LASTEXITCODE)" }

# ---- 2. env from the explicit spec ----------------------------------------
$base = (& $conda info --base | Select-Object -Last 1).Trim()
$envPrefix = Join-Path $base "envs\$EnvName"
$py = Join-Path $envPrefix "python.exe"
if (Test-Path $py) {
    Write-Host "[2/4] env '$EnvName' already exists at $envPrefix (reusing)"
} else {
    Write-Host "[2/4] creating env '$EnvName' from $Explicit (about 3 GB of downloads)"
    & $conda create -y -n $EnvName --file $Explicit
    if ($LASTEXITCODE -ne 0) { throw "conda create failed ($LASTEXITCODE)" }
}
if (-not (Test-Path $py)) { throw "python.exe not found: $py" }

# ---- 3. pip layer -----------------------------------------------------------
# The pin list is the complete pip layer of a working env (every dependency is either in it or
# conda-managed), so it is installed without dependency resolution: pip's resolver otherwise
# backtracks for a long time on declared-but-unneeded constraints (e.g. opencv's numpy>=2).
Write-Host "[3/4] pip install --no-deps -r $PipReq"
& $py -m pip install --no-deps --no-warn-script-location -r $PipReq
if ($LASTEXITCODE -ne 0) { throw "pip install failed ($LASTEXITCODE)" }
Write-Host "      pip check (informational; the reference env reports the same lines):"
& $py -m pip check
$global:LASTEXITCODE = 0

# ---- 4. check ---------------------------------------------------------------
if (-not $SkipCheck) {
    Write-Host "[4/4] environment check"
    $chk = @((Join-Path $PSScriptRoot "check_env.py"))
    $modelsDir = Join-Path $Repo "models"
    if (Test-Path $modelsDir) {
        $model = Get-ChildItem $modelsDir -File -Filter "omni_model_*" | Sort-Object Name | Select-Object -Last 1
        if ($model) { $chk += @("--model", $model.FullName) }
    }
    & $py @chk
    if ($LASTEXITCODE -ne 0) { throw "check_env.py failed ($LASTEXITCODE)" }
}

Write-Host ""
Write-Host "Done. Analysis Python: $py"
Write-Host "Next: & `"$py`" scripts\run_dataset_pipeline.py datasets\<YYMMDD>.yaml --plan"
