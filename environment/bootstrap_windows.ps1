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
    [switch]$SkipCheck,
    [switch]$ForceInstall    # install Miniconda into -CondaRoot even if another conda exists (tests the installer branch)
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
# A shell where another conda is active exports CONDA_ROOT / CONDA_PREFIX / CONDA_EXE; a different conda.bat
# then reports that installation as its base and creates the env there (seen 2026-09-14). Drop them.
foreach ($v in @("CONDA_ROOT", "CONDA_PREFIX", "CONDA_EXE", "CONDA_DEFAULT_ENV", "CONDA_PYTHON_EXE",
                 "CONDA_SHLVL", "CONDA_PROMPT_MODIFIER", "_CONDA_ROOT", "_CONDA_EXE", "CONDA_ENVS_PATH", "CONDA_ENVS_DIRS")) {
    if (Test-Path "Env:$v") { Remove-Item "Env:$v" }
}
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
$conda = if ($ForceInstall) { $null } else { Find-Conda }
if (-not $conda) {
    # The Miniconda installer exits with code 2 (nothing installed, no message) when the target path is
    # long, contains spaces or non-ASCII characters (tested 2026-09-14: a 125-character path failed,
    # C:\TEMP\mc_fresh_test worked). Keep the default %USERPROFILE%\miniconda3 or pass a short -CondaRoot.
    if ($CondaRoot.Length -gt 64 -or $CondaRoot -match '[\s]' -or $CondaRoot -match '[^\x20-\x7E]') {
        throw "CondaRoot must be a short ASCII path without spaces (got $($CondaRoot.Length) chars: $CondaRoot)"
    }
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
# Part of the explicit spec comes from Anaconda's defaults channel (repo.anaconda.com/pkgs/main), whose
# Terms of Service a non-interactive conda >= 24.x must have accepted, or `conda create` stops.
# Same terms the lab's existing Anaconda installations run under. Older conda has no `tos` command (ignored).
foreach ($ch in @("https://repo.anaconda.com/pkgs/main", "https://repo.anaconda.com/pkgs/r", "https://repo.anaconda.com/pkgs/msys2")) {
    & $conda tos accept --override-channels --channel $ch 2>$null | Out-Null
}
$global:LASTEXITCODE = 0

# ---- 2. env from the explicit spec ----------------------------------------
# The env lives under the conda installation that conda.bat belongs to (<root>\condabin\conda.bat),
# addressed by --prefix so no other installation's settings can redirect it.
$base = Split-Path -Parent (Split-Path -Parent (Resolve-Path $conda).Path)
$envPrefix = Join-Path $base "envs\$EnvName"
$py = Join-Path $envPrefix "python.exe"
if (Test-Path $py) {
    Write-Host "[2/4] env '$EnvName' already exists at $envPrefix (reusing)"
} else {
    Write-Host "[2/4] creating env at $envPrefix from $Explicit (about 3 GB of downloads)"
    & $conda create -y --prefix $envPrefix --file $Explicit
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

# cellpose_omni downloads GUI assets into ~\.omnipose at package import when they are missing, and two
# of those URLs answer 404 (seen 2026-09-15: gui/logo.png, gui/gamma.svg), which makes `import cellpose_omni`
# fail on a fresh PC. Both only decorate the GUI window, so 1x1 stand-ins take their place.
$guiAssets = @{
    "logo.png"  = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    "gamma.svg" = "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxIiBoZWlnaHQ9IjEiLz4="
}
foreach ($name in $guiAssets.Keys) {
    $assetPath = Join-Path $env:USERPROFILE (".omnipose\" + $name)
    if (-not (Test-Path $assetPath)) {
        New-Item -ItemType Directory -Force (Split-Path -Parent $assetPath) | Out-Null
        [IO.File]::WriteAllBytes($assetPath, [Convert]::FromBase64String($guiAssets[$name]))
        Write-Host "      placed a 1x1 stand-in for the cellpose_omni GUI asset $assetPath"
    }
}

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
