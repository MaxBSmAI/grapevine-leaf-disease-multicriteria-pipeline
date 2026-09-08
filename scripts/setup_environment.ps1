[CmdletBinding()]
param(
    [ValidateSet("cuda128", "cpu")]
    [string]$Backend = "cuda128",
    [string]$PythonCommand = "py"
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$EnvironmentPath = Join-Path $RepositoryRoot ".venv"
$PythonPath = Join-Path $EnvironmentPath "Scripts\python.exe"

if ($PythonCommand -eq "py" -and -not (Get-Command py -ErrorAction SilentlyContinue)) {
    $BundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
    if (Test-Path -LiteralPath $BundledPython) {
        $PythonCommand = $BundledPython
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        $PythonCommand = (Get-Command python).Source
    }
    else {
        throw "Python 3.12 was not found. Supply -PythonCommand with its executable path."
    }
}

if (-not (Test-Path -LiteralPath $PythonPath)) {
    if ($PythonCommand -eq "py") {
        & $PythonCommand -3.12 -m venv $EnvironmentPath
    }
    else {
        & $PythonCommand -m venv $EnvironmentPath
    }
}

& $PythonPath -m pip install --upgrade "pip==25.1.1" "setuptools==80.9.0" "wheel==0.45.1"
if ($Backend -eq "cuda128") {
    & $PythonPath -m pip install "torch==2.7.1" "torchvision==0.22.1" --index-url "https://download.pytorch.org/whl/cu128"
}
else {
    & $PythonPath -m pip install "torch==2.7.1" "torchvision==0.22.1" --index-url "https://download.pytorch.org/whl/cpu"
}
& $PythonPath -m pip install -r (Join-Path $RepositoryRoot "requirements-lock.txt")
& $PythonPath -m pip install --no-deps -e $RepositoryRoot
& $PythonPath -m pip check
& $PythonPath -c "import torch; print({'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'cuda': torch.version.cuda})"

Write-Host "Environment ready: $EnvironmentPath"
