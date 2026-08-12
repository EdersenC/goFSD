[CmdletBinding()]
param(
    [string]$DataRoot = "",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ServerArgs = @()
)

$ErrorActionPreference = "Stop"
$stopSignProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$stopSignPython = Join-Path $stopSignProjectRoot ".venv\Scripts\python.exe"
$stopSignServer = Join-Path $stopSignProjectRoot "fsd_trainer\src\gta_fsd\server.py"
$stopSignConfig = Join-Path $stopSignProjectRoot "fsd_trainer\train_config.toml"

if ($DataRoot.Trim()) {
    $env:FSD_DATA_ROOT = $DataRoot.Trim()
}

if (-not (Test-Path -LiteralPath $stopSignPython -PathType Leaf)) {
    throw "Python environment not found at $stopSignPython"
}

& $stopSignPython $stopSignServer --config $stopSignConfig @ServerArgs
exit $LASTEXITCODE
