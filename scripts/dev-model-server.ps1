[CmdletBinding()]
param(
    [string]$DataRoot = "",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ServerArgs = @()
)

$ErrorActionPreference = "Stop"
$parkingProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$parkingPython = Join-Path $parkingProjectRoot ".venv\Scripts\python.exe"
$parkingServer = Join-Path $parkingProjectRoot "fsd_trainer\src\gta_fsd\server.py"
$parkingConfig = Join-Path $parkingProjectRoot "fsd_trainer\train_config.toml"

if ($DataRoot.Trim()) {
    $env:FSD_DATA_ROOT = $DataRoot.Trim()
}

if (-not (Test-Path -LiteralPath $parkingPython -PathType Leaf)) {
    throw "Python environment not found at $parkingPython"
}

& $parkingPython $parkingServer --config $parkingConfig @ServerArgs
exit $LASTEXITCODE
