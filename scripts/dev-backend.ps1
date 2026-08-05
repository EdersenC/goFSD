[CmdletBinding()]
param(
    [string]$DataRoot = ""
)

$ErrorActionPreference = "Stop"
$parkingProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if ($DataRoot.Trim()) {
    $env:FSD_DATA_ROOT = $DataRoot.Trim()
} elseif (-not $env:FSD_DATA_ROOT) {
    $env:FSD_DATA_ROOT = "S:\fsd_fivem_data"
}

Set-Location (Join-Path $parkingProjectRoot "backend")
$parkingGoCommand = Get-Command go.exe -ErrorAction SilentlyContinue
$parkingGoPath = if ($parkingGoCommand) { $parkingGoCommand.Source } else { "" }
if (-not $parkingGoPath) {
    $parkingGoCandidate = Join-Path $env:ProgramFiles "Go\bin\go.exe"
    if (Test-Path -LiteralPath $parkingGoCandidate -PathType Leaf) {
        $parkingGoPath = $parkingGoCandidate
    }
}
if (-not $parkingGoPath) {
    throw "Windows Go was not found on PATH or under Program Files\Go\bin"
}

& $parkingGoPath run ./cmd
exit $LASTEXITCODE
