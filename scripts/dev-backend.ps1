[CmdletBinding()]
param(
    [string]$DataRoot = "",
    [switch]$SkipWebBuild,
    [Parameter(ValueFromRemainingArguments = $true, Position = 0)]
    [string[]]$BackendArgs = @()
)

$ErrorActionPreference = "Stop"
$parkingProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$parkingWebRoot = Join-Path $parkingProjectRoot "backend\cmd\web"
$parkingBackendMode = if ($BackendArgs.Count -eq 0) { "serve" } else { $BackendArgs[0].Trim() }
$parkingShouldBuildWeb = (-not $SkipWebBuild) -and ($parkingBackendMode -eq "serve")

if ($parkingShouldBuildWeb) {
    $parkingNpmCommand = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if (-not $parkingNpmCommand) {
        throw "Node.js npm.cmd was not found on PATH; install the Stop Sign Lab web dependencies before starting the backend"
    }
    $parkingWebBuild = Start-Process `
        -FilePath $parkingNpmCommand.Source `
        -ArgumentList @("--prefix", $parkingWebRoot, "run", "build") `
        -NoNewWindow `
        -Wait `
        -PassThru
    if ($parkingWebBuild.ExitCode -ne 0) {
        throw "Stop Sign Lab web build failed with exit code $($parkingWebBuild.ExitCode)"
    }
}

# WSL can pass Windows PowerShell a truncated PATHEXT, which breaks Go's downloaded toolchain lookup.
$parkingPathExtensions = @($env:PATHEXT -split ";" | Where-Object { $_ })
if ($parkingPathExtensions -notcontains ".EXE") {
    $env:PATHEXT = (@(".COM", ".EXE", ".BAT", ".CMD") + $parkingPathExtensions) -join ";"
}

function ConvertTo-WindowsCommandLineArgument {
    param([AllowEmptyString()][string]$Argument)

    if ($Argument.Length -gt 0 -and $Argument -notmatch '[\s"]') {
        return $Argument
    }

    $builder = New-Object System.Text.StringBuilder
    [void]$builder.Append('"')
    $backslashCount = 0

    foreach ($character in $Argument.ToCharArray()) {
        if ($character -eq '\') {
            $backslashCount++
            continue
        }
        if ($character -eq '"') {
            [void]$builder.Append(('\' * (($backslashCount * 2) + 1)))
            [void]$builder.Append('"')
            $backslashCount = 0
            continue
        }
        if ($backslashCount -gt 0) {
            [void]$builder.Append(('\' * $backslashCount))
            $backslashCount = 0
        }
        [void]$builder.Append($character)
    }

    if ($backslashCount -gt 0) {
        [void]$builder.Append(('\' * ($backslashCount * 2)))
    }
    [void]$builder.Append('"')
    return $builder.ToString()
}

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

$parkingGoArguments = @("run", "./cmd") + $BackendArgs
$parkingGoArgumentLine = ($parkingGoArguments | ForEach-Object {
    ConvertTo-WindowsCommandLineArgument $_
}) -join " "
$parkingGoProcess = Start-Process `
    -FilePath $parkingGoPath `
    -ArgumentList $parkingGoArgumentLine `
    -NoNewWindow `
    -Wait `
    -PassThru
exit $parkingGoProcess.ExitCode
