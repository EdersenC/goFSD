[CmdletBinding()]
param(
    [string]$DataRoot = "",
    [switch]$SkipWebBuild,
    [Parameter(ValueFromRemainingArguments = $true, Position = 0)]
    [string[]]$BackendArgs = @()
)

$ErrorActionPreference = "Stop"
$stopSignProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$stopSignWebRoot = Join-Path $stopSignProjectRoot "backend\cmd\web"
$stopSignBackendMode = if ($BackendArgs.Count -eq 0) { "serve" } else { $BackendArgs[0].Trim() }
$stopSignShouldBuildWeb = (-not $SkipWebBuild) -and ($stopSignBackendMode -eq "serve")

if ($stopSignShouldBuildWeb) {
    $stopSignNpmCommand = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if (-not $stopSignNpmCommand) {
        throw "Node.js npm.cmd was not found on PATH; install the Stop Sign Lab web dependencies before starting the backend"
    }
    $stopSignWebBuild = Start-Process `
        -FilePath $stopSignNpmCommand.Source `
        -ArgumentList @("--prefix", $stopSignWebRoot, "run", "build") `
        -NoNewWindow `
        -Wait `
        -PassThru
    if ($stopSignWebBuild.ExitCode -ne 0) {
        throw "Stop Sign Lab web build failed with exit code $($stopSignWebBuild.ExitCode)"
    }
}

# WSL can pass Windows PowerShell a truncated PATHEXT, which breaks Go's downloaded toolchain lookup.
$stopSignPathExtensions = @($env:PATHEXT -split ";" | Where-Object { $_ })
if ($stopSignPathExtensions -notcontains ".EXE") {
    $env:PATHEXT = (@(".COM", ".EXE", ".BAT", ".CMD") + $stopSignPathExtensions) -join ";"
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

Set-Location (Join-Path $stopSignProjectRoot "backend")
$stopSignGoCommand = Get-Command go.exe -ErrorAction SilentlyContinue
$stopSignGoPath = if ($stopSignGoCommand) { $stopSignGoCommand.Source } else { "" }
if (-not $stopSignGoPath) {
    $stopSignGoCandidate = Join-Path $env:ProgramFiles "Go\bin\go.exe"
    if (Test-Path -LiteralPath $stopSignGoCandidate -PathType Leaf) {
        $stopSignGoPath = $stopSignGoCandidate
    }
}
if (-not $stopSignGoPath) {
    throw "Windows Go was not found on PATH or under Program Files\Go\bin"
}

$stopSignGoArguments = @("run", "./cmd") + $BackendArgs
$stopSignGoArgumentLine = ($stopSignGoArguments | ForEach-Object {
    ConvertTo-WindowsCommandLineArgument $_
}) -join " "
$stopSignGoProcess = Start-Process `
    -FilePath $stopSignGoPath `
    -ArgumentList $stopSignGoArgumentLine `
    -NoNewWindow `
    -Wait `
    -PassThru
exit $stopSignGoProcess.ExitCode
