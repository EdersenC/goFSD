[CmdletBinding()]
param(
    [string]$DataRoot = "",
    [string]$BackendUrl = "http://127.0.0.1:8080",
    [string]$ModelUrl = "http://127.0.0.1:8090",
    [switch]$NoModel,
    [switch]$NoBrowser,
    [switch]$SkipWebBuild,
    [ValidateRange(1, 300)]
    [int]$WaitSeconds = 90,
    [switch]$HealthCheckOnly,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$parkingProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$parkingBackendScript = Join-Path $PSScriptRoot "dev-backend.ps1"
$parkingModelScript = Join-Path $PSScriptRoot "dev-model-server.ps1"
$parkingPython = Join-Path $parkingProjectRoot ".venv\Scripts\python.exe"
$parkingBackendService = "stop-sign-lab-backend"
$parkingModelService = "stop-sign-lab-model"

function Assert-LoopbackHttpUrl {
    param(
        [string]$Name,
        [string]$Value
    )

    $parsed = $null
    if (-not [Uri]::TryCreate($Value, [UriKind]::Absolute, [ref]$parsed)) {
        throw "$Name must be an absolute URL"
    }
    if ($parsed.Scheme -ne "http") {
        throw "$Name must use http"
    }
    if ($parsed.Host -ne "localhost" -and $parsed.Host -ne "127.0.0.1" -and $parsed.Host -ne "::1") {
        throw "$Name must use a loopback host; Stop Sign Lab control endpoints are not authenticated"
    }
}

function ConvertTo-PowerShellLiteral {
    param([AllowEmptyString()][string]$Value)

    return "'" + $Value.Replace("'", "''") + "'"
}

function New-ServiceCommand {
    param(
        [string]$Title,
        [string]$Script,
        [switch]$IncludeSkipWebBuild
    )

    $parts = @(
        '$Host.UI.RawUI.WindowTitle = ' + (ConvertTo-PowerShellLiteral $Title) + ';',
        '&',
        (ConvertTo-PowerShellLiteral $Script)
    )
    if ($DataRoot.Trim()) {
        $parts += @('-DataRoot', (ConvertTo-PowerShellLiteral $DataRoot.Trim()))
    }
    if ($IncludeSkipWebBuild) {
        $parts += '-SkipWebBuild'
    }
    return $parts -join ' '
}

function Test-LocalService {
    param(
        [string]$Url,
        [string]$ExpectedService
    )

    try {
        $response = Invoke-RestMethod `
            -Uri ($Url.TrimEnd('/') + '/healthz') `
            -TimeoutSec 1
        return ($response.status -is [string]) `
            -and ($response.service -is [string]) `
            -and ($response.status -ceq 'ok') `
            -and ($response.service -ceq $ExpectedService)
    } catch {
        return $false
    }
}

function Start-ServiceWindow {
    param(
        [string]$Name,
        [string]$Command
    )

    $powerShellPath = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
    $encodedServiceCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($Command))
    $powerShellPathLiteral = ConvertTo-PowerShellLiteral $powerShellPath
    $nameLiteral = ConvertTo-PowerShellLiteral $Name
    $runnerCommand = @"
`$parkingService = Start-Process -FilePath $powerShellPathLiteral -ArgumentList @('-NoLogo', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-EncodedCommand', '$encodedServiceCommand') -NoNewWindow -Wait -PassThru
if (`$parkingService.ExitCode -eq 0) {
    Write-Host ($nameLiteral + ' stopped.')
} else {
    Write-Host ($nameLiteral + ' stopped with exit code ' + `$parkingService.ExitCode + '.') -ForegroundColor Red
}
"@
    $encodedRunnerCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($runnerCommand))
    $process = Start-Process `
        -FilePath $powerShellPath `
        -ArgumentList @(
            '-NoLogo',
            '-NoProfile',
            '-ExecutionPolicy',
            'Bypass',
            '-NoExit',
            '-EncodedCommand',
            $encodedRunnerCommand
        ) `
        -WindowStyle Normal `
        -PassThru
    Write-Host "start $Name terminal (pid=$($process.Id))"
}

function Wait-ForLocalService {
    param(
        [string]$Name,
        [string]$Url,
        [string]$ExpectedService,
        [int]$TimeoutSeconds
    )

    Write-Host "wait  $Name (up to $TimeoutSeconds seconds)"
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        if (Test-LocalService -Url $Url -ExpectedService $ExpectedService) {
            Write-Host "ok    ${Name}: $Url"
            return $true
        }
        Start-Sleep -Milliseconds 500
    }
    Write-Warning "$Name did not become ready at $Url; check its terminal for the exact error"
    return $false
}

Assert-LoopbackHttpUrl -Name "BackendUrl" -Value $BackendUrl
Assert-LoopbackHttpUrl -Name "ModelUrl" -Value $ModelUrl

if (-not (Test-Path -LiteralPath $parkingBackendScript -PathType Leaf)) {
    throw "Backend launcher not found at $parkingBackendScript"
}
if (-not (Test-Path -LiteralPath $parkingModelScript -PathType Leaf)) {
    throw "Model launcher not found at $parkingModelScript"
}

$parkingBackendCommand = New-ServiceCommand `
    -Title "Stop Sign Lab - backend" `
    -Script $parkingBackendScript `
    -IncludeSkipWebBuild:$SkipWebBuild
$parkingModelCommand = New-ServiceCommand `
    -Title "Stop Sign Lab - model server" `
    -Script $parkingModelScript

Write-Host "Stop Sign Lab start"
if ($DataRoot.Trim()) {
    Write-Host "data  $($DataRoot.Trim())"
}

if ($DryRun) {
    Write-Host "dry   backend: $parkingBackendCommand"
    if ($NoModel) {
        Write-Host "skip  model server (--no-model)"
    } elseif (-not (Test-Path -LiteralPath $parkingPython -PathType Leaf)) {
        Write-Host "skip  model server (Python environment is missing)"
    } else {
        Write-Host "dry   model server: $parkingModelCommand"
    }
    if ($NoBrowser) {
        Write-Host "skip  browser (--no-browser)"
    } else {
        Write-Host "dry   browser: $BackendUrl"
    }
    exit 0
}

if ($HealthCheckOnly) {
    $parkingBackendIdentityMatches = Test-LocalService `
        -Url $BackendUrl `
        -ExpectedService $parkingBackendService
    if ($parkingBackendIdentityMatches) {
        Write-Host "ok    backend identity: $parkingBackendService"
    } else {
        Write-Warning "backend at $BackendUrl is not $parkingBackendService"
    }

    $parkingModelIdentityMatches = $true
    if (-not $NoModel) {
        $parkingModelIdentityMatches = Test-LocalService `
            -Url $ModelUrl `
            -ExpectedService $parkingModelService
        if ($parkingModelIdentityMatches) {
            Write-Host "ok    model server identity: $parkingModelService"
        } else {
            Write-Warning "model server at $ModelUrl is not $parkingModelService"
        }
    }

    if (-not $parkingBackendIdentityMatches -or -not $parkingModelIdentityMatches) {
        exit 1
    }
    exit 0
}

$parkingBackendReady = Test-LocalService `
    -Url $BackendUrl `
    -ExpectedService $parkingBackendService
if ($parkingBackendReady) {
    Write-Host "ok    backend already running: $BackendUrl"
} else {
    Start-ServiceWindow -Name "backend" -Command $parkingBackendCommand
}

$parkingModelWanted = -not $NoModel
$parkingModelReady = $false
if (-not $parkingModelWanted) {
    Write-Host "skip  model server (--no-model)"
} elseif (-not (Test-Path -LiteralPath $parkingPython -PathType Leaf)) {
    Write-Warning "Python environment is missing at $parkingPython; Collect and Prepare still work, but Train and Drive need the model server"
} else {
    $parkingModelReady = Test-LocalService `
        -Url $ModelUrl `
        -ExpectedService $parkingModelService
    if ($parkingModelReady) {
        Write-Host "ok    model server already running: $ModelUrl"
    } else {
        Start-ServiceWindow -Name "model server" -Command $parkingModelCommand
    }
}

if (-not $parkingBackendReady) {
    $parkingBackendReady = Wait-ForLocalService `
        -Name "backend" `
        -Url $BackendUrl `
        -ExpectedService $parkingBackendService `
        -TimeoutSeconds $WaitSeconds
}
if ($parkingModelWanted -and (Test-Path -LiteralPath $parkingPython -PathType Leaf) -and -not $parkingModelReady) {
    $parkingModelReady = Wait-ForLocalService `
        -Name "model server" `
        -Url $ModelUrl `
        -ExpectedService $parkingModelService `
        -TimeoutSeconds $WaitSeconds
}

if (-not $parkingBackendReady) {
    Write-Error "Stop Sign Lab could not start because the backend is unavailable"
    exit 1
}

if (-not $NoBrowser) {
    Start-Process $BackendUrl
    Write-Host "open  $BackendUrl"
}

if ($parkingModelWanted -and -not $parkingModelReady) {
    Write-Warning "The workspace is open for Collect and Prepare; fix the model-server terminal before using Train or Drive"
}
Write-Host "ready Keep the service terminals open while you use Stop Sign Lab."
