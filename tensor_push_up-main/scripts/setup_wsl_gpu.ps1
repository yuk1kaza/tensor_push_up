param(
    [string]$Distro = "Ubuntu",
    [string]$Venv = ".venv-wsl",
    [switch]$SkipApt
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$repoRootWsl = $repoRoot -replace '\\', '/'

if ($repoRootWsl.Length -ge 2 -and $repoRootWsl[1] -eq ':') {
    $drive = $repoRootWsl.Substring(0, 1).ToLower()
    $pathPart = $repoRootWsl.Substring(2)
    $repoRootWsl = "/mnt/$drive$pathPart"
}

$setupArgs = @("--venv", $Venv)
if ($SkipApt) {
    $setupArgs += "--skip-apt"
}

$argString = [string]::Join(' ', ($setupArgs | ForEach-Object {
    if ($_ -match '\s') { "'$_'" } else { $_ }
}))

$command = "cd '$repoRootWsl' && bash scripts/setup_wsl_gpu.sh $argString"

Write-Host "[wsl] $command"
wsl -d $Distro bash -lc $command
