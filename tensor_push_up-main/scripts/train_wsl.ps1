param(
    [string]$Distro = "Ubuntu",
    [switch]$Smoke,
    [switch]$AllowSingleClass,
    [string]$Config = "configs/train.yaml",
    [string]$Venv = ".venv-wsl"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$repoRootWsl = $repoRoot -replace '\\', '/'

if ($repoRootWsl.Length -ge 2 -and $repoRootWsl[1] -eq ':') {
    $drive = $repoRootWsl.Substring(0, 1).ToLower()
    $pathPart = $repoRootWsl.Substring(2)
    $repoRootWsl = "/mnt/$drive$pathPart"
}

$trainArgs = @("--venv", $Venv, "--config", $Config)
if ($Smoke) {
    $trainArgs += "--smoke"
}
if ($AllowSingleClass) {
    $trainArgs += "--allow-single-class"
}

$argString = [string]::Join(' ', ($trainArgs | ForEach-Object {
    if ($_ -match '\s') { "'$_'" } else { $_ }
}))

$command = "cd '$repoRootWsl' && bash scripts/train_wsl.sh $argString"

Write-Host "[wsl] $command"
wsl -d $Distro bash -lc $command
