param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$mainPy = Join-Path $PSScriptRoot "main.py"
if (-not (Test-Path $mainPy)) {
    Write-Host "main.py not found next to this script: $mainPy"
    exit 1
}

$mainPyResolved = (Resolve-Path $mainPy).Path
$pattern = [Regex]::Escape($mainPyResolved)

$targets = Get-CimInstance Win32_Process -Filter "name='python.exe'" |
    Where-Object { $_.CommandLine -match $pattern }

if (-not $targets) {
    Write-Host "No running CellPose main.py process found."
    exit 0
}

Write-Host "Found process(es):"
$targets | Select-Object ProcessId, CommandLine | Format-Table -AutoSize

if ($DryRun) {
    Write-Host "DryRun mode: no process terminated."
    exit 0
}

$killed = 0
foreach ($proc in $targets) {
    try {
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop
        Write-Host "Stopped PID $($proc.ProcessId)"
        $killed++
    } catch {
        Write-Host "Failed to stop PID $($proc.ProcessId): $($_.Exception.Message)"
    }
}

Write-Host "Done. Stopped $killed process(es)."
