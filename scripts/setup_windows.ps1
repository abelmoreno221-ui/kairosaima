# Crea C:\Users\<tu usuario>\pmarb, descarga el código, instala y lanza el PASO 1 (build pista A).
# Requisitos: Python 3.10+ y Git for Windows. Uso, desde PowerShell:
#   powershell -ExecutionPolicy Bypass -File setup_windows.ps1
$ErrorActionPreference = "Stop"
$Repo   = "https://github.com/abelmoreno221-ui/kairosaima.git"
$Branch = "claude/fervent-wright-vltalv"
$Dir    = Join-Path $HOME "pmarb"

foreach ($cmd in "git", "python") {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) { throw "Falta '$cmd'. Instálalo y vuelve a lanzar el script." }
}

if (Test-Path (Join-Path $Dir ".git")) {
    git -C $Dir fetch origin $Branch
    git -C $Dir checkout $Branch
    git -C $Dir pull origin $Branch
} else {
    git clone -b $Branch $Repo $Dir
}
Set-Location $Dir

if (-not (Test-Path ".venv")) { python -m venv .venv }
$Py = Join-Path $Dir ".venv\Scripts\python.exe"
& $Py -m pip install -q --upgrade pip
& $Py -m pip install -q -e .
& $Py -m unittest discover -s tests -t . 2>&1 | Select-Object -Last 2

# PASO 1. El log recoge también stderr (HTTP-RETRY / HTTP-ERROR).
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$fd = @()
if ($args.Count -gt 0) { $fd = @("--fd", $args[0]) }   # carpeta con los 5 CSV si football-data da 403
& $Py -m pmarb build --track A @fd 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath build_A.log

Write-Host "`n=== Pega esto en el chat ==="
Write-Host "Líneas del log: $((Get-Content build_A.log).Count)"
if (Test-Path dataset_A.drops.csv) { Get-Content dataset_A.drops.csv }
Write-Host "Carpeta: $Dir"
