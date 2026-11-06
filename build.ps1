# Builds dist\VLMPy (PyInstaller, onedir) and then the installer (Inno Setup) in one go.
# Usage: .\build.ps1 [-Version 1.1.0] [-SkipInstaller] [-PythonVersion 3.15]
param(
    [string]$Version = "1.1.0",
    [string]$PythonVersion = "3.15",
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$src = Join-Path $root "src\vlm"
$img = Join-Path $root "docs\img"
$utils = Join-Path $src "utils"

py "-$PythonVersion" $utils\make_splash.py

$excludes = "IPython","pytest","sympy","tkinter.test","scipy","pandas","numba","llvmlite",
            "statsmodels","pyarrow","lxml","openpyxl","jedi","zmq","gevent","kaleido"
$excludeArgs = $excludes | ForEach-Object { "--exclude-module"; $_ }

py "-$PythonVersion" -m PyInstaller --noconfirm --clean --onedir --windowed `
    --name VLMPy `
    --icon "$img\icon.ico" `
    --version-file "$utils\version_info.txt" `
    --manifest "$utils\VLMPy.manifest" `
    --noupx `
    --paths "$src" `
    --add-data "$src\templates;vlm/templates" `
    --add-data "$src\static;vlm/static" `
    --add-data "$src\data\default_*.txt;vlm/data" `
    @excludeArgs `
    --distpath "$root\dist" --workpath "$root\build" --specpath "$root\build" `
    --hidden-import app `
    "$src\launcher.py"
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

# Drop Qt payload the app never uses (devtools, debug resources, translations, QML).
$qt = Join-Path $root "dist\VLMPy\_internal\PySide6"
Get-ChildItem "$qt\resources" -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like "*.debug.*" -or $_.Name -like "qtwebengine_devtools_resources*" } | Remove-Item -Force
Get-ChildItem "$qt\translations" -Filter *.qm -File -ErrorAction SilentlyContinue | Remove-Item -Force
Get-ChildItem "$qt\translations\qtwebengine_locales" -Filter *.pak -File -ErrorAction SilentlyContinue |
    Where-Object { $_.BaseName -notin "en-US","es","es-419" } | Remove-Item -Force
Remove-Item "$qt\qml" -Recurse -Force -ErrorAction SilentlyContinue

if ($SkipInstaller) { return }

$iscc = (Get-Command iscc -ErrorAction SilentlyContinue).Source
if (-not $iscc) {
    $iscc = @("$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
              "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
              "$env:ProgramFiles\Inno Setup 6\ISCC.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $iscc) { throw "Inno Setup 6 not found (winget install JRSoftware.InnoSetup)" }

& $iscc "/DAppVersion=$Version" "$root\installer\VLMPy.iss"
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed" }
Write-Host "Installer: $root\dist\installer\VLMPy-Setup-$Version.exe"
