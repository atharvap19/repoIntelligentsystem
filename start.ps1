# Starts the backend (port 8000) and frontend (port 5173), each in its own window.
# Usage, from the repointplat folder:  .\start.ps1

$root = $PSScriptRoot
$backend = Join-Path $root "backend"
$frontend = Join-Path $root "frontend"

Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$backend'; & '.\.venv\Scripts\Activate.ps1'; uvicorn app.main:app --reload --port 8000"
# The backend takes several seconds to import; open the frontend only once it
# answers, so the UI does not load into "backend is not reachable".
Write-Host "Waiting for the backend..."
for ($i = 0; $i -lt 60; $i++) {
    try { Invoke-WebRequest http://localhost:8000/ -UseBasicParsing -TimeoutSec 2 | Out-Null; break }
    catch { Start-Sleep -Seconds 1 }
}

Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$frontend'; npm run dev"

Write-Host "Backend:  http://localhost:8000"
Write-Host "Frontend: http://localhost:5173"
