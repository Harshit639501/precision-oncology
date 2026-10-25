@echo off
setlocal
cd /d "%~dp0"

curl.exe --silent --fail http://127.0.0.1:8000/health >nul 2>&1
if errorlevel 1 (
    echo Starting the Precision Oncology API...
    start "Precision Oncology API" /min py -3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
)

set /a attempts=0
:wait_for_api
curl.exe --silent --fail http://127.0.0.1:8000/health >nul 2>&1
if not errorlevel 1 goto api_ready
set /a attempts+=1
if %attempts% geq 90 goto api_failed
timeout /t 2 /nobreak >nul
goto wait_for_api

:api_ready
curl.exe --silent --fail http://127.0.0.1:5500/index.html >nul 2>&1
if errorlevel 1 start "Precision Oncology UI" /min py -3 -m http.server 5500
timeout /t 1 /nobreak >nul
start "" http://127.0.0.1:5500/index.html

echo Precision Oncology is ready. You can sign in.
endlocal
exit /b 0

:api_failed
echo.
echo The FastAPI service did not become ready. Check that Python and the project dependencies are installed.
echo You can also open a terminal in this folder and run:
echo py -3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
pause
endlocal
exit /b 1