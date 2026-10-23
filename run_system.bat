@echo off
title Fire & Smoke Security Monitor
cd /d "%~dp0"
echo ========================================================
echo   FIRE ^& SMOKE SECURITY MONITORING DASHBOARD
echo ========================================================
echo.
echo Dashboard URL: http://localhost:5000
echo.
.venv\Scripts\python.exe app.py
pause
