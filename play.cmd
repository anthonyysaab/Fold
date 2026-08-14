@echo off
REM One command: play live Arena poker continuously until Ctrl+C or shutdown.
REM Credentials resolve from this folder's .arena-credentials automatically.
cd /d "%~dp0"
python live_session.py %*
echo.
pause
