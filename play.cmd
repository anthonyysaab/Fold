@echo off
REM One command: play live Arena poker continuously until Ctrl+C or shutdown.
REM
REM Serves whatever artifacts\approved.json names once a candidate is promoted,
REM and the heuristic otherwise; pass --aggressive or --standard to force a
REM heuristic, or any live_session.py flag (e.g. --session-seconds 3600).
REM
REM Credentials resolve from this folder's .arena-credentials automatically.
cd /d "%~dp0"

if not exist ".arena-credentials" if "%ARENA_CREDENTIALS%"=="" (
    echo play.cmd: no .arena-credentials here and ARENA_CREDENTIALS is unset 1>&2
    exit /b 1
)

REM Unbuffered so the console shows decisions as they happen.
python -u live_session.py %*
set "EXITCODE=%ERRORLEVEL%"
echo.
echo session exited with code %EXITCODE%
pause
exit /b %EXITCODE%
