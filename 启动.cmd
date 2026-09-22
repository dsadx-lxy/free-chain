@echo off
cd /d "%~dp0"
title FJC Chain Calculator

rem IMPORTANT: keep this file pure ASCII.
rem cmd.exe reads .cmd in the OEM codepage (GBK here), so UTF-8 Chinese text
rem desyncs the parser and breaks the script - the same lesson the MingZhang
rem FMS launcher records in its own header.
rem
rem No "chcp 65001" here on purpose: server.py's banner is plain GBK-safe
rem Chinese, and Python writes it through the Windows console API, so it
rem renders correctly without touching the console codepage. One less thing
rem that can differ between machines.

set "PY=D:\anaconda3\python.exe"
if not exist "%PY%" set "PY=python"

"%PY%" server.py --open

echo.
echo   Server stopped.
pause
