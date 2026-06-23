@echo off
setlocal
cd /d "%~dp0"
if exist _build rmdir /s /q _build
python -m sphinx -E -b html -d _build\doctrees . _build\html
if errorlevel 1 pause
endlocal
