@echo off
setlocal
cd /d "%~dp0"

set "GIT=C:\Program Files\Git\cmd\git.exe"
if not exist "%GIT%" set "GIT=C:\Program Files\Git\bin\git.exe"
if not exist "%GIT%" (
    echo Could not find git.exe in C:\Program Files\Git. Falling back to PATH.
    set "GIT=git"
)

echo === Using git: %GIT% ===
(
echo === Using git: %GIT% ===

"%GIT%" rev-parse --is-inside-work-tree 2>nul
if errorlevel 1 (
    echo Initialising new repo...
    "%GIT%" init -b main
) else (
    echo Already a git repo.
)

echo.
echo === Configuring identity ===
"%GIT%" config user.name  "Cian Woodsy"
"%GIT%" config user.email "cian6woodsy@gmail.com"

echo.
echo === Staging files ===
"%GIT%" add -A

echo.
echo === Initial commit ===
"%GIT%" commit -m "chore: initial commit of dissertation project"

echo.
echo === Git log ===
"%GIT%" log --oneline -5

echo.
echo DONE.
) > _git_init.log 2>&1
type _git_init.log
pause
