@echo off

REM Check if an argument was provided
if "%~1"=="" (
    echo Drag and drop a shader file onto this script
    pause
    exit /b
)

for %%x in (%*) do (
    bin\ShaderCompile.exe /O 3 -ver 30  -shaderpath "%cd%" %%x
)
pause

REM #-ver 20b