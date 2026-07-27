@echo off
setlocal

set "REPO_ROOT=%~dp0"
cd /d "%REPO_ROOT%"

set "PYTHON_EXE=%REPO_ROOT%\.venv311fa\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo Missing .venv311fa. Please recreate the flash-attn environment first.
  exit /b 1
)

set "CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
if exist "%CUDA_ROOT%" (
  set "CUDA_HOME=%CUDA_ROOT%"
  set "CUDA_PATH=%CUDA_ROOT%"
  set "PATH=%CUDA_ROOT%\bin;%PATH%"
)

echo Running main.py with flash-attn environment...
echo Usage: run_main_flashattn.bat [main.py arguments...]
echo.

"%PYTHON_EXE%" ".\main.py" %*
