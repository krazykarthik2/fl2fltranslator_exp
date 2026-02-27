@echo off
setlocal

if "%1"=="" (
  echo Usage: modelctl.bat [stage] [input_file] [checkpoint_dir] [device]
  echo Stages: c2ir, ir2rust
  goto end
)

set STAGE=%1
set INPUT=%2
set CKDIR=%3
set DEVICE=%4

rem Heuristic: if %2 is a .pt and %3 is a source file, they might be swapped
echo "%INPUT%" | findstr /i ".pt" >nul
if not errorlevel 1 (
  if not "%CKDIR%"=="" (
    echo "%CKDIR%" | findstr /i ".c .ir .rs" >nul
    if not errorlevel 1 (
      echo Swapping detected args: using %3 as input and %2 as checkpoint
      set INPUT=%3
      set CKDIR=%2
    )
  )
)

if "%INPUT%"=="" (
  echo Error: input file required
  goto end
)

if not exist "%INPUT%" (
  echo Error: input file "%INPUT%" not found
  goto end
)

if not exist .venv\Scripts\activate.bat (
  echo Activating system python; consider running do.bat setup first
  set ACT=
) else (
  call .venv\Scripts\activate.bat
)

set ARGS=%STAGE% %INPUT%
if not "%CKDIR%"=="" set ARGS=%ARGS% --checkpoint-dir %CKDIR%
if not "%DEVICE%"=="" set ARGS=%ARGS% --device %DEVICE%

set PYTHONPATH=%~dp0
python src\tools\run_inference.py %ARGS%

:end
endlocal
