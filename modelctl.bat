@echo off
setlocal

if "%1"=="" (
  echo Usage: modelctl.bat [input_file] [checkpoint_dir] [device]
  echo Translates C source to Rust via latent-space IR
  goto end
)

set INPUT=%1
set CKDIR=%2
set DEVICE=%3

rem Heuristic: if %1 is a .pt and %2 is a source file, they might be swapped
echo "%INPUT%" | findstr /i ".pt" >nul
if not errorlevel 1 (
  if not "%CKDIR%"=="" (
    echo "%CKDIR%" | findstr /i ".c .rs" >nul
    if not errorlevel 1 (
      echo Swapping detected args: using %2 as input and %1 as checkpoint
      set INPUT=%2
      set CKDIR=%1
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

set ARGS=c2rust %INPUT%
if not "%CKDIR%"=="" set ARGS=%ARGS% --checkpoint-dir %CKDIR%
if not "%DEVICE%"=="" set ARGS=%ARGS% --device %DEVICE%

set PYTHONPATH=%~dp0
python src\tools\run_inference.py %ARGS%

:end
endlocal
