@echo off
rem Simple helper to setup env, train models, run tests, and save artifacts
setlocal

if "%1"=="" (
  set CMD=all
) else (
  set CMD=%1
)

rem Create virtualenv
if not exist .venv (python -m venv .venv)
call .venv\Scripts\activate.bat

rem Upgrade pip and install requirements
python -m pip install --upgrade pip
if exist requirements.txt (
  pip install -r requirements.txt
)

rem Quick mode runs short jobs (1 epoch) to verify pipeline
if /I "%CMD%"=="quick" (
  echo Running quick setup + train ^(1 epoch^)
  set PYTHONPATH=%~dp0
  python src\training\train_c_to_ir.py --epochs 1
  python src\training\train_ir_to_rust.py --epochs 1
  goto end
)

if /I "%CMD%"=="setup" (
  echo Environment setup complete.
  goto end
)

if /I "%CMD%"=="train-c2ir" (
  echo Training C->IR model (use EPOCHS env var to override)
  set PYTHONPATH=%~dp0
  if "%EPOCHS%"=="" (
    python src\training\train_c_to_ir.py
  ) else (
    python src\training\train_c_to_ir.py --epochs %EPOCHS%
  )
  goto end
)

if /I "%CMD%"=="train-ir2rust" (
  echo Training IR->Rust model (use EPOCHS env var to override)
  set PYTHONPATH=%~dp0
  if "%EPOCHS%"=="" (
    python src\training\train_ir_to_rust.py
  ) else (
    python src\training\train_ir_to_rust.py --epochs %EPOCHS%
  )
  goto end
)

if /I "%CMD%"=="test" (
  echo Running tests
  set PYTHONPATH=%~dp0
  python -m pytest -q
  goto end
)

if /I "%CMD%"=="all" (
  echo Full pipeline: train both models and run tests
  set PYTHONPATH=%~dp0
  rem Train stage1 then stage2
  if "%EPOCHS%"=="" (
    python src\training\train_c_to_ir.py
    python src\training\train_ir_to_rust.py
  ) else (
    python src\training\train_c_to_ir.py --epochs %EPOCHS%
    python src\training\train_ir_to_rust.py --epochs %EPOCHS%
  )
  echo Running tests
  python -m pytest -q
  goto end
)

:end
echo Done.
endlocal
