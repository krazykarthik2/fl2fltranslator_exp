@echo off
rem Simple helper to setup env, train the C->Rust model, run tests, and save artifacts
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
  python src\training\train_c_to_rust.py --epochs 1
  goto end
)

if /I "%CMD%"=="setup" (
  echo Environment setup complete.
  goto end
)

if /I "%CMD%"=="train" (
  echo Training C-^>Rust model (use EPOCHS env var to override)
  set PYTHONPATH=%~dp0
  if "%EPOCHS%"=="" (
    python src\training\train_c_to_rust.py
  ) else (
    python src\training\train_c_to_rust.py --epochs %EPOCHS%
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
  echo Full pipeline: train model and run tests
  set PYTHONPATH=%~dp0
  if "%EPOCHS%"=="" (
    python src\training\train_c_to_rust.py
  ) else (
    python src\training\train_c_to_rust.py --epochs %EPOCHS%
  )
  echo Running tests
  python -m pytest -q
  goto end
)

:end
echo Done.
endlocal
