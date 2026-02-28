@echo off
setlocal enabledelayedexpansion

if "%1"=="" (
    echo Usage: convert.bat [input_c_file] [output_rust_file]
    goto :end
)

set INPUT_C=%1
set OUTPUT_RS=%2
if "%OUTPUT_RS%"=="" (
    set OUTPUT_RS=output.rs
)

rem Ensure the project root is in PYTHONPATH
set PYTHONPATH=%PYTHONPATH%;.

echo [Pipeline] Starting conversion: %INPUT_C% --^> %OUTPUT_RS%

echo [Stage] Converting C to Rust (via latent-space IR)...
python src\tools\run_inference.py c2rust "%INPUT_C%" --raw > "%OUTPUT_RS%"
if errorlevel 1 (
    echo [Error] Conversion failed.
    goto :end
)

echo [Pipeline] Success! Rust code generated in %OUTPUT_RS%

:end
endlocal
