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

set TEMP_IR=temp_pipeline.ir

rem Ensure the project root is in PYTHONPATH
set PYTHONPATH=%PYTHONPATH%;.

echo [Pipeline] Starting conversion: %INPUT_C% --^> %OUTPUT_RS%

echo [Stage 1] Converting C to IR (Deterministic GIMPLE)...
rem Clear old gimple files
if exist "%~n1.c.*t.gimple" del "%~n1.c.*t.gimple"

gcc -fdump-tree-gimple -c "%INPUT_C%" -o "%~n1.o"
if errorlevel 1 (
    echo [Error] GCC failed.
    goto :cleanup
)

rem Find the generated gimple file (look for any .gimple file in CWD)
set GIMPLE_FILE=
for /f "delims=" %%i in ('dir /b *.gimple 2^>nul') do (
    set GIMPLE_FILE=%%i
)

if "%GIMPLE_FILE%"=="" (
    echo [Error] GIMPLE file not found after GCC run.
    goto :cleanup
)

python src\tools\gimple_parser.py "%GIMPLE_FILE%" > "%TEMP_IR%"
if errorlevel 1 (
    echo [Error] GIMPLE parser failed.
    goto :cleanup
)

rem Cleanup GCC artifacts
if exist "%~n1.o" del "%~n1.o"
if exist "%GIMPLE_FILE%" del "%GIMPLE_FILE%"

rem Extracting only the Translated Output part from IR log if needed, 
rem but run_inference outputs everything to stdout.
rem For now, we assume the user wants to see the progress.
echo [Stage 1] Completed. IR generated in %TEMP_IR%

echo [Stage 2] Converting IR to Rust...
python src\tools\run_inference.py ir2rust "%TEMP_IR%" --raw > "%OUTPUT_RS%"
if errorlevel 1 (
    echo [Error] Stage 2 failed.
    goto :cleanup
)

echo [Pipeline] Success! Rust code generated in %OUTPUT_RS%

:cleanup
if exist %TEMP_IR% del %TEMP_IR%

:end
endlocal
