@echo off
title Shadow LLM Toolbox
color 0A

set LIBRARY_DIR=D:/shadow-library

:MENU
color 0A
cls
echo ============================================================
echo    SHADOW LLM TOOLBOX
echo ============================================================
echo.
echo   [1] Extract a model to library
echo   [2] Inspect library
echo   [3] List all extracted models
echo   [4] Reassemble model from blueprint
echo.
echo   [0] Exit
echo.
echo ============================================================
set /p choice="Select [0-4]: "

if "%choice%"=="1" goto EXTRACT
if "%choice%"=="2" goto INSPECT
if "%choice%"=="3" goto LIST_MODELS
if "%choice%"=="4" goto REASSEMBLE
if "%choice%"=="0" goto EXIT
goto MENU

:EXTRACT
cls
echo ============================================================
echo    EXTRACT MODEL TO SHADOW LIBRARY
echo ============================================================
echo.
set /p gguf_path="Path to .gguf file: "
if not exist "%gguf_path%" (
    echo [ERROR] File not found: %gguf_path%
    pause
    goto MENU
)

set /p model_name="Model name (Enter for auto from filename): "

if "%model_name%"=="" (
    python alloy_shadow_extract.py --model "%gguf_path%" --out-dir %LIBRARY_DIR%
) else (
    python alloy_shadow_extract.py --model "%gguf_path%" --out-dir %LIBRARY_DIR% --name "%model_name%"
)

echo.
echo Extraction complete!
pause
goto MENU

:INSPECT
cls
echo ============================================================
echo    INSPECT SHADOW LIBRARY
echo ============================================================
echo.
python shadow_inspector.py --library %LIBRARY_DIR%
echo.
pause
goto MENU

:LIST_MODELS
cls
echo ============================================================
echo    MODELS IN SHADOW LIBRARY
echo ============================================================
echo.
if not exist "%LIBRARY_DIR%" (
    echo Library directory not found: %LIBRARY_DIR%
    echo Run extraction first.
    pause
    goto MENU
)

echo.
echo Model folders in %LIBRARY_DIR%:
echo.
dir "%LIBRARY_DIR%" /AD /B
echo.
echo To inspect a specific model:
echo   python shadow_inspector.py --library %LIBRARY_DIR% --model MODELNAME
echo.
pause
goto MENU

:REASSEMBLE
cls
echo ============================================================
echo    REASSEMBLE MODEL FROM BLUEPRINT
echo ============================================================
echo.
echo Rebuilds a model exactly from its shadow library blueprint.
echo No original GGUF required.
echo.
set /p model_name="Model name to reassemble: "
if "%model_name%"=="" (
    echo [ERROR] Model name required
    pause
    goto MENU
)

set /p out_name="Output filename (without .gguf): "
if "%out_name%"=="" set out_name=%model_name%_rebuilt

python alloy_shadow_compose.py --library %LIBRARY_DIR% --blueprint "%model_name%" --out "%out_name%.gguf"

echo.
echo Reassembly complete!
pause
goto MENU

:EXIT
echo.
echo Goodbye!
exit /b 0
