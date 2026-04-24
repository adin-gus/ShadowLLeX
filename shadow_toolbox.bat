@echo off
title Shadow LLeX Toolbox
color 0A

:: ============================================================
:: CONFIGURATION — set your library path here or change at [L]
:: ============================================================
set LIBRARY_DIR=C:/shadow-library

:MENU
color 0A
cls
echo ============================================================
echo    SHADOW LLeX TOOLBOX
echo ============================================================
echo.
echo   Library: %LIBRARY_DIR%
echo.
echo   [1] Extract a model to library
echo   [2] Inspect library
echo   [3] List all extracted models
echo   [4] Reassemble model from blueprint
echo   [L] Change library path
echo.
echo   [0] Exit
echo.
echo ============================================================
set /p choice="Select [0-4/L]: "

if /i "%choice%"=="1" goto EXTRACT
if /i "%choice%"=="2" goto INSPECT
if /i "%choice%"=="3" goto LIST_MODELS
if /i "%choice%"=="4" goto REASSEMBLE
if /i "%choice%"=="L" goto SET_LIBRARY
if    "%choice%"=="0" goto EXIT
goto MENU

:SET_LIBRARY
cls
echo ============================================================
echo    SET LIBRARY PATH
echo ============================================================
echo.
echo   Current: %LIBRARY_DIR%
echo.
set /p NEW_LIB="  New library path (Enter to keep current): "
if not "%NEW_LIB%"=="" set "LIBRARY_DIR=%NEW_LIB%"
goto MENU

:EXTRACT
cls
echo ============================================================
echo    EXTRACT MODEL TO SHADOW LIBRARY
echo ============================================================
echo.
echo   Library: %LIBRARY_DIR%
echo.
set /p gguf_path="Path to .gguf file: "
if not exist "%gguf_path%" (
    echo [ERROR] File not found: %gguf_path%
    pause
    goto MENU
)
set /p model_name="Model name (Enter for auto from filename): "
if "%model_name%"=="" (
    python alloy_shadow_extract.py --model "%gguf_path%" --out-dir "%LIBRARY_DIR%"
) else (
    python alloy_shadow_extract.py --model "%gguf_path%" --out-dir "%LIBRARY_DIR%" --name "%model_name%"
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
python shadow_inspector.py --library "%LIBRARY_DIR%"
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
    echo Library not found: %LIBRARY_DIR%
    echo Use [L] to set a different path.
    pause
    goto MENU
)
echo Model folders in %LIBRARY_DIR%:
echo.
dir "%LIBRARY_DIR%" /AD /B
echo.
pause
goto MENU

:REASSEMBLE
cls
echo ============================================================
echo    REASSEMBLE MODEL FROM BLUEPRINT
echo ============================================================
echo.
echo Rebuilds a model from its shadow library blueprint.
echo No original GGUF required.
echo.
set /p model_name="Model name to reassemble: "
if "%model_name%"=="" (
    echo [ERROR] Model name required
    pause
    goto MENU
)
set /p out_path="Output path (Enter for default): "
if "%out_path%"=="" set "out_path=%LIBRARY_DIR%\%model_name%_rebuilt.gguf"
python alloy_shadow_compose.py --library "%LIBRARY_DIR%" --blueprint "%model_name%" --out "%out_path%"
echo.
echo Reassembly complete!
pause
goto MENU

:EXIT
echo.
echo Goodbye!
exit /b 0
