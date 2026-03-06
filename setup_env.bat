@echo off
REM Setup script for GAN project environment

echo.
echo ===============================================
echo  GAN Project - Environment Setup Script
echo ===============================================
echo.

REM Get conda path
set CONDA_PATH=%USERPROFILE%\miniconda3\Scripts\conda.exe

REM Check if conda exists
if not exist "%CONDA_PATH%" (
    echo ERROR: Conda not found at %CONDA_PATH%
    echo Please install Miniconda or Anaconda first.
    pause
    exit /b 1
)

echo [1/3] Creating conda environment: env_rl_project
echo.
call "%CONDA_PATH%" create -n env_rl_project python=3.11 numpy pandas matplotlib -y --override-channels -c conda-forge

if %errorlevel% neq 0 (
    echo ERROR: Failed to create conda environment
    pause
    exit /b 1
)

echo.
echo [2/3] Environment created successfully!
echo.
echo [3/3] To activate the environment, run:
echo   conda activate env_rl_project
echo.
echo Then run the training script:
echo   python train_gan.py
echo.
echo ===============================================
echo  Setup Complete!
echo ===============================================
echo.

pause
