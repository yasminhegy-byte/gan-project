# PowerShell setup script for GAN project environment

Write-Host "`n" -NoNewline
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host " GAN Project - Environment Setup Script" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "`n" -NoNewline

# Get conda path
$CONDA_PATH = "$env:USERPROFILE\miniconda3\Scripts\conda.exe"

# Check if conda exists
if (-not (Test-Path $CONDA_PATH)) {
    Write-Host "ERROR: Conda not found at $CONDA_PATH" -ForegroundColor Red
    Write-Host "Please install Miniconda or Anaconda first." -ForegroundColor Red
    exit 1
}

Write-Host "[1/3] Creating conda environment: env_rl_project" -ForegroundColor Yellow
Write-Host ""

# Create environment
& $CONDA_PATH create -n env_rl_project python=3.11 numpy pandas matplotlib -y --override-channels -c conda-forge

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to create conda environment" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[2/3] Environment created successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "[3/3] To activate the environment, run:" -ForegroundColor Yellow
Write-Host "  conda activate env_rl_project" -ForegroundColor White
Write-Host ""
Write-Host "Then run the training script:" -ForegroundColor Yellow
Write-Host "  python train_gan.py" -ForegroundColor White
Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host " Setup Complete!" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""
