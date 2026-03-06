# GAN Training Project

A reproducible Generative Adversarial Network (GAN) training script with conda environment and Docker support.

## Project Structure

```
gan_project/
├── train_gan.py           # Main training script
├── requirements.txt       # Python dependencies for Docker
├── environment.yml        # Conda environment specification
├── Dockerfile            # Docker container configuration
└── README.md             # This file
```

## Part 1: Conda Environment Setup

### Prerequisites
- Conda installed on your system

### Create the Environment

```bash
# Create conda environment from file
conda env create -f environment.yml

# OR create manually
conda create -n env_rl_project python=3.11 numpy pandas matplotlib -y
```

### Activate the Environment

```bash
conda activate env_rl_project
```

### Export Environment (if needed)

```bash
conda env export > environment.yml
```

### Run Training

```bash
cd gan_project
python train_gan.py
```

## Part 2: Docker Setup

### Prerequisites
- Docker installed on your system

### Build the Docker Image

```bash
docker build -t gan-trainer:latest .
```

### Run the Container

```bash
docker run --rm \
  -v $(pwd)/output:/app/output \
  gan-trainer:latest
```

On Windows (PowerShell):
```powershell
docker run --rm `
  -v ${PWD}/output:/app/output `
  gan-trainer:latest
```

### Docker Build Strategy

The Dockerfile follows efficient layering best practices:

1. **Lightweight base image**: `python:3.11-slim` (≈150MB)
2. **Copy requirements first**: Enables Docker layer caching
3. **Install dependencies**: Only re-runs if requirements.txt changes
4. **Copy application code**: Final layer, changes frequently
5. **Set working directory**: Clean container environment

## Outputs

The training script generates:
- `pixel_data.csv` - Generated training data
- `gan_loss.png` - Training loss plot

## Environment Details

### Dependencies
- `numpy==1.26.4` - Numerical computing
- `pandas==3.0.1` - Data manipulation
- `matplotlib==3.10.8` - Visualization

### Python Version
- Python 3.11

## Training Parameters

- **Epochs**: 300
- **Batch Size**: 32
- **Learning Rate**: 0.0002
- **Latent Dimension**: 16
- **Image Dimension**: 784 (28x28 pixels)

## Student
Created by: Student A

## Notes

- The script is fully reproducible with fixed random seeds
- Training takes approximately 2-5 minutes on a standard CPU
- The GAN trains on synthetic pixel data for demonstration
