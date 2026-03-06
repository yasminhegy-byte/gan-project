# Use lightweight Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for efficient layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training script
COPY train_gan.py .

# Set the default command to run the training script
CMD ["python", "train_gan.py"]
