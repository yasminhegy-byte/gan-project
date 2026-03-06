#!/usr/bin/env python3
"""
GAN Training Script
Trains a Generative Adversarial Network on pixel data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def relu(x):
    """Rectified Linear Unit activation function."""
    return np.maximum(0, x)


def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def leaky_relu(x, alpha=0.2):
    """Leaky ReLU activation function."""
    return np.where(x > 0, x, alpha * x)


def bce_loss(pred, target):
    """Binary Cross Entropy loss."""
    pred = np.clip(pred, 1e-7, 1 - 1e-7)
    return -np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred))


def update_weights(weights, grads, lr=0.0002):
    """Update weights using gradient descent."""
    return [w - lr * g for w, g in zip(weights, grads)]


def generator_forward(z, G_w1, G_b1, G_w2, G_b2, G_w3, G_b3):
    """Forward pass through generator."""
    h1 = relu(z @ G_w1 + G_b1)
    h2 = relu(h1 @ G_w2 + G_b2)
    out = sigmoid(h2 @ G_w3 + G_b3)
    return out, h1, h2


def discriminator_forward(x, D_w1, D_b1, D_w2, D_b2, D_w3, D_b3):
    """Forward pass through discriminator."""
    h1 = leaky_relu(x @ D_w1 + D_b1)
    h2 = leaky_relu(h1 @ D_w2 + D_b2)
    out = sigmoid(h2 @ D_w3 + D_b3)
    return out, h1, h2


def generator(z, G_w1, G_b1, G_w2, G_b2, G_w3, G_b3):
    """Generate fake samples."""
    h1 = relu(z @ G_w1 + G_b1)
    h2 = relu(h1 @ G_w2 + G_b2)
    out = sigmoid(h2 @ G_w3 + G_b3)
    return out


def discriminator(x, D_w1, D_b1, D_w2, D_b2, D_w3, D_b3):
    """Discriminate real vs fake samples."""
    h1 = leaky_relu(x @ D_w1 + D_b1)
    h2 = leaky_relu(h1 @ D_w2 + D_b2)
    out = sigmoid(h2 @ D_w3 + D_b3)
    return out


def main():
    """Main training loop."""
    # Set random seed for reproducibility
    np.random.seed(99)
    
    # Generate and load data
    print("Generating training data...")
    data = np.random.randint(0, 256, (200, 784))
    df = pd.DataFrame(data)
    df.to_csv("pixel_data.csv", index=False)
    print("pixel_data.csv created!")
    print(f"Data shape: {data.shape}")
    
    # Load and normalize data
    print("\nLoading and normalizing data...")
    df = pd.read_csv("pixel_data.csv")
    data = df.values.astype(np.float32)
    data = (data - data.min()) / (data.max() - data.min())
    print(f"Data loaded! Shape: {data.shape}")
    print(f"Min: {data.min():.2f}  Max: {data.max():.2f}")
    
    # Initialize networks
    print("\nInitializing networks...")
    latent_dim = 16
    img_dim = 784
    
    # Generator weights
    np.random.seed(None)
    G_w1 = np.random.randn(latent_dim, 64) * 0.01
    G_b1 = np.zeros(64)
    G_w2 = np.random.randn(64, 128) * 0.01
    G_b2 = np.zeros(128)
    G_w3 = np.random.randn(128, img_dim) * 0.01
    G_b3 = np.zeros(img_dim)
    print("Generator built!")
    
    # Discriminator weights
    D_w1 = np.random.randn(img_dim, 128) * 0.01
    D_b1 = np.zeros(128)
    D_w2 = np.random.randn(128, 64) * 0.01
    D_b2 = np.zeros(64)
    D_w3 = np.random.randn(64, 1) * 0.01
    D_b3 = np.zeros(1)
    print("Discriminator built!")
    
    # Training parameters
    epochs = 300
    batch_size = 32
    lr = 0.0002
    
    d_losses = []
    g_losses = []
    
    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        # Get real batch
        idx = np.random.randint(0, len(data), batch_size)
        real = data[idx]
        
        # Generate fake batch
        noise = np.random.randn(batch_size, latent_dim)
        fake = generator(noise, G_w1, G_b1, G_w2, G_b2, G_w3, G_b3)
        
        # Discriminator on real and fake
        real_preds, _, _ = discriminator_forward(real, D_w1, D_b1, D_w2, D_b2, D_w3, D_b3)
        fake_preds, _, _ = discriminator_forward(fake, D_w1, D_b1, D_w2, D_b2, D_w3, D_b3)
        
        # Discriminator loss
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))
        d_loss = (bce_loss(real_preds, real_labels) + 
                  bce_loss(fake_preds, fake_labels)) / 2
        
        # Generator loss
        noise2 = np.random.randn(batch_size, latent_dim)
        fake2 = generator(noise2, G_w1, G_b1, G_w2, G_b2, G_w3, G_b3)
        fake2_preds, _, _ = discriminator_forward(fake2, D_w1, D_b1, D_w2, D_b2, D_w3, D_b3)
        g_loss = bce_loss(fake2_preds, real_labels)
        
        d_losses.append(d_loss)
        g_losses.append(g_loss)
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | D Loss: {d_loss:.4f} | G Loss: {g_loss:.4f}")
    
    print("Training complete!")
    
    # Evaluation
    print("\nEvaluating...")
    noise = np.random.randn(100, latent_dim)
    fake_final = generator(noise, G_w1, G_b1, G_w2, G_b2, G_w3, G_b3)
    real_sample = data[:100]
    
    real_preds, _, _ = discriminator_forward(real_sample, D_w1, D_b1, D_w2, D_b2, D_w3, D_b3)
    fake_preds, _, _ = discriminator_forward(fake_final, D_w1, D_b1, D_w2, D_b2, D_w3, D_b3)
    
    real_acc = np.mean(real_preds > 0.5)
    fake_acc = np.mean(fake_preds < 0.5)
    accuracy = (real_acc + fake_acc) / 2
    
    print(f"Final Discriminator Accuracy: {accuracy:.4f}")
    print("Training script by Student A")
    
    # Plot losses
    print("\nGenerating loss plot...")
    plt.figure(figsize=(8, 4))
    plt.plot(d_losses, label='Discriminator Loss', color='blue')
    plt.plot(g_losses, label='Generator Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN Training Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('gan_loss.png')
    print("Plot saved to gan_loss.png!")


if __name__ == "__main__":
    main()
