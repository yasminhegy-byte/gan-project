
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


np.random.seed(99)
torch.manual_seed(99)
if torch.cuda.is_available():
    torch.cuda.manual_seed(99)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Generator(nn.Module):
    
    def __init__(self, latent_dim=16, img_dim=784):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, img_dim),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    
    def __init__(self, img_dim=784):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)


def main():
    print(f"Using device: {device}")
    
    print("Generating training data...")
    data = np.random.randint(0, 256, (200, 784))
    df = pd.DataFrame(data)
    df.to_csv("pixel_data.csv", index=False)
    print("pixel_data.csv created!")
    print(f"Data shape: {data.shape}")
    
    print("\nLoading and normalizing data...")
    df = pd.read_csv("pixel_data.csv")
    data = df.values.astype(np.float32)
    data = (data - data.min()) / (data.max() - data.min())
    data_tensor = torch.FloatTensor(data).to(device)
    print(f"Data loaded! Shape: {data.shape}")
    print(f"Min: {data.min():.2f}  Max: {data.max():.2f}")
    
    print("\nInitializing networks...")
    latent_dim = 16
    img_dim = 784
    
    generator = Generator(latent_dim, img_dim).to(device)
    discriminator_net = Discriminator(img_dim).to(device)
    
    print("Generator built!")
    print("Discriminator built!")
    
    lr = 0.0002
    beta1 = 0.5
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_d = optim.Adam(discriminator_net.parameters(), lr=lr, betas=(beta1, 0.999))
    criterion = nn.BCELoss()
    
    epochs = 300
    batch_size = 32
    
    d_losses = []
    g_losses = []
    
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        idx = np.random.randint(0, len(data), batch_size)
        real = data_tensor[idx]
        
        noise = torch.randn(batch_size, latent_dim).to(device)
        fake = generator(noise)
        
        optimizer_d.zero_grad()
        
        real_preds = discriminator_net(real)
        real_labels = torch.ones(batch_size, 1).to(device)
        d_loss_real = criterion(real_preds, real_labels)
        
        fake_preds = discriminator_net(fake.detach())
        fake_labels = torch.zeros(batch_size, 1).to(device)
        d_loss_fake = criterion(fake_preds, fake_labels)
        
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        optimizer_d.step()
        
        optimizer_g.zero_grad()
        
        noise2 = torch.randn(batch_size, latent_dim).to(device)
        fake2 = generator(noise2)
        fake2_preds = discriminator_net(fake2)
        g_loss = criterion(fake2_preds, real_labels)
        
        g_loss.backward()
        optimizer_g.step()
        
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
    
    print("Training complete!")
    
    print("\nEvaluating...")
    generator.eval()
    discriminator_net.eval()
    
    with torch.no_grad():
        noise = torch.randn(100, latent_dim).to(device)
        fake_final = generator(noise)
        real_sample = data_tensor[:100]
        
        real_preds = discriminator_net(real_sample)
        fake_preds = discriminator_net(fake_final)
        
        real_acc = (real_preds > 0.5).float().mean().item()
        fake_acc = (fake_preds < 0.5).float().mean().item()
        accuracy = (real_acc + fake_acc) / 2
    
    print(f"Final Discriminator Accuracy: {accuracy:.4f}")
    print("Training script by Student A (PyTorch)")
    
    print("\nGenerating loss plot...")
    plt.figure(figsize=(8, 4))
    plt.plot(d_losses, label='Discriminator Loss', color='blue')
    plt.plot(g_losses, label='Generator Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN Training Loss (PyTorch)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('gan_loss.png')
    print("Plot saved to gan_loss.png!")


if __name__ == "__main__":
    main()
