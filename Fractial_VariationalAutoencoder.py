import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

def create_diverse_fractal_dataset(n_samples, size):
    def mandelbrot(h, w, max_iter, cx, cy, zoom):
        y, x = np.ogrid[cy-1/zoom:cy+1/zoom:h*1j, cx-1.5/zoom:cx+0.5/zoom:w*1j]
        c = x + y*1j
        z = c
        divtime = max_iter + np.zeros(z.shape, dtype=int)
        for i in range(max_iter):
            z = z**2 + c
            diverge = z*np.conj(z) > 2**2
            div_now = diverge & (divtime == max_iter)
            divtime[div_now] = i
            z[diverge] = 2
        return divtime / max_iter

    dataset = []
    for _ in range(n_samples):
        cx = np.random.uniform(-2, 0.5)
        cy = np.random.uniform(-1, 1)
        zoom = np.random.uniform(0.5, 5)
        dataset.append(mandelbrot(size, size, 100, cx, cy, zoom))
    return np.array(dataset)

class VAE(nn.Module):
    def __init__(self, img_size, latent_dim):
        super(VAE, self).__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(img_size * img_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_var = nn.Linear(128, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, img_size * img_size),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, img_size, img_size))
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

# Generate dataset
img_size = 64
n_samples = 5000
X_train = create_diverse_fractal_dataset(n_samples, img_size)
X_train = torch.FloatTensor(X_train).unsqueeze(1)  # Add channel dimension

# Create DataLoader
dataset = TensorDataset(X_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model, loss, and optimizer
latent_dim = 64
model = VAE(img_size, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Loss function
def loss_function(recon_x, x, mu, log_var, kl_weight):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + kl_weight * KLD

# Train the model
n_epochs = 200
kl_weight = 0.1
for epoch in range(n_epochs):
    total_loss = 0
    for data in dataloader:
        img = data[0]
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(img)
        loss = loss_function(recon_batch, img, mu, log_var, kl_weight)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {total_loss/len(dataloader.dataset):.4f}')

# Generate new fractals
n_gen = 5
temperature = 1.0
z = torch.randn(n_gen, model.latent_dim) * temperature
model.eval()
start_time = time.time()
with torch.no_grad():
    generated_fractals = model.decode(z)
end_time = time.time()
print(f"Generation time: {end_time - start_time} seconds")

# Plot results
fig, axes = plt.subplots(1, n_gen, figsize=(15, 3))
for i, ax in enumerate(axes):
    ax.imshow(generated_fractals[i].squeeze().numpy(), cmap='hot')
    ax.axis('off')
plt.suptitle('Generated Fractal-like Patterns')
plt.show()