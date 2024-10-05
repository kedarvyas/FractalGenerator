import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

def create_fractal_dataset(n_samples, size):
    def mandelbrot(h, w, max_iter):
        y, x = np.ogrid[-1.4:1.4:h*1j, -2:0.8:w*1j]
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

    return np.array([mandelbrot(size, size, 100) for _ in range(n_samples)])

class Autoencoder(nn.Module):
    def __init__(self, img_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(img_size * img_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, img_size * img_size),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, img_size, img_size))
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Generate dataset
img_size = 64
n_samples = 2000
X_train = create_fractal_dataset(n_samples, img_size)
X_train = torch.FloatTensor(X_train).unsqueeze(1)  # Add channel dimension

# Create DataLoader
dataset = TensorDataset(X_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model, loss, and optimizer
model = Autoencoder(img_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
n_epochs = 100
for epoch in range(n_epochs):
    for data in dataloader:
        img = data[0]
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, img)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

# Generate new fractals
n_gen = 5
noise = torch.randn(n_gen, 32)
model.eval()
start_time = time.time()
with torch.no_grad():
    generated_fractals = model.decoder(noise)
end_time = time.time()
print(f"Generation time: {end_time - start_time} seconds")

# Plot results
fig, axes = plt.subplots(1, n_gen, figsize=(15, 3))
for i, ax in enumerate(axes):
    ax.imshow(generated_fractals[i].squeeze().numpy(), cmap='hot')
    ax.axis('off')
plt.suptitle('Generated Fractal-like Patterns')
plt.show()