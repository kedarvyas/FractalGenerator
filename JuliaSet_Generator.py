import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time

start_time = time.time()

def julia_set(h, w, max_iter, c):
    y, x = np.ogrid[-1.5:1.5:h*1j, -1.5:1.5:w*1j]
    z = x + y*1j
    divtime = max_iter + np.zeros(z.shape, dtype=int)

    for i in range(max_iter):
        z = z**2 + c
        diverge = z*np.conj(z) > 2**2
        div_now = diverge & (divtime == max_iter)
        divtime[div_now] = i
        z[diverge] = 2

    return divtime

h, w = 1000, 1500
max_iter = 100

# Generating different Julia sets
c_values = [-0.4 + 0.6j, -0.8 + 0.156j, -0.7269 + 0.1889j]

# Creating a custom colormap
colors = ['#000000', '#08d5fc', '#ba5cca', '#da3e5c', '#403741', '#0794fa', '#FF00FF']
n_bins = len(colors)
cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

for idx, c in enumerate(c_values):
    plt.figure(figsize=(10, 10))
    plt.imshow(julia_set(h, w, max_iter, c), cmap= cmap, extent=[-1.5, 1.5, -1.5, 1.5])
    plt.title(f'Julia Set (c = {c})')
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.colorbar(label='Iteration count')
    plt.show()

end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")