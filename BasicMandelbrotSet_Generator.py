import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time

start_time = time.time()

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

    return divtime

def plot_mandelbrot(h, w, max_iter, cx, cy, zoom):
    plt.figure(figsize=(12, 8))
    
    # Creating a custom colormap
    colors = ['#000000', '#08d5fc', '#ba5cca', '#da3e5c', '#403741', '#0794fa', '#FF00FF']
    n_bins = len(colors)
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    
    plt.imshow(mandelbrot(h, w, max_iter, cx, cy, zoom), cmap=cmap, extent=[cx-1.5/zoom, cx+0.5/zoom, cy-1/zoom, cy+1/zoom])
    plt.title(f'Mandelbrot Set (zoom: {zoom}x)')
    plt.xlabel('Re(c)')
    plt.ylabel('Im(c)')
    plt.colorbar(label='Iteration count')
    plt.show()

# Generating a series of zoomed images
h, w = 1000, 1500
max_iter = 100
cx, cy = -0.743643887037158704752191506114774, 0.131825904205311970493132056385139
zoom_levels = [1, 10, 100, 1000]

for zoom in zoom_levels:
    plot_mandelbrot(h, w, max_iter, cx, cy, zoom)

end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")