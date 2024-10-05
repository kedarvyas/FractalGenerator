import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()

def sierpinski_triangle(n_points):
    # Defining the vertices of the triangle
    vertices = np.array([[0, 0], [0.5, np.sqrt(3)/2], [1, 0]])
    
    # Start with a random point
    points = np.random.rand(n_points, 2)
    
    # Iteratively choosing a random vertex and moving halfway towards it
    for i in range(20):  # Discard the first 20 iterations
        points = vertices[np.random.randint(0, 3, n_points)] * 0.5 + points * 0.5

    return points

# Generating the Sierpinski Triangle
n_points = 100000
points = sierpinski_triangle(n_points)

# Plot the result
plt.figure(figsize=(10, 10))
plt.scatter(points[:, 0], points[:, 1], s=0.1, c='black')
plt.title('Sierpinski Triangle')
plt.axis('equal')
plt.axis('off')
plt.show()

end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")