from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import io
import base64
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class MandelbrotRequest(BaseModel):
    h: int = 1000
    w: int = 1500
    max_iter: int = 100
    cx: float = -0.743643887037158704752191506114774
    cy: float = 0.131825904205311970493132056385139
    zoom: float = 1

class JuliaRequest(BaseModel):
    h: int = 1000
    w: int = 1500
    max_iter: int = 100
    c_real: float = -0.4
    c_imag: float = 0.6
    zoom: float = 1

class SierpinskiRequest(BaseModel):
    n_points: int = 100000
    iterations: int = 20

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

def julia_set(h, w, max_iter, c, zoom):
    y, x = np.ogrid[-1.5/zoom:1.5/zoom:h*1j, -1.5/zoom:1.5/zoom:w*1j]
    z = x + y*1j
    divtime = max_iter + np.zeros(z.shape, dtype=int)

    for i in range(max_iter):
        z = z**2 + c
        diverge = z*np.conj(z) > 2**2
        div_now = diverge & (divtime == max_iter)
        divtime[div_now] = i
        z[diverge] = 2

    return divtime

def sierpinski_triangle(n_points, iterations):
    vertices = np.array([[0, 0], [0.5, np.sqrt(3)/2], [1, 0]])
    points = np.random.rand(n_points, 2)
    
    for i in range(iterations):
        points = vertices[np.random.randint(0, 3, n_points)] * 0.5 + points * 0.5
    
    return points

def generate_fractal_image(data, cmap_name='custom'):
    plt.figure(figsize=(12, 8))
    
    colors = ['#000000', '#08d5fc', '#ba5cca', '#da3e5c', '#403741', '#0794fa', '#FF00FF']
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=len(colors))
    
    plt.imshow(data, cmap=cmap)
    plt.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    
    return buf

def generate_sierpinski_image(points):
    plt.figure(figsize=(10, 10))
    plt.scatter(points[:, 0], points[:, 1], s=0.1, c='black')
    plt.axis('equal')
    plt.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    
    return buf

@app.post("/api/generate-mandelbrot")
async def generate_mandelbrot(request: MandelbrotRequest):
    try:
        data = mandelbrot(
            request.h, request.w, request.max_iter,
            request.cx, request.cy, request.zoom
        )
        img_buf = generate_fractal_image(data)
        img_str = base64.b64encode(img_buf.getvalue()).decode()
        return {"image": img_str}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-julia")
async def generate_julia(request: JuliaRequest):
    try:
        c = complex(request.c_real, request.c_imag)
        data = julia_set(request.h, request.w, request.max_iter, c, request.zoom)
        img_buf = generate_fractal_image(data)
        img_str = base64.b64encode(img_buf.getvalue()).decode()
        return {"image": img_str}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-sierpinski")
async def generate_sierpinski(request: SierpinskiRequest):
    try:
        points = sierpinski_triangle(request.n_points, request.iterations)
        img_buf = generate_sierpinski_image(points)
        img_str = base64.b64encode(img_buf.getvalue()).decode()
        return {"image": img_str}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))