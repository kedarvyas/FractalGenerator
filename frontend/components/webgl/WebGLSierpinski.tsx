// WebGLSierpinski.tsx
'use client';

import React, { useEffect, useRef } from 'react';

interface WebGLSierpinskiProps {
  width: number;
  height: number;
  iterations: number;
  points: number;
}

const vertexShader = `
  attribute vec2 position;
  void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    gl_PointSize = 1.0;
  }
`;

const fragmentShader = `
  precision highp float;
  void main() {
    gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
  }
`;

const WebGLSierpinski: React.FC<WebGLSierpinskiProps> = ({
  width,
  height,
  iterations,
  points: numPoints
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const programInfoRef = useRef<any>(null);
  const vertexBufferRef = useRef<WebGLBuffer | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext('webgl');
    if (!gl) {
      console.error('WebGL not supported');
      return;
    }

    // Create shader program
    const program = createShaderProgram(gl, vertexShader, fragmentShader);
    if (!program) return;

    // Generate initial points for the Sierpinski triangle
    const vertices = generateSierpinskiPoints(numPoints, iterations);
    
    const vertexBuffer = gl.createBuffer();
    vertexBufferRef.current = vertexBuffer;
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

    // Set up program info
    const programInfo = {
      program,
      attribLocations: {
        position: gl.getAttribLocation(program, 'position'),
      },
    };

    programInfoRef.current = programInfo;
    render();
  }, [iterations, numPoints]);

  const generateSierpinskiPoints = (points: number, iterations: number) => {
    // Define the three vertices of the triangle
    const vertices = [
      -0.8, -0.8,  // Bottom left
      0.8, -0.8,   // Bottom right
      0.0, 0.8     // Top
    ];

    // Generate points using the chaos game method
    const result = new Float32Array(points * 2);
    let x = 0;
    let y = 0;

    for (let i = 0; i < points * 2; i += 2) {
      // Choose a random vertex
      const vertex = Math.floor(Math.random() * 3) * 2;
      
      // Move halfway to the chosen vertex
      x = (x + vertices[vertex]) / 2;
      y = (y + vertices[vertex + 1]) / 2;
      
      result[i] = x;
      result[i + 1] = y;
    }

    return result;
  };

  const createShaderProgram = (gl: WebGLRenderingContext, vsSource: string, fsSource: string) => {
    const vertexShader = createShader(gl, gl.VERTEX_SHADER, vsSource);
    const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fsSource);
    if (!vertexShader || !fragmentShader) return null;

    const program = gl.createProgram();
    if (!program) return null;

    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error('Unable to initialize shader program:', gl.getProgramInfoLog(program));
      return null;
    }

    return program;
  };

  const createShader = (gl: WebGLRenderingContext, type: number, source: string) => {
    const shader = gl.createShader(type);
    if (!shader) return null;

    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.error('Shader compilation error:', gl.getShaderInfoLog(shader));
      gl.deleteShader(shader);
      return null;
    }

    return shader;
  };

  const render = () => {
    const canvas = canvasRef.current;
    const programInfo = programInfoRef.current;
    const vertexBuffer = vertexBufferRef.current;
    if (!canvas || !programInfo || !vertexBuffer) return;

    const gl = canvas.getContext('webgl');
    if (!gl) return;

    // Update canvas size to match display size
    const displayWidth = canvas.clientWidth;
    const displayHeight = canvas.clientHeight;
    if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
      canvas.width = displayWidth;
      canvas.height = displayHeight;
    }

    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(1.0, 1.0, 1.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(programInfo.program);

    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.enableVertexAttribArray(programInfo.attribLocations.position);
    gl.vertexAttribPointer(
      programInfo.attribLocations.position,
      2,
      gl.FLOAT,
      false,
      0,
      0
    );

    gl.drawArrays(gl.POINTS, 0, numPoints);
  };

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={{ width: '100%', height: '400px' }}
      className="rounded-lg"
    />
  );
};

export default WebGLSierpinski;