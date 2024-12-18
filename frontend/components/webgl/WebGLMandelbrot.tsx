'use client';

import React, { useEffect, useRef } from 'react';
import { fragmentShader, vertexShader } from './shaders';

interface WebGLMandelbrotProps {
  width: number;
  height: number;
  zoom: number;
  centerX: number;
  centerY: number;
  maxIterations: number;
}

const WebGLMandelbrot: React.FC<WebGLMandelbrotProps> = ({
  width,
  height,
  zoom,
  centerX,
  centerY,
  maxIterations
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const programInfoRef = useRef<any>(null);
  const vertexBufferRef = useRef<WebGLBuffer | null>(null); // Add this line

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext('webgl');
    if (!gl) {
      console.error('WebGL not supported');
      return;
    }

    console.log('WebGL initialized successfully');

    // Create shader program
    const program = createShaderProgram(gl, vertexShader, fragmentShader);
    if (!program) return;

    // Set up buffer with vertices for a full-screen quad
    const vertices = new Float32Array([
      -1, -1,
       1, -1,
      -1,  1,
       1,  1
    ]);

    const vertexBuffer = gl.createBuffer();
    vertexBufferRef.current = vertexBuffer; // Store the buffer reference
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);


    // Get attribute and uniform locations
    const programInfo = {
      program,
      attribLocations: {
        position: gl.getAttribLocation(program, 'position'),
      },
      uniformLocations: {
        resolution: gl.getUniformLocation(program, 'resolution'),
        center: gl.getUniformLocation(program, 'center'),
        zoom: gl.getUniformLocation(program, 'zoom'),
        maxIterations: gl.getUniformLocation(program, 'maxIterations'),
        colorA: gl.getUniformLocation(program, 'colorA'),
        colorB: gl.getUniformLocation(program, 'colorB'),
      },
    };

    programInfoRef.current = programInfo;

    // Initial render
    render();
  }, []);

  useEffect(() => {
    render();
  }, [zoom, centerX, centerY, maxIterations, width, height]);

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
    if (!canvas || !programInfo) {
      console.error('Missing canvas or program info');
      return;
    }
  
    const gl = canvas.getContext('webgl');
    if (!gl) {
      console.error('Could not get WebGL context');
      return;
    }
  
    // Update canvas size to match display size
    const displayWidth = canvas.clientWidth;
    const displayHeight = canvas.clientHeight;
    if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
      canvas.width = displayWidth;
      canvas.height = displayHeight;
    }
  
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);
  
    gl.useProgram(programInfo.program);
  
    // Set uniforms
    gl.uniform2f(programInfo.uniformLocations.resolution, canvas.width, canvas.height);
    gl.uniform2f(programInfo.uniformLocations.center, centerX, centerY);
    gl.uniform1f(programInfo.uniformLocations.zoom, zoom);
    gl.uniform1i(programInfo.uniformLocations.maxIterations, maxIterations);
    gl.uniform3f(programInfo.uniformLocations.colorA, 0.0, 0.0, 0.0);
    gl.uniform3f(programInfo.uniformLocations.colorB, 0.7, 0.0, 1.0);
  
    // Bind the position buffer
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
  
    // Draw
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    // Calculate new center based on mouse position
    // Update parent component
  };

  return (
<canvas
  ref={canvasRef}
  width={width}
  height={height}
  style={{ width: '100%', height: '400px' }} 
  className="rounded-lg"
  onMouseMove={handleMouseMove}
/>
  );
};

export default WebGLMandelbrot;