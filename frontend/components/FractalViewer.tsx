'use client';

import React, { useState, useCallback, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { toast } from "@/components/ui/use-toast";
import { Loader2 } from "lucide-react";
import WebGLMandelbrot from './webgl/WebGLMandelbrot';
import WebGLSierpinski from './webgl/WebGLSierpinski';

const FractalViewer = () => {
  const [selectedFractal, setSelectedFractal] = useState('mandelbrot');
  const [zoom, setZoom] = useState([2]);
  const [iterations, setIterations] = useState([100]);
  const [loading, setLoading] = useState(false);
  const [fractalImage, setFractalImage] = useState<string | null>(null);
  const [centerX, setCenterX] = useState(-0.7436);
  const [centerY, setCenterY] = useState(0.1318);
  const [juliaReal, setJuliaReal] = useState([-0.4]);
  const [juliaImag, setJuliaImag] = useState([0.6]);
  const [points, setPoints] = useState([100000]);

  // Debounce timer
  const [debounceTimer, setDebounceTimer] = useState<NodeJS.Timeout | null>(null);

  const fractalTypes = [
    { id: 'mandelbrot', name: 'Mandelbrot Set' },
    { id: 'julia', name: 'Julia Set' },
    { id: 'sierpinski', name: 'Sierpinski Triangle' }
  ];

  const debouncedGenerate = useCallback(() => {
    if (debounceTimer) {
      clearTimeout(debounceTimer);
    }
    const timer = setTimeout(() => {
      generateFractal();
    }, 250);
    setDebounceTimer(timer);
  }, [debounceTimer]);

  // Effect to trigger generation when controls change
  useEffect(() => {
    debouncedGenerate();
    return () => {
      if (debounceTimer) {
        clearTimeout(debounceTimer);
      }
    };
  }, [zoom, iterations, juliaReal, juliaImag, points, selectedFractal]);

  const handleImageClick = useCallback((e: React.MouseEvent<HTMLImageElement>) => {
    if (selectedFractal === 'sierpinski') return;

    const rect = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;

    if (selectedFractal === 'mandelbrot') {
      const newX = centerX + (x - 0.5) / zoom[0];
      const newY = centerY + (y - 0.5) / zoom[0];
      setCenterX(newX);
      setCenterY(newY);
    }

    setZoom([zoom[0] * 1.5]);
  }, [selectedFractal, centerX, centerY, zoom]);

  const handleMandelbrotClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (selectedFractal !== 'mandelbrot') return;

    const rect = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;

    // Calculate new center based on current zoom level
    const newX = centerX + (x - 0.5) * 4.0 / zoom[0];
    const newY = centerY + (y - 0.5) * 4.0 / zoom[0];

    setCenterX(newX);
    setCenterY(newY);
    setZoom([zoom[0] * 1.5]); // Zoom in by 1.5x
  }, [selectedFractal, centerX, centerY, zoom]);

  const generateFractal = async () => {
    if (selectedFractal === 'mandelbrot') return;

    if (loading) return;
    setLoading(true);
    try {
      let endpoint = '';
      let body = {};

      switch (selectedFractal) {
        case 'julia':
          endpoint = '/api/generate-julia';
          body = {
            max_iter: iterations[0],
            zoom: zoom[0],
            c_real: juliaReal[0],
            c_imag: juliaImag[0],
            h: 800,
            w: 1200,
          };
          break;
        case 'sierpinski':
          endpoint = '/api/generate-sierpinski';
          body = {
            n_points: points[0],
            iterations: iterations[0],
          };
          break;
      }

      const response = await fetch(`http://localhost:8000${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        throw new Error('Failed to generate fractal');
      }

      const data = await response.json();
      setFractalImage(`data:image/png;base64,${data.image}`);
    } catch (error) {
      console.error('Error generating fractal:', error);
      toast({
        title: "Error",
        description: "Failed to generate fractal. Please try again.",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setZoom([1]);
    setIterations([100]);
    setPoints([100000]);
    setCenterX(-0.7436);
    setCenterY(0.1318);
    setJuliaReal([-0.4]);
    setJuliaImag([0.6]);
    setFractalImage(null);
  };

  return (
    <div className="p-4 max-w-3xl mx-auto">
      <Card className="bg-white shadow-sm">
        <CardHeader className="pb-4">
          <CardTitle className="text-lg font-medium">Fractal Generator</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* Fractal Type Selector */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-700">Select Fractal Type</label>
              <Select value={selectedFractal} onValueChange={setSelectedFractal}>
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Select a fractal type" />
                </SelectTrigger>
                <SelectContent>
                  {fractalTypes.map(fractal => (
                    <SelectItem key={fractal.id} value={fractal.id}>
                      {fractal.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Controls Section */}
            <div className="grid grid-cols-1 gap-4">
              {/* Show zoom control for Mandelbrot and Julia */}
              {selectedFractal !== 'sierpinski' && (
                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-700">Zoom Level: {zoom[0].toFixed(2)}x</label>
                  <Slider
                    value={zoom}
                    onValueChange={setZoom}
                    min={1}
                    max={50}
                    step={0.1}
                    className="w-full"
                  />
                </div>
              )}

              {/* Julia Set specific controls */}
              {selectedFractal === 'julia' && (
                <>
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-gray-700">Real Component: {juliaReal[0].toFixed(2)}</label>
                    <Slider
                      value={juliaReal}
                      onValueChange={setJuliaReal}
                      min={-2}
                      max={2}
                      step={0.01}
                      className="w-full"
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-gray-700">Imaginary Component: {juliaImag[0].toFixed(2)}</label>
                    <Slider
                      value={juliaImag}
                      onValueChange={setJuliaImag}
                      min={-2}
                      max={2}
                      step={0.01}
                      className="w-full"
                    />
                  </div>
                </>
              )}

              {/* Sierpinski specific controls */}
              {selectedFractal === 'sierpinski' && (
                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-700">Points: {points[0].toLocaleString()}</label>
                  <Slider
                    value={points}
                    onValueChange={setPoints}
                    min={1000}
                    max={500000}
                    step={1000}
                    className="w-full"
                  />
                </div>
              )}

              {/* Iterations control for all fractals */}
              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-700">Iterations: {iterations[0]}</label>
                <Slider
                  value={iterations}
                  onValueChange={setIterations}
                  min={10}
                  max={1000}
                  step={10}
                  className="w-full"
                />
              </div>
            </div>

            {/* Display Section */}
            {selectedFractal === 'mandelbrot' ? (
              <div
                className="w-full h-[400px] rounded-lg bg-gray-50 border border-gray-100 overflow-hidden"
                onClick={handleMandelbrotClick}
              >
                <WebGLMandelbrot
                  width={800}
                  height={600}
                  zoom={zoom[0]}
                  centerX={centerX}
                  centerY={centerY}
                  maxIterations={iterations[0]}
                />
              </div>
            ) : selectedFractal === 'sierpinski' ? (
              <div className="w-full h-[400px] rounded-lg bg-gray-50 border border-gray-100 overflow-hidden">
                <WebGLSierpinski
                  width={800}
                  height={600}
                  points={points[0]}
                  iterations={iterations[0]}
                />
              </div>
            ) : (
              <div className="w-full rounded-lg bg-gray-50 border border-gray-100 overflow-hidden">
                {loading ? (
                  <div className="h-[400px] flex flex-col items-center justify-center gap-2">
                    <Loader2 className="h-8 w-8 animate-spin text-gray-500" />
                    <p className="text-sm text-gray-500">Generating fractal...</p>
                  </div>
                ) : fractalImage ? (
                  <div className="relative w-full h-[400px] flex items-center justify-center">
                    <img
                      src={fractalImage}
                      alt="Generated Fractal"
                      className="max-w-full max-h-full object-contain cursor-crosshair"
                      onClick={handleImageClick}
                    />
                    <div className="absolute bottom-2 left-2 text-xs text-gray-500 bg-white/80 px-2 py-1 rounded">
                      Zoom: {zoom[0].toFixed(2)}x | Iterations: {iterations[0]}
                    </div>
                  </div>
                ) : (
                  <div className="h-[400px] flex items-center justify-center">
                    <p className="text-sm text-gray-500">Click Generate to create fractal</p>
                  </div>
                )}
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex space-x-2 pt-2">
              <Button
                onClick={generateFractal}
                disabled={loading}
                className="bg-black hover:bg-gray-800"
              >
                {loading ? 'Generating...' : 'Generate Fractal'}
              </Button>
              <Button
                variant="outline"
                onClick={handleReset}
                className="border-gray-200 hover:bg-gray-50"
              >
                Reset
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default FractalViewer;