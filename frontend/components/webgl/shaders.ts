export const fragmentShader = `
precision highp float;

uniform vec2 resolution;
uniform vec2 center;
uniform float zoom;
uniform int maxIterations;
uniform vec3 colorA;
uniform vec3 colorB;

vec2 squareImaginary(vec2 number) {
    return vec2(
        number.x * number.x - number.y * number.y,
        2.0 * number.x * number.y
    );
}

vec3 mapColor(float t) {
    return mix(colorA, colorB, t);
}

void main() {
    vec2 uv = gl_FragCoord.xy / resolution;
    vec2 c = (uv - 0.5) * 4.0 / zoom;
    c = c + center;
    
    vec2 z = vec2(0.0);
    float iter = 0.0;
    
    // GLSL 1.0 requires loop variables to be constant or simple
    for (int i = 0; i < 1000; i++) {
        if (i >= maxIterations) break;
        z = squareImaginary(z) + c;
        if (dot(z, z) > 4.0) break;
        iter += 1.0;
    }
    
    float smooth_iter = iter - log2(log2(dot(z, z))) + 4.0;
    float t = smooth_iter / float(maxIterations);
    
    gl_FragColor = vec4(mapColor(t), 1.0);
}
`;

export const vertexShader = `
  attribute vec2 position;
  void main() {
      gl_Position = vec4(position, 0.0, 1.0);
  }
`;

