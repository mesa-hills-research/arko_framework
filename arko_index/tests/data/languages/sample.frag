#version 450 core

// Phong shading with one light.
precision highp float;

in vec3 vNormal;
in vec3 vPosition;
out vec4 fragColor;

uniform vec3 lightPosition;
uniform vec3 baseColor;

struct Material {
    float shininess;
    vec3 specular;
};

uniform Material material;

// Diffuse term.
float diffuse(vec3 n, vec3 l) {
    return max(dot(n, l), 0.0);
}

float specular(vec3 n, vec3 l, vec3 v, float shininess) {
    vec3 r = reflect(-l, n);
    return pow(max(dot(r, v), 0.0), shininess);
}

void main() {
    vec3 n = normalize(vNormal);
    vec3 l = normalize(lightPosition - vPosition);
    vec3 v = normalize(-vPosition);
    vec3 color = baseColor * diffuse(n, l)
        + material.specular * specular(n, l, v, material.shininess);
    fragColor = vec4(color, 1.0);
}
