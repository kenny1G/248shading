uniform mat4 obj2world;                 // object to world space transform
uniform int  num_spot_lights;
#define MAX_NUM_LIGHTS 10
uniform float time;                   // global time for animation

uniform mat3 obj2worldNorm;             // object to world transform for normals
uniform vec3 camera_position;           // world space camera position
uniform mat4 mvp;                       // ModelViewProjection Matrix

uniform bool useNormalMapping;         // true if normal mapping should be used

// per vertex input attributes
in vec3 vtx_position;            // object space position
in vec3 vtx_tangent;
in vec3 vtx_normal;              // object space normal
in vec2 vtx_texcoord;
in vec3 vtx_diffuse_color;

// per vertex outputs
out vec3 position;                  // world space position
out vec3 vertex_diffuse_color;
out vec2 texcoord;
out vec3 dir2camera;                // world space vector from surface point to camera
out vec3 normal;
out mat3 tan2world;                 // tangent space rotation matrix multiplied by obj2WorldNorm
uniform mat4 obj2shadowlight[MAX_NUM_LIGHTS]; // object to light space transforms
out vec4 position_shadowlight[MAX_NUM_LIGHTS]; // position in light space for each light

// Add noise functions at the top
vec2 hash( vec2 p ) {
    p = vec2( dot(p,vec2(127.1,311.7)), dot(p,vec2(269.5,183.3)) );
    return -1.0 + 2.0*fract(sin(p)*43758.5453123);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f*f*(3.0-2.0*f);
    float a = dot(hash(i + vec2(0.0,0.0)), f - vec2(0.0,0.0));
    float b = dot(hash(i + vec2(1.0,0.0)), f - vec2(1.0,0.0));
    float c = dot(hash(i + vec2(0.0,1.0)), f - vec2(0.0,1.0));
    float d = dot(hash(i + vec2(1.0,1.0)), f - vec2(1.0,1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float fbm(vec2 p) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 3.0;
    for(int i = 0; i < 5; i++) {
        value += amplitude * noise(p * frequency);
        frequency *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}

void main(void)
{
    // Create wave motion using noise
    vec3 wave_pos = vtx_position;

    // Large scale waves
    float large_waves = fbm(wave_pos.xz * 0.3 + time * 0.1) * 0.5;

    // Medium scale waves
    float medium_waves = fbm(wave_pos.xz * 0.7 + time * 0.2) * 0.2;

    // Small ripples
    float ripples = fbm(wave_pos.xz * 2.0 + time * 0.4) * 0.1;

    wave_pos.y += large_waves + medium_waves + ripples;

    position = vec3(obj2world * vec4(wave_pos, 1));

    for (int i = 0; i < num_spot_lights; i++) {
        position_shadowlight[i] = obj2shadowlight[i] * vec4(wave_pos, 1.0);
    }

    vec3 T = normalize(obj2worldNorm * vtx_tangent);
    vec3 N = normalize(obj2worldNorm * vtx_normal);
    vec3 B = normalize(cross(N, T));
    T = normalize(cross(B, N));

    tan2world = mat3(T, B, N);

    normal = obj2worldNorm * vtx_normal;

    vertex_diffuse_color = vtx_diffuse_color;
    texcoord = vtx_texcoord;
    dir2camera = camera_position - position;
    gl_Position = mvp * vec4(wave_pos, 1);
}