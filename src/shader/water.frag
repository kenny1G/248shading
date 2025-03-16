//
// Parameters that control fragment shader behavior. Different materials
// will set these flags to true/false for different looks
//

uniform bool useTextureMapping;     // true if basic texture mapping (diffuse) should be used
uniform bool useNormalMapping;      // true if normal mapping should be used
uniform bool useEnvironmentMapping; // true if environment mapping should be used
uniform bool useMirrorBRDF;         // true if mirror brdf should be used (default: phong)

uniform float time;                // global time for animation

//
// texture maps
//

uniform sampler2D diffuseTextureSampler;
uniform sampler2D normalTextureSampler;
uniform sampler2D environmentTextureSampler;

//
// lighting environment definition
//

#define MAX_NUM_LIGHTS 10
uniform int num_directional_lights;
uniform vec3 directional_light_vectors[MAX_NUM_LIGHTS];
uniform int num_point_lights;
uniform vec3 point_light_positions[MAX_NUM_LIGHTS];

uniform int   num_spot_lights;
uniform vec3  spot_light_positions[MAX_NUM_LIGHTS];
uniform vec3  spot_light_directions[MAX_NUM_LIGHTS];
uniform vec3  spot_light_intensities[MAX_NUM_LIGHTS];
uniform float spot_light_angles[MAX_NUM_LIGHTS];

uniform sampler2DArray depthTextureArray;

uniform float spec_exp;

in vec3 position;
in vec3 normal;
in vec2 texcoord;
in vec3 dir2camera;
in mat3 tan2world;
in vec3 vertex_diffuse_color;
in vec4 position_shadowlight[MAX_NUM_LIGHTS];

out vec4 fragColor;

#define PI 3.14159265358979323846

//
// Noise functions
//

// Hash function for noise
vec2 hash( vec2 p ) {
    p = vec2( dot(p,vec2(127.1,311.7)), dot(p,vec2(269.5,183.3)) );
    return -1.0 + 2.0*fract(sin(p)*43758.5453123);
}

// Gradient noise (Perlin)
float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);

    vec2 u = f*f*(3.0-2.0*f); // Cubic interpolation

    // Sample corners
    float a = dot(hash(i + vec2(0.0,0.0)), f - vec2(0.0,0.0));
    float b = dot(hash(i + vec2(1.0,0.0)), f - vec2(1.0,0.0));
    float c = dot(hash(i + vec2(0.0,1.0)), f - vec2(0.0,1.0));
    float d = dot(hash(i + vec2(1.0,1.0)), f - vec2(1.0,1.0));

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Fractal Brownian Motion (fBm)
float fbm(vec2 p) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 3.0;
    // Add multiple layers of noise
    for(int i = 0; i < 5; i++) {
        value += amplitude * noise(p * frequency);
        frequency *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}

vec3 Diffuse_BRDF(vec3 L, vec3 N, vec3 diffuseColor) {
    return diffuseColor * max(dot(N, L), 0.);
}

vec3 Phong_BRDF(vec3 L, vec3 V, vec3 N, vec3 diffuse_color, vec3 specular_color, float specular_exponent)
{
    // Reduce diffuse contribution for water
    vec3 diffuse = diffuse_color * max(dot(N, L), 0.0) * 0.3;

    // Enhanced specular for water
    vec3 H = normalize(L + V);  // Half vector
    float NdotH = max(dot(N, H), 0.0);
    vec3 specular = specular_color * pow(NdotH, specular_exponent) * 2.0;

    return diffuse + specular;
}

vec3 SampleEnvironmentMap(vec3 D)
{
    vec3 dir = normalize(D);
    float theta = acos(dir.y);
    float phi = atan(dir.x, dir.z);
    if (phi < 0.0) {
        phi += 2.0 * PI;
    }
    float u = (2.0 * PI - phi) / (2.0 * PI);
    float v = theta / PI;
    return texture(environmentTextureSampler, vec2(u, v)).rgb;
}

// Add Fresnel calculation function before main()
float fresnel(float cosTheta, float F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

void main(void)
{
    vec3 diffuseColor = vec3(1.0, 1.0, 1.0);
    vec3 specularColor = vec3(1.0, 1.0, 1.0);
    float specularExponent = spec_exp;

    if (useTextureMapping) {
        // Use fbm for more natural water movement
        vec2 noise_offset = vec2(
            fbm(texcoord * 4.0 + time * 0.3),
            fbm(texcoord * 4.0 + vec2(2.0) + time * 0.3)
        ) * 0.1;

        vec2 moving_texcoord = texcoord + noise_offset;
        diffuseColor = texture(diffuseTextureSampler, moving_texcoord).rgb;
    } else {
        diffuseColor = vertex_diffuse_color;
    }

    vec3 N = vec3(0);
    if (useNormalMapping) {
        // Use fbm for normal map distortion
        vec2 noise_offset = vec2(
            fbm(position.xz * 0.5 + time * 0.2),
            fbm(position.xz * 0.5 + vec2(1.0) + time * 0.2)
        ) * 0.15;

        vec2 moving_normal_texcoord = texcoord + noise_offset;
        vec3 tangent_space_normal = texture(normalTextureSampler, moving_normal_texcoord).rgb * 2.0 - 1.0;
        N = normalize(tan2world * tangent_space_normal);
    } else {
        N = normalize(normal);
    }

    vec3 V = normalize(dir2camera);
    vec3 Lo = vec3(0.1 * diffuseColor);   // ambient

    // Enhanced environment reflection for water
    if (useEnvironmentMapping) {
        vec3 R = reflect(-V, N);
        vec3 envColor = SampleEnvironmentMap(R);

        // Fresnel factor for water (F0 = 0.02 for water)
        float F = fresnel(max(dot(N, V), 0.0), 0.02);

        // Blend environment reflection based on Fresnel
        Lo += envColor * F * 0.8;
    }

    // Lighting calculations
    for (int i = 0; i < num_directional_lights; ++i) {
        vec3 L = normalize(-directional_light_vectors[i]);
        vec3 brdf_color = Phong_BRDF(L, V, N, diffuseColor, specularColor, specularExponent);
        Lo += brdf_color;
    }

    for (int i = 0; i < num_point_lights; ++i) {
        vec3 light_vector = point_light_positions[i] - position;
        vec3 L = normalize(light_vector);
        float distance = length(light_vector);
        vec3 brdf_color = Phong_BRDF(L, V, N, diffuseColor, specularColor, specularExponent);
        float falloff = 1.0 / (0.01 + distance * distance);
        Lo += falloff * brdf_color;
    }

    for (int i = 0; i < num_spot_lights; ++i) {
        vec3 intensity = spot_light_intensities[i];
        vec3 light_pos = spot_light_positions[i];
        float cone_angle = spot_light_angles[i];

        vec3 dir_to_surface = position - light_pos;
        float angle = acos(dot(normalize(dir_to_surface), spot_light_directions[i])) * 180.0 / PI;

        float distance = length(dir_to_surface);
        float distance_attenuation = 1.0 / (1.0 + distance * distance);

        float SMOOTHING = 0.1;
        float angle_attenuation = 1.0;
        float inner_angle = (1.0 - SMOOTHING) * cone_angle;
        float outer_angle = (1.0 + SMOOTHING) * cone_angle;

        if (angle > outer_angle) {
            angle_attenuation = 0.0;
        } else if (angle > inner_angle) {
            angle_attenuation = 1.0 - (angle - inner_angle) / (outer_angle - inner_angle);
        }

        intensity *= distance_attenuation * angle_attenuation;

        float shadow = 1.0;
        if (i < num_spot_lights) {
            vec3 shadow_coords = position_shadowlight[i].xyz / position_shadowlight[i].w;
            float bias = 0.005;
            float pcf_step_size = 256.0;
            int shadow_samples = 0;
            int total_samples = 25;

            for (int j = -2; j <= 2; j++) {
                for (int k = -2; k <= 2; k++) {
                    vec2 offset = vec2(j, k) / pcf_step_size;
                    float depth_sample = texture(depthTextureArray, vec3(shadow_coords.xy + offset, i)).r;
                    if (shadow_coords.z - bias > depth_sample) {
                        shadow_samples++;
                    }
                }
            }
            shadow = 1.0 - float(shadow_samples) / float(total_samples);
        }

        vec3 L = normalize(-spot_light_directions[i]);
        vec3 brdf_color = Phong_BRDF(L, V, N, diffuseColor, specularColor, specularExponent);
        Lo += intensity * shadow * brdf_color;
    }

    // Adjust alpha based on view angle (more transparent when looking straight down)
    float alpha = mix(0.6, 0.9, fresnel(max(dot(N, V), 0.0), 0.02));
    fragColor = vec4(Lo, alpha);
}