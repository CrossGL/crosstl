
#version 450 core
struct Particle {
  vec3 position;
  vec3 velocity;
  vec3 acceleration;
  float mass;
  float lifetime;
  vec4 color;
  int type;
  bool active;
};
struct PhysicsConstants {
  float gravity;
  float damping;
  float timestep;
  float collision_radius;
  vec3 world_bounds_min;
  vec3 world_bounds_max;
  int max_particles;
  float attraction_strength;
};
struct SimulationState {
  int active_particle_count;
  int frame_number;
  float total_time;
  vec3 attractor_position;
};
struct ParticleBuffer {
  Particle particles[4096];
};
struct AtomicCounters {
  int collision_count;
  int active_count;
  int spawn_count;
};
layout(std140, binding = 0) int MAX_PARTICLES;
layout(std140, binding = 1) int WORKGROUP_SIZE;
layout(std140, binding = 2) float PI;
layout(std140, binding = 3)
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) GRAVITY_VECTOR;
float random(VectorType(element_type = PrimitiveType(name = float,
                                                     size_bits = None),
                        size = 2) st) {
  return IdentifierNode(name = fract)(
      (IdentifierNode(name = sin)(IdentifierNode(name = dot)(
           st.xy, IdentifierNode(name = vec2)(12.9898, 78.233))) *
       43758.5453123));
}

VectorType(element_type = PrimitiveType(name = float, size_bits = None),
           size = 3)
    random3(VectorType(element_type = PrimitiveType(name = float,
                                                    size_bits = None),
                       size = 3) seed) {
  vec3 p = IdentifierNode(name = vec3)(
      IdentifierNode(name = dot)(
          seed, IdentifierNode(name = vec3)(127.1, 311.7, 74.7)),
      IdentifierNode(name = dot)(
          seed, IdentifierNode(name = vec3)(269.5, 183.3, 246.1)),
      IdentifierNode(name = dot)(
          seed, IdentifierNode(name = vec3)(113.5, 271.9, 124.6)));
  return ((-1.0) + (2.0 * IdentifierNode(name = fract)((
                              IdentifierNode(name = sin)(p) * 43758.5453123))));
}

VectorType(element_type = PrimitiveType(name = float, size_bits = None),
           size = 3)
    calculateAttraction(
        VectorType(element_type = PrimitiveType(name = float, size_bits = None),
                   size = 3) position,
        VectorType(element_type = PrimitiveType(name = float, size_bits = None),
                   size = 3) attractor_pos,
        float strength) {
  vec3 direction = (attractor_pos - position);
  float distance = IdentifierNode(name = length)(direction);
  if ((distance < 0.001)) {
  }
  float force = (strength / ((distance * distance) + 0.1));
  return (IdentifierNode(name = normalize)(direction) * force);
}

bool checkCollision(
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) pos1,
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) pos2,
    float radius) {
  return (IdentifierNode(name = length)((pos1 - pos2)) < (radius * 2.0));
}

VectorType(element_type = PrimitiveType(name = float, size_bits = None),
           size = 3)
    resolveCollision(
        VectorType(element_type = PrimitiveType(name = float, size_bits = None),
                   size = 3) pos1,
        VectorType(element_type = PrimitiveType(name = float, size_bits = None),
                   size = 3) vel1,
        VectorType(element_type = PrimitiveType(name = float, size_bits = None),
                   size = 3) pos2,
        VectorType(element_type = PrimitiveType(name = float, size_bits = None),
                   size = 3) vel2,
        float mass1, float mass2) {
  vec3 relative_pos = (pos1 - pos2);
  vec3 relative_vel = (vel1 - vel2);
  float distance = IdentifierNode(name = length)(relative_pos);
  if ((distance < 0.001)) {
  }
  vec3 normal = (relative_pos / distance);
  float relative_speed = IdentifierNode(name = dot)(relative_vel, normal);
  if ((relative_speed > 0.0)) {
    return vel1;
  }
  float impulse = ((2.0 * relative_speed) / (mass1 + mass2));
  return (vel1 - ((impulse * mass2) * normal));
}

// Compute Shader
void main() {
  if ((counters.active_count >= physics.max_particles)) {
  }
  for (int i = 0;; (i < physics.max_particles); (++i)) {
    if ((!particle_buffer.particles[i].active)) {
      Particle new_particle;
      vec3 random_offset = IdentifierNode(name = random3)(IdentifierNode(
          name = vec3)(i, sim_state.frame_number, sim_state.total_time));
      new_particle.position =
          (sim_state.attractor_position + (random_offset * 2.0));
      new_particle.velocity =
          (IdentifierNode(name = random3)(IdentifierNode(name = vec3)(
               (i + 1000), sim_state.frame_number, sim_state.total_time)) *
           5.0);
      new_particle.acceleration = IdentifierNode(name = vec3)(0.0);
      new_particle.mass = (1.0 + (IdentifierNode(name = random)(IdentifierNode(
                                      name = vec2)(i, sim_state.frame_number)) *
                                  2.0));
      new_particle.lifetime =
          (10.0 + (IdentifierNode(name = random)(IdentifierNode(name = vec2)(
                       (i + 500), sim_state.frame_number)) *
                   20.0));
      new_particle.color = IdentifierNode(name = vec4)(
          IdentifierNode(name = random)(
              IdentifierNode(name = vec2)((i + 100), sim_state.frame_number)),
          IdentifierNode(name = random)(
              IdentifierNode(name = vec2)((i + 200), sim_state.frame_number)),
          IdentifierNode(name = random)(
              IdentifierNode(name = vec2)((i + 300), sim_state.frame_number)),
          1.0);
      new_particle.type = IdentifierNode(name = int)(
          (IdentifierNode(name = random)(
               IdentifierNode(name = vec2)((i + 400), sim_state.frame_number)) *
           3.0));
      new_particle.active = true;
      particle_buffer.particles[i] = new_particle;
      IdentifierNode(name = atomicAdd)(counters.active_count, 1);
      IdentifierNode(name = atomicAdd)(counters.spawn_count, 1);
      BreakNode(label = None);
    }
  }
}
