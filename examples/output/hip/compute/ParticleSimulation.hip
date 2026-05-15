#include <hip/device_functions.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/math_functions.h>

struct Particle {
  float3 position;
  float3 velocity;
  float3 acceleration;
  float mass;
  float lifetime;
  float4 color;
  int type;
  bool active;
};

struct PhysicsConstants {
  float gravity;
  float damping;
  float timestep;
  float collision_radius;
  float3 world_bounds_min;
  float3 world_bounds_max;
  int max_particles;
  float attraction_strength;
};

struct SimulationState {
  int active_particle_count;
  int frame_number;
  float total_time;
  float3 attractor_position;
};

struct ParticleBuffer {
  Particle[LiteralNode(
      value = 4096,
      literal_type = PrimitiveType(name = int, size_bits = None))] particles;
};

struct AtomicCounters {
  int collision_count;
  int active_count;
  int spawn_count;
};

__device__ float random(
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 2) st) {
  return fracf(
      (sinf(dot(st.xy, make_float2(12.9898, 78.233))) * 43758.5453123));
}

__device__ VectorType(element_type = PrimitiveType(name = float,
                                                   size_bits = None),
                      size = 3)
    random3(VectorType(element_type = PrimitiveType(name = float,
                                                    size_bits = None),
                       size = 3) seed) {
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) p =
      make_float3(dot(seed, make_float3(127.1, 311.7, 74.7)),
                  dot(seed, make_float3(269.5, 183.3, 246.1)),
                  dot(seed, make_float3(113.5, 271.9, 124.6)));
  return (-1.0 + (2.0 * fracf((sinf(p) * 43758.5453123))));
}

__device__ VectorType(element_type = PrimitiveType(name = float,
                                                   size_bits = None),
                      size = 3)
    calculateAttraction(
        VectorType(element_type = PrimitiveType(name = float, size_bits = None),
                   size = 3) position,
        VectorType(element_type = PrimitiveType(name = float, size_bits = None),
                   size = 3) attractor_pos,
        float strength) {
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) direction = (attractor_pos - position);
  float distance = length(direction);
  if ((distance < 0.001)) {
    return make_float3(0.0);
  }
  float force = (strength / ((distance * distance) + 0.1));
  return (normalize(direction) * force);
}

__device__ bool checkCollision(
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) pos1,
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) pos2,
    float radius) {
  return (length((pos1 - pos2)) < (radius * 2.0));
}

__device__ VectorType(element_type = PrimitiveType(name = float,
                                                   size_bits = None),
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
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) relative_pos = (pos1 - pos2);
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) relative_vel = (vel1 - vel2);
  float distance = length(relative_pos);
  if ((distance < 0.001)) {
    return vel1;
  }
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) normal = (relative_pos / distance);
  float relative_speed = dot(relative_vel, normal);
  if ((relative_speed > 0.0)) {
    return vel1;
  }
  float impulse = ((2.0 * relative_speed) / (mass1 + mass2));
  return (vel1 - ((impulse * mass2) * normal));
}

__global__ void spawn() {
  if ((counters.active_count >= physics.max_particles)) {
    return;
  }
  int i = 0;
  for (; (i < physics.max_particles); ++i) {
    if (!particle_buffer.particles[i].active) {
      Particle new_particle;
      VectorType(element_type = PrimitiveType(name = float, size_bits = None),
                 size = 3) random_offset =
          random3(make_float3(i, sim_state.frame_number, sim_state.total_time));
      new_particle.position =
          (sim_state.attractor_position + (random_offset * 2.0));
      new_particle.velocity =
          (random3(make_float3((i + 1000), sim_state.frame_number,
                               sim_state.total_time)) *
           5.0);
      new_particle.acceleration = make_float3(0.0);
      new_particle.mass =
          (1.0 + (random(make_float2(i, sim_state.frame_number)) * 2.0));
      new_particle.lifetime =
          (10.0 +
           (random(make_float2((i + 500), sim_state.frame_number)) * 20.0));
      new_particle.color = make_float4(
          random(make_float2((i + 100), sim_state.frame_number)),
          random(make_float2((i + 200), sim_state.frame_number)),
          random(make_float2((i + 300), sim_state.frame_number)), 1.0);
      new_particle.type =
          int((random(make_float2((i + 400), sim_state.frame_number)) * 3.0));
      new_particle.active = true;
      particle_buffer.particles[i] = new_particle;
      atomicAdd(counters.active_count, 1);
      atomicAdd(counters.spawn_count, 1);
      break;
    }
  }
}
