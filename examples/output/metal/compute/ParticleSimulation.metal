
#include <metal_stdlib>
using namespace metal;

__attribute__((unused)) constant int MAX_PARTICLES = 4096;
__attribute__((unused)) constant int WORKGROUP_SIZE = 64;
__attribute__((unused)) constant float PI = 3.14159265359;
__attribute__((unused)) constant float3 GRAVITY_VECTOR =
    float3(0.0, -9.81, 0.0);

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
  Particle particles[4096];
};
struct AtomicCounters {
  int collision_count;
  int active_count;
  int spawn_count;
};
float random(float2 st) {
  return fract(sin(dot(st.xy, float2(12.9898, 78.233))) * 43758.5453123);
}

float3 random3(float3 seed) {
  float3 p = float3(dot(seed, float3(127.1, 311.7, 74.7)),
                    dot(seed, float3(269.5, 183.3, 246.1)),
                    dot(seed, float3(113.5, 271.9, 124.6)));
  return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

float3 calculateAttraction(float3 position, float3 attractor_pos,
                           float strength) {
  float3 direction = attractor_pos - position;
  float distance = length(direction);
  if (distance < 0.001) {
    return float3(0.0);
  }
  float force = strength / distance * distance + 0.1;
  return normalize(direction) * float3(force);
}

bool checkCollision(float3 pos1, float3 pos2, float radius) {
  return length(pos1 - pos2) < radius * 2.0;
}

float3 resolveCollision(float3 pos1, float3 vel1, float3 pos2, float3 vel2,
                        float mass1, float mass2) {
  float3 relative_pos = pos1 - pos2;
  float3 relative_vel = vel1 - vel2;
  float distance = length(relative_pos);
  if (distance < 0.001) {
    return vel1;
  }
  float3 normal = relative_pos / float3(distance);
  float relative_speed = dot(relative_vel, normal);
  if (relative_speed > 0.0) {
    return vel1;
  }
  float impulse = 2.0 * relative_speed / mass1 + mass2;
  return vel1 - float3(impulse * mass2) * normal;
}

// Compute Shader
kernel void kernel_main(uint3 gl_GlobalInvocationID [[thread_position_in_grid]],
                        uint3 gl_LocalInvocationID
                        [[thread_position_in_threadgroup]],
                        constant PhysicsConstants &physics [[buffer(0)]],
                        constant SimulationState &sim_state [[buffer(1)]],
                        device ParticleBuffer &particle_buffer [[buffer(2)]],
                        device AtomicCounters &counters [[buffer(3)]]) {
  __attribute__((unused)) threadgroup float3 shared_positions[WORKGROUP_SIZE];
  __attribute__((unused)) threadgroup float3 shared_velocities[WORKGROUP_SIZE];
  __attribute__((unused)) threadgroup int shared_types[WORKGROUP_SIZE];
  __attribute__((unused)) threadgroup bool shared_active[WORKGROUP_SIZE];
  int particle_id = int(gl_GlobalInvocationID.x);
  int local_id = int(gl_LocalInvocationID.x);
  if (particle_id >= physics.max_particles) {
    return;
  }
  Particle particle = particle_buffer.particles[particle_id];
  if (!particle.active) {
    shared_active[local_id] = false;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return;
  }
  shared_positions[local_id] = particle.position;
  shared_velocities[local_id] = particle.velocity;
  shared_types[local_id] = particle.type;
  shared_active[local_id] = true;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float3 force = float3(0.0);
  if (particle.type == 0) {
    force += GRAVITY_VECTOR * particle.mass;
  }
  if (particle.type != 2) {
    force +=
        calculateAttraction(particle.position, sim_state.attractor_position,
                            physics.attraction_strength);
  }
  int collision_count = 0;
  for (int i = 0; i < WORKGROUP_SIZE; ++i) {
    if (i == local_id || !shared_active[i]) {
      continue;
    }
    if (checkCollision(particle.position, shared_positions[i],
                       physics.collision_radius)) {
      ++collision_count;
      particle.velocity = resolveCollision(
          particle.position, particle.velocity, shared_positions[i],
          shared_velocities[i], particle.mass, 1.0);
      particle.color = mix(particle.color, float4(1.0, 0.0, 0.0, 1.0), 0.1);
    }
    if (shared_types[i] == 1 && particle.type == 0) {
      float3 attraction =
          calculateAttraction(particle.position, shared_positions[i], 0.5);
      force += attraction;
    }
  }
  particle.acceleration = force / float3(particle.mass);
  particle.velocity += particle.acceleration * float3(physics.timestep);
  particle.velocity *= physics.damping;
  particle.position += particle.velocity * float3(physics.timestep);
  if (particle.position.x < physics.world_bounds_min.x ||
      particle.position.x > physics.world_bounds_max.x) {
    particle.velocity.x *= -0.8;
    particle.position.x = clamp(particle.position.x, physics.world_bounds_min.x,
                                physics.world_bounds_max.x);
  }
  if (particle.position.y < physics.world_bounds_min.y ||
      particle.position.y > physics.world_bounds_max.y) {
    particle.velocity.y *= -0.8;
    particle.position.y = clamp(particle.position.y, physics.world_bounds_min.y,
                                physics.world_bounds_max.y);
  }
  if (particle.position.z < physics.world_bounds_min.z ||
      particle.position.z > physics.world_bounds_max.z) {
    particle.velocity.z *= -0.8;
    particle.position.z = clamp(particle.position.z, physics.world_bounds_min.z,
                                physics.world_bounds_max.z);
  }
  particle.lifetime -= physics.timestep;
  if (particle.lifetime <= 0.0) {
    particle.active = false;
    atomic_fetch_add_explicit(
        reinterpret_cast<device atomic_int *>(&counters.active_count), -1,
        memory_order_relaxed);
  }
  float speed = length(particle.velocity);
  particle.color.rgb = mix(float3(0.2, 0.4, 1.0), float3(1.0, 0.4, 0.2),
                           clamp(speed / 10.0, 0.0, 1.0));
  particle_buffer.particles[particle_id] = particle;
  if (collision_count > 0) {
    atomic_fetch_add_explicit(
        reinterpret_cast<device atomic_int *>(&counters.collision_count),
        collision_count, memory_order_relaxed);
  }
}

// Compute Shader
kernel void kernel_spawn(constant PhysicsConstants &physics [[buffer(0)]],
                         constant SimulationState &sim_state [[buffer(1)]],
                         device ParticleBuffer &particle_buffer [[buffer(2)]],
                         device AtomicCounters &counters [[buffer(3)]]) {
  if (counters.active_count >= physics.max_particles) {
    return;
  }
  for (int i = 0; i < physics.max_particles; ++i) {
    if (!particle_buffer.particles[i].active) {
      Particle new_particle;
      float3 random_offset =
          random3(float3(i, sim_state.frame_number, sim_state.total_time));
      new_particle.position =
          sim_state.attractor_position + random_offset * float3(2.0);
      new_particle.velocity = random3(float3(i + 1000, sim_state.frame_number,
                                             sim_state.total_time)) *
                              float3(5.0);
      new_particle.acceleration = float3(0.0);
      new_particle.mass = 1.0 + random(float2(i, sim_state.frame_number)) * 2.0;
      new_particle.lifetime =
          10.0 + random(float2(i + 500, sim_state.frame_number)) * 20.0;
      new_particle.color =
          float4(random(float2(i + 100, sim_state.frame_number)),
                 random(float2(i + 200, sim_state.frame_number)),
                 random(float2(i + 300, sim_state.frame_number)), 1.0);
      new_particle.type =
          int(random(float2(i + 400, sim_state.frame_number)) * 3.0);
      new_particle.active = true;
      particle_buffer.particles[i] = new_particle;
      atomic_fetch_add_explicit(
          reinterpret_cast<device atomic_int *>(&counters.active_count), 1,
          memory_order_relaxed);
      atomic_fetch_add_explicit(
          reinterpret_cast<device atomic_int *>(&counters.spawn_count), 1,
          memory_order_relaxed);
      break;
    }
  }
}
