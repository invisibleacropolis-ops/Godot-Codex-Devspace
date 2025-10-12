# Galaxy Particle Effect

This document summarizes the structure of the `Galaxy.tscn` scene and how to tune it for alternative nebula or galaxy looks. It is intended to help engineers understand which particle layers are responsible for specific visual motifs seen in the concept art.

## Scene Overview

The `Galaxy` scene is a `Node3D` with five independent `GPUParticles3D` layers. Each layer uses its own `ParticleProcessMaterial`, gradient-driven color ramp, and billboarded `QuadMesh` draw pass to recreate the luminous spiral with surrounding celestial bodies.

| Node | Visual Role | Key Settings |
| --- | --- | --- |
| `Core` | Bright galactic nucleus | Fast orbit velocity and warm gradient ramp to create the intense white/yellow swirl in the center. |
| `NebulaSpiral` | Primary spiral arms | Mid-speed rotation with tangential acceleration to generate the magenta-to-cyan arms from the artwork. |
| `DustSwirl` | Diffuse outer haze | Large emission radius with soft gradient for the teal/purple dust cloud around the spiral. |
| `StarField` | Background star sparkle | Slow-moving particles, high spread, tiny scale for twinkling stars in the scene. |
| `Planets` | Large orbiting bodies | Sparse, slow orbits and oversized billboards to suggest the stylized planets. |

All layers disable local coordinates so the system can be instanced anywhere without accumulating transforms.

## Editing Guidelines

- **Color palettes:** Adjust the `Gradient` resources at the top of the scene file to recolor each layer. Gradients are authored in HDR to keep emission intensities vivid.
- **Spiral motion:** Combine `orbit_velocity`, `radial_velocity`, and `angular_velocity` in the process materials to tighten or loosen the spiral. Increasing `tangential_accel` introduces more curvature.
- **Layer density:** Modify the `amount` on the particle nodes to add or remove clusters. For performance-sensitive contexts, reduce `amount` evenly across layers to retain the look while lowering fill rate.
- **Glow control:** Emission strengths live in each `StandardMaterial3D`. Raising `emission_energy_multiplier` brightens the layer without touching the particle counts.
- **Scale:** The `scale_min`/`scale_max` ranges in the process materials are tuned for a hero shot. Shrink them uniformly if the effect needs to be viewed from a distance.

## Usage

Instance `Galaxy.tscn` wherever the effect is needed. Optionally add a `WorldEnvironment` with a black background and enable glow in the project rendering settings to achieve the luminous look showcased in the reference image.
