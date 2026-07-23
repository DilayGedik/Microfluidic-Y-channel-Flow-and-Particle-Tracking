from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ChannelConfig:
    """Geometry and fluid settings for a symmetric Y-shaped microchannel."""

    half_width_um: float = 60.0
    inlet_length_um: float = 500.0
    branch_length_um: float = 650.0
    branch_angle_deg: float = 35.0
    max_velocity_um_s: float = 900.0
    viscosity_pa_s: float = 0.0035
    temperature_k: float = 310.15

    @property
    def branch_angle_rad(self) -> float:
        return np.deg2rad(self.branch_angle_deg)


@dataclass(frozen=True)
class ParticleConfig:
    """Particle settings. Radius is used to calculate Stokes-Einstein diffusion."""

    count: int = 80
    radius_nm: float = 100.0
    enable_brownian_motion: bool = True
    seed: int = 7
    dt_s: float = 0.002
    duration_s: float = 1.6


def stokes_einstein_diffusion_um2_s(
    radius_nm: float,
    viscosity_pa_s: float,
    temperature_k: float,
) -> float:
    """Return translational diffusion coefficient in µm²/s."""
    if radius_nm <= 0:
        raise ValueError("Particle radius must be positive.")
    if viscosity_pa_s <= 0:
        raise ValueError("Viscosity must be positive.")
    if temperature_k <= 0:
        raise ValueError("Temperature must be positive.")

    boltzmann = 1.380649e-23
    radius_m = radius_nm * 1e-9
    diffusion_m2_s = boltzmann * temperature_k / (
        6.0 * np.pi * viscosity_pa_s * radius_m
    )
    return float(diffusion_m2_s * 1e12)


def branch_basis(config: ChannelConfig, side: int) -> tuple[np.ndarray, np.ndarray]:
    """Return branch centerline direction and transverse normal."""
    theta = config.branch_angle_rad
    direction = np.array([side * np.sin(theta), np.cos(theta)], dtype=float)
    normal = np.array([np.cos(theta), -side * np.sin(theta)], dtype=float)
    return direction, normal


def point_in_channel(
    x_um: np.ndarray | float,
    y_um: np.ndarray | float,
    config: ChannelConfig,
) -> np.ndarray:
    """Boolean mask identifying points inside the inlet or either branch."""
    x = np.asarray(x_um, dtype=float)
    y = np.asarray(y_um, dtype=float)

    inlet = (
        (y >= -config.inlet_length_um)
        & (y <= 0.0)
        & (np.abs(x) <= config.half_width_um)
    )

    branch_masks = []
    for side in (-1, 1):
        direction, normal = branch_basis(config, side)
        s = x * direction[0] + y * direction[1]
        n = x * normal[0] + y * normal[1]
        branch_masks.append(
            (s >= 0.0)
            & (s <= config.branch_length_um)
            & (np.abs(n) <= config.half_width_um)
            & ((x * side) >= -0.25 * config.half_width_um)
        )

    return inlet | branch_masks[0] | branch_masks[1]


def velocity_field(
    x_um: np.ndarray | float,
    y_um: np.ndarray | float,
    config: ChannelConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Approximate a pressure-driven laminar velocity field.

    The inlet and branches use parabolic cross-sectional profiles. A smooth
    junction blend avoids a discontinuous velocity direction at the split.
    """
    x = np.asarray(x_um, dtype=float)
    y = np.asarray(y_um, dtype=float)
    u = np.zeros_like(x)
    v = np.zeros_like(y)

    # Inlet: vertical flow with a parabolic profile.
    inlet_mask = (
        (y <= 0.0)
        & (y >= -config.inlet_length_um)
        & (np.abs(x) <= config.half_width_um)
    )
    inlet_profile = np.clip(
        1.0 - (x / config.half_width_um) ** 2,
        0.0,
        None,
    )
    v_inlet = config.max_velocity_um_s * inlet_profile

    # Smoothly rotate the flow near the bifurcation.
    blend_height = 1.5 * config.half_width_um
    blend = np.clip((y + blend_height) / blend_height, 0.0, 1.0)

    u[inlet_mask] = 0.0
    v[inlet_mask] = v_inlet[inlet_mask]

    for side in (-1, 1):
        direction, normal = branch_basis(config, side)
        s = x * direction[0] + y * direction[1]
        n = x * normal[0] + y * normal[1]

        branch_mask = (
            (s >= 0.0)
            & (s <= config.branch_length_um)
            & (np.abs(n) <= config.half_width_um)
            & ((x * side) >= -0.25 * config.half_width_um)
        )
        profile = np.clip(
            1.0 - (n / config.half_width_um) ** 2,
            0.0,
            None,
        )

        # Each daughter branch receives approximately half the parent flow.
        branch_speed = 0.58 * config.max_velocity_um_s * profile
        u_branch = branch_speed * direction[0]
        v_branch = branch_speed * direction[1]

        # Blend inlet-like and branch-like directions close to the junction.
        local_blend = blend
        u_candidate = local_blend * u_branch
        v_candidate = (1.0 - local_blend) * (
            config.max_velocity_um_s * np.clip(
                1.0 - (x / config.half_width_um) ** 2, 0.0, None
            )
        ) + local_blend * v_branch

        u[branch_mask] = u_candidate[branch_mask]
        v[branch_mask] = v_candidate[branch_mask]

    inside = point_in_channel(x, y, config)
    u = np.where(inside, u, np.nan)
    v = np.where(inside, v, np.nan)
    return u, v


def _reflect_to_channel(
    previous: np.ndarray,
    proposed: np.ndarray,
    config: ChannelConfig,
) -> np.ndarray:
    """
    Apply a simple no-penetration correction.

    Invalid moves are repeatedly shortened toward the previous valid point.
    This is stable for visualization and particle-tracking demonstrations.
    """
    corrected = proposed.copy()
    invalid = ~point_in_channel(corrected[:, 0], corrected[:, 1], config)

    for _ in range(10):
        if not np.any(invalid):
            break
        corrected[invalid] = 0.5 * (
            corrected[invalid] + previous[invalid]
        )
        invalid = ~point_in_channel(
            corrected[:, 0], corrected[:, 1], config
        )

    corrected[invalid] = previous[invalid]
    return corrected


def simulate_particles(
    channel: ChannelConfig,
    particles: ParticleConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Simulate particle trajectories using advection and Brownian diffusion.

    Returns
    -------
    times_s:
        Array with shape (time_steps,).
    positions_um:
        Array with shape (time_steps, particle_count, 2).
    diffusion_um2_s:
        Stokes-Einstein diffusion coefficient used in the model.
    """
    if particles.count < 1:
        raise ValueError("Particle count must be at least 1.")
    if particles.dt_s <= 0 or particles.duration_s <= 0:
        raise ValueError("Time step and duration must be positive.")

    rng = np.random.default_rng(particles.seed)
    steps = int(np.ceil(particles.duration_s / particles.dt_s)) + 1
    times = np.arange(steps, dtype=float) * particles.dt_s

    positions = np.full((steps, particles.count, 2), np.nan, dtype=float)
    positions[0, :, 0] = rng.uniform(
        -0.8 * channel.half_width_um,
        0.8 * channel.half_width_um,
        particles.count,
    )
    positions[0, :, 1] = -0.96 * channel.inlet_length_um

    diffusion = stokes_einstein_diffusion_um2_s(
        particles.radius_nm,
        channel.viscosity_pa_s,
        channel.temperature_k,
    )
    brownian_sigma = (
        np.sqrt(2.0 * diffusion * particles.dt_s)
        if particles.enable_brownian_motion
        else 0.0
    )

    active = np.ones(particles.count, dtype=bool)

    for k in range(1, steps):
        previous = positions[k - 1].copy()
        current = previous.copy()

        if not np.any(active):
            positions[k:] = positions[k - 1]
            break

        active_idx = np.where(active)[0]
        x = previous[active_idx, 0]
        y = previous[active_idx, 1]
        u, v = velocity_field(x, y, channel)
        velocity = np.column_stack(
            [np.nan_to_num(u), np.nan_to_num(v)]
        )

        # A small unbiased junction perturbation prevents numerical crowding
        # exactly on the symmetry line without forcing branch assignment.
        near_center = (
            (np.abs(x) < 0.02 * channel.half_width_um)
            & (y > -0.3 * channel.half_width_um)
            & (y < 0.4 * channel.half_width_um)
        )
        velocity[near_center, 0] += rng.normal(
            0.0,
            0.015 * channel.max_velocity_um_s,
            np.count_nonzero(near_center),
        )

        displacement = velocity * particles.dt_s
        if particles.enable_brownian_motion:
            displacement += rng.normal(
                0.0,
                brownian_sigma,
                size=displacement.shape,
            )

        proposed = previous[active_idx] + displacement
        current[active_idx] = _reflect_to_channel(
            previous[active_idx],
            proposed,
            channel,
        )

        # Stop particles after they reach a branch outlet.
        for side in (-1, 1):
            direction, _ = branch_basis(channel, side)
            s = (
                current[active_idx, 0] * direction[0]
                + current[active_idx, 1] * direction[1]
            )
            exited_local = s >= 0.995 * channel.branch_length_um
            active[active_idx[exited_local]] = False

        positions[k] = current

    return times, positions, diffusion


def trajectories_to_dataframe(
    times_s: np.ndarray,
    positions_um: np.ndarray,
) -> pd.DataFrame:
    """Convert trajectory arrays into a tidy table suitable for CSV export."""
    step_count, particle_count, _ = positions_um.shape
    return pd.DataFrame(
        {
            "time_s": np.repeat(times_s, particle_count),
            "particle_id": np.tile(np.arange(particle_count), step_count),
            "x_um": positions_um[:, :, 0].reshape(-1),
            "y_um": positions_um[:, :, 1].reshape(-1),
        }
    )


def branch_outlet_counts(
    positions_um: np.ndarray,
    channel: ChannelConfig,
) -> dict[str, int]:
    """Classify final particle positions by left branch, right branch, or channel."""
    final = positions_um[-1]
    left = int(np.count_nonzero(final[:, 0] < -0.2 * channel.half_width_um))
    right = int(np.count_nonzero(final[:, 0] > 0.2 * channel.half_width_um))
    center = int(len(final) - left - right)
    return {"Left branch": left, "Right branch": right, "In channel": center}
