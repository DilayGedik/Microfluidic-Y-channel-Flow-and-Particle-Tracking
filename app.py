from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from simulation import (
    ChannelConfig,
    ParticleConfig,
    branch_basis,
    branch_outlet_counts,
    simulate_particles,
    trajectories_to_dataframe,
    velocity_field,
)


st.set_page_config(
    page_title="Y-Channel Microfluidic Simulator",
    page_icon="🧪",
    layout="wide",
)

st.title("Y-Channel Microfluidic Flow and Particle Tracking")
st.caption(
    "Interactive laminar-flow model for blood-scale fluids, nanoparticles, "
    "and drug-delivery microfluidics."
)

with st.sidebar:
    st.header("Channel")
    channel_width = st.slider(
        "Channel width (µm)", 60, 240, 120, 10
    )
    branch_angle = st.slider(
        "Branch angle (degrees)", 20, 60, 35, 1
    )
    max_velocity = st.slider(
        "Peak inlet velocity (µm/s)", 100, 3000, 900, 50
    )
    viscosity = st.number_input(
        "Dynamic viscosity (Pa·s)",
        min_value=0.0005,
        max_value=0.0200,
        value=0.0035,
        step=0.0005,
        format="%.4f",
    )

    st.header("Particles")
    particle_count = st.slider("Particle count", 10, 300, 80, 10)
    radius_nm = st.slider(
        "Particle radius (nm)", 10, 2000, 100, 10
    )
    brownian = st.toggle("Enable Brownian motion", value=True)
    duration_s = st.slider(
        "Simulation duration (s)", 0.4, 4.0, 1.6, 0.1
    )
    seed = st.number_input(
        "Random seed", min_value=0, max_value=100000, value=7
    )

    run = st.button(
        "Run simulation",
        type="primary",
        use_container_width=True,
    )


channel = ChannelConfig(
    half_width_um=channel_width / 2,
    branch_angle_deg=branch_angle,
    max_velocity_um_s=max_velocity,
    viscosity_pa_s=viscosity,
)

particles = ParticleConfig(
    count=particle_count,
    radius_nm=radius_nm,
    enable_brownian_motion=brownian,
    duration_s=duration_s,
    seed=int(seed),
)

parameter_signature = (
    channel,
    particles,
)

if run or "simulation" not in st.session_state:
    with st.spinner("Calculating velocity field and particle trajectories..."):
        st.session_state.simulation = simulate_particles(
            channel,
            particles,
        )
        st.session_state.channel = channel
        st.session_state.particles = particles
        st.session_state.signature = parameter_signature

times, positions, diffusion = st.session_state.simulation
channel = st.session_state.channel
particles = st.session_state.particles

if st.session_state.signature != parameter_signature:
    st.info("Parameters changed. Select **Run simulation** to update the results.")

speed_grid_size = 170
x_limit = (
    channel.branch_length_um * np.sin(channel.branch_angle_rad)
    + 1.4 * channel.half_width_um
)
y_max = (
    channel.branch_length_um * np.cos(channel.branch_angle_rad)
    + 1.2 * channel.half_width_um
)
xg = np.linspace(-x_limit, x_limit, speed_grid_size)
yg = np.linspace(
    -channel.inlet_length_um,
    y_max,
    speed_grid_size,
)
X, Y = np.meshgrid(xg, yg)
U, V = velocity_field(X, Y, channel)
speed = np.sqrt(U**2 + V**2)

counts = branch_outlet_counts(positions, channel)
final_time = times[-1]
mean_displacement = np.nanmean(
    np.linalg.norm(positions[-1] - positions[0], axis=1)
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Diffusion coefficient", f"{diffusion:.3f} µm²/s")
m2.metric("Mean displacement", f"{mean_displacement:.0f} µm")
m3.metric("Left / right split", f"{counts['Left branch']} / {counts['Right branch']}")
m4.metric("Simulated time", f"{final_time:.2f} s")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Flow field",
        "Particle trajectories",
        "Velocity profiles",
        "Data and model notes",
    ]
)

with tab1:
    fig, ax = plt.subplots(figsize=(9, 8))
    contour = ax.contourf(X, Y, speed, levels=24)
    ax.streamplot(
        xg,
        yg,
        np.nan_to_num(U),
        np.nan_to_num(V),
        density=1.15,
        linewidth=0.7,
        arrowsize=0.8,
    )
    ax.set_title("Velocity magnitude and streamlines")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_aspect("equal")
    fig.colorbar(contour, ax=ax, label="Speed (µm/s)")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with tab2:
    fig, ax = plt.subplots(figsize=(9, 8))
    stride = max(1, len(times) // 280)

    for particle_id in range(particles.count):
        xy = positions[::stride, particle_id]
        ax.plot(xy[:, 0], xy[:, 1], linewidth=0.9, alpha=0.7)

    ax.scatter(
        positions[0, :, 0],
        positions[0, :, 1],
        s=12,
        label="Injection",
    )
    ax.scatter(
        positions[-1, :, 0],
        positions[-1, :, 1],
        s=12,
        label="Final position",
    )
    ax.set_title("Particle trajectories")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylim(-channel.inlet_length_um, y_max)
    ax.set_aspect("equal")
    ax.legend()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    branch_df = pd.DataFrame(
        {
            "Region": list(counts.keys()),
            "Particles": list(counts.values()),
        }
    )
    st.bar_chart(branch_df.set_index("Region"))

with tab3:
    sample_x = np.linspace(
        -channel.half_width_um,
        channel.half_width_um,
        250,
    )
    _, inlet_v = velocity_field(
        sample_x,
        np.full_like(sample_x, -0.6 * channel.inlet_length_um),
        channel,
    )

    direction, normal = branch_basis(channel, 1)
    branch_center = 0.55 * channel.branch_length_um * direction
    branch_points = (
        branch_center[None, :]
        + sample_x[:, None] * normal[None, :]
    )
    branch_u, branch_v = velocity_field(
        branch_points[:, 0],
        branch_points[:, 1],
        channel,
    )
    branch_speed = np.sqrt(branch_u**2 + branch_v**2)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sample_x, inlet_v, label="Inlet axial velocity")
    ax.plot(sample_x, branch_speed, label="Daughter branch speed")
    ax.set_title("Parabolic laminar velocity profiles")
    ax.set_xlabel("Cross-channel position (µm)")
    ax.set_ylabel("Speed (µm/s)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown(
        """
        For pressure-driven laminar flow, the no-slip boundary condition gives
        zero speed at the walls and the maximum speed near the channel center.
        The daughter-branch speed is reduced to approximate flow splitting.
        """
    )

with tab4:
    trajectory_df = trajectories_to_dataframe(times, positions)
    st.subheader("Trajectory preview")
    st.dataframe(trajectory_df.head(1000), use_container_width=True)

    csv_bytes = trajectory_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download trajectory CSV",
        data=csv_bytes,
        file_name="y_channel_particle_trajectories.csv",
        mime="text/csv",
    )

    st.subheader("Model assumptions")
    st.markdown(
        """
        - Incompressible, steady, low-Reynolds-number flow.
        - Parabolic velocity profiles approximate pressure-driven laminar flow.
        - The geometry is a reduced-order 2D representation of a symmetric
          Y-bifurcation.
        - Particle motion combines advection with Stokes-Einstein diffusion.
        - Particle-particle interactions, red-cell deformation, lift forces,
          and full Navier-Stokes coupling are intentionally omitted.
        """
    )

    st.subheader("Portfolio interpretation")
    st.markdown(
        """
        This application demonstrates transport modeling, microfluidic geometry,
        numerical integration, Brownian particle tracking, parameterized
        simulation, scientific visualization, and exportable analysis data.
        """
    )
