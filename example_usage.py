from simulation import (
    ChannelConfig,
    ParticleConfig,
    branch_outlet_counts,
    simulate_particles,
    trajectories_to_dataframe,
)


def main() -> None:
    channel = ChannelConfig(
        half_width_um=60,
        branch_angle_deg=35,
        max_velocity_um_s=900,
    )
    particles = ParticleConfig(
        count=50,
        radius_nm=100,
        enable_brownian_motion=True,
        duration_s=1.6,
        seed=7,
    )

    times, positions, diffusion = simulate_particles(channel, particles)
    results = trajectories_to_dataframe(times, positions)
    counts = branch_outlet_counts(positions, channel)

    results.to_csv("example_trajectories.csv", index=False)
    print(f"Diffusion coefficient: {diffusion:.4f} µm²/s")
    print(f"Outlet counts: {counts}")
    print("Saved example_trajectories.csv")


if __name__ == "__main__":
    main()
