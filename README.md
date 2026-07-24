# Y-Channel Microfluidic Flow and Particle Tracking

An interactive simulation of laminar flow and particle transport
through a symmetric Y shaped microchannel.

## What it demonstrates

- Pressure driven laminar velocity profiles
- Streamlines and velocity magnitude visualization
- Nanoparticle advection
- Optional Brownian diffusion using the Stokes-Einstein equation
- Particle trajectory visualization
- Branch outlet particle counts
- Downloadable trajectory data
- Reusable simulation functions separated from the interface

## Important modeling note

This is a reduced order 2D engineering model, not a full computational fluid 
dynamics solver. It is designed for rapid experimentation, portfolio
demonstration, and early design exploration. A publication grade model would
normally solve the Navier-Stokes equations on a meshed geometry and include
mesh convergence and validation studies.

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Create a virtual environment:

   Windows:

       python -m venv .venv
       .venv\Scripts\activate

   macOS or Linux:

       python3 -m venv .venv
       source .venv/bin/activate

4. Install dependencies:

       pip install -r requirements.txt

5. Start the application:

       streamlit run app.py

Streamlit will print a local URL, usually `http://localhost:8501`.

## Project structure

- `app.py`: interactive Streamlit interface and visualizations
- `simulation.py`: reusable geometry, flow, diffusion, and trajectory functions
- `requirements.txt`: Python dependencies
- `example_usage.py`: example of using the simulation module without Streamlit

## Reuse in another project

```python
from simulation import ChannelConfig, ParticleConfig, simulate_particles

channel = ChannelConfig(max_velocity_um_s=1200)
particles = ParticleConfig(count=100, radius_nm=80)

times, positions, diffusion = simulate_particles(channel, particles)
```

`positions` has shape:

    (time_steps, particle_count, 2)

The final dimension contains x and y positions in micrometers.

## Possible extensions

- Unequal daughter-branch flow rates
- Two inlet concentration mixing
- Magnetic or acoustic particle steering
- Particle wall adhesion
- Red blood cell margination
- Residence time distributions
- COMSOL, OpenFOAM, or FEniCS validation
- Optimization of branch geometry for targeted delivery
