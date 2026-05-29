# GalacticMergerSim

N-body simulation of two colliding galaxies for a cosmology class project.

## Notebook

Open and run [`galaxy_merger.ipynb`](galaxy_merger.ipynb) (Jupyter). It is self-contained and generates all outputs:

| Week | Output |
|------|--------|
| 1 | `week1_trajectories.png` — core orbits & energy conservation |
| 2 | `week2_particles.png` — stellar disks + NFW halos at t=0 and 4 Gyr |
| 3 | `merger_animation.gif`, `week3_tidal_analysis.png` |
| 4 | `merger_dataset.csv`, `week4_ml_results.png` |

## Setup

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
jupyter notebook galaxy_merger.ipynb
```

Full run (all weeks) takes roughly 2–3 minutes on a typical laptop.

## Units

- \(G = 1\)
- Mass in \(10^{11}\,M_\odot\)
- Distance in kpc
- Time: \(1\,\mathrm{tu} \approx 0.978\,\mathrm{Gyr}\)
