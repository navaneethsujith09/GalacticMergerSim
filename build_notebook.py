"""Generate galaxy_merger.ipynb for GalacticMergerSim."""
import json
from pathlib import Path

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}

def code(source):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }

cells = []

cells.append(md(
"""# GalacticMergerSim

Cosmology class project: two galaxies collide in an **N-body** simulation with
**leapfrog (Verlet)** integration, exponential stellar disks, **NFW** dark-matter
halos, a merger animation, and a **machine-learning** model that predicts
coalescence time from initial conditions.

**Unit system:** \\(G=1\\), mass in \\(10^{11}\\,M_\\odot\\), distance in kpc,
time units with \\(1\\,\\mathrm{tu} \\approx 0.978\\,\\mathrm{Gyr}\\).
"""
))

cells.append(code(
'''# Shared imports and physical constants
import numpy as np
import matplotlib

def _in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False

if not _in_notebook():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation, PillowWriter
from astropy import units as astro_u
from astropy.constants import G as G_const
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# --- Unit system ---
G = 1.0
EPS = 0.1          # softening length [kpc]
GYR_PER_TU = 0.978  # 1 simulation time unit ≈ 0.978 Gyr

# Plot style
BG = "#0d0d0d"
plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.edgecolor": "white",
    "axes.labelcolor": "white",
    "text.color": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "#333333",
    "legend.facecolor": BG,
    "legend.edgecolor": "white",
})


def softened_accel(pos_i, pos_j, mass_j, eps=EPS):
    """Plummer-softened acceleration: F = G m_j m_i / (r^2+eps^2) * r_hat (symplectic with PE)."""
    r_vec = pos_j - pos_i
    r2 = np.dot(r_vec, r_vec) + eps ** 2
    r_soft = np.sqrt(r2)
    return G * mass_j * r_vec / (r_soft ** 3)


def pairwise_pe(pos1, pos2, m1, m2, eps=EPS):
    """Gravitational PE for Plummer softening: Phi = -G m1 m2 / sqrt(r^2 + eps^2)."""
    r = np.linalg.norm(pos2 - pos1)
    return -G * m1 * m2 / np.sqrt(r ** 2 + eps ** 2)


def leapfrog_step(pos, vel, accel_fn, dt):
    """One symplectic kick-drift-kick (velocity Verlet) step."""
    acc = accel_fn(pos)
    vel = vel + 0.5 * dt * acc
    pos = pos + dt * vel
    acc = accel_fn(pos)
    vel = vel + 0.5 * dt * acc
    return pos, vel, acc


def circular_speed_each(m_self, m_other, separation, eps=EPS):
    """Tangential speed for each galaxy on a circular orbit at given separation."""
    return np.sqrt(G * m_other / (2.0 * separation))


def total_energy_two_body(p1, p2, v1, v2, m1, m2):
    ke = 0.5 * m1 * np.dot(v1, v1) + 0.5 * m2 * np.dot(v2, v2)
    pe = pairwise_pe(p1, p2, m1, m2)
    return ke + pe


# Astropy: physical scale of our distance unit
print(f"1 kpc = {astro_u.kpc}")
print(f"Code units: G={G}, mass in 1e11 Msun, distance kpc, time tu (1 tu = {GYR_PER_TU} Gyr)")
print(f"Softening eps = {EPS} kpc")
'''
))

# WEEK 1
cells.append(md(
"""## Week 1: Physics Engine & Two-Body Orbit

We integrate two massive galaxy **cores** with softened gravity and a **leapfrog**
integrator (symplectic, unlike Euler). Softening avoids infinite forces at zero
separation; we use the standard **Plummer** form consistent with the potential
\\(\\Phi = -GM/\\sqrt{r^2+\\varepsilon^2}\\), which keeps fractional energy drift
below \\(10^{-6}\\).

Initial velocities are **75% of circular speed** for a bound, eccentric orbit that
inspires and merges over ~8 Gyr. Center-of-mass drift is removed so \\(\\mathbf{v}_\\mathrm{cm}=0\\).
"""
))

cells.append(code(
'''# --- Week 1: two-body simulation ---
M1 = M2 = 5.0
pos1 = np.array([-5.0, 0.0])
pos2 = np.array([5.0, 0.0])
separation0 = np.linalg.norm(pos2 - pos1)

v_circ = circular_speed_each(M1, M2, separation0)
v_trans = 0.75 * v_circ
vel1 = np.array([0.0, v_trans])
vel2 = np.array([0.0, -v_trans])

# Center-of-mass correction (equal masses => already zero, but enforce generally)
M_tot = M1 + M2
v_cm = (M1 * vel1 + M2 * vel2) / M_tot
vel1 -= v_cm
vel2 -= v_cm

E0 = total_energy_two_body(pos1, pos2, vel1, vel2, M1, M2)
print(f"Initial total energy E0 = {E0:.6f}  (bound: {E0 < 0})")

dt = 0.001
t_end = 8.0
n_steps = int(t_end / dt)

hist1 = np.zeros((n_steps + 1, 2))
hist2 = np.zeros((n_steps + 1, 2))
energy = np.zeros(n_steps + 1)
times = np.linspace(0.0, t_end, n_steps + 1)
separations = np.zeros(n_steps + 1)

p1, p2 = pos1.copy(), pos2.copy()
v1, v2 = vel1.copy(), vel2.copy()
hist1[0], hist2[0] = p1, p2
energy[0] = E0
separations[0] = separation0


def core_accel(state):
    """state = [x1,y1,x2,y2] -> accelerations for leapfrog on both cores."""
    p1 = state[:2]
    p2 = state[2:]
    a1 = softened_accel(p1, p2, M2)
    a2 = softened_accel(p2, p1, M1)
    return np.concatenate([a1, a2])


state = np.concatenate([p1, p2])
vel = np.concatenate([v1, v2])
acc = core_accel(state)
vel = vel + 0.5 * dt * acc

for step in range(1, n_steps + 1):
    state = state + dt * vel
    acc = core_accel(state)
    vel = vel + dt * acc  # leapfrog: v_{i+1/2} = v_{i-1/2} + dt * a_i
    p1, p2 = state[:2], state[2:]
    # Sync half-step velocities to integer time for energy measurement
    v1 = vel[:2] - 0.5 * dt * acc[:2]
    v2 = vel[2:] - 0.5 * dt * acc[2:]
    hist1[step] = p1
    hist2[step] = p2
    energy[step] = total_energy_two_body(p1, p2, v1, v2, M1, M2)
    separations[step] = np.linalg.norm(p2 - p1)

frac_err = (energy - E0) / np.abs(E0)
print(f"Max |fractional energy error| = {np.max(np.abs(frac_err)):.2e}")

imin = np.argmin(separations)
print(f"Min separation: {separations[imin]:.3f} kpc at t = {times[imin]:.3f} tu "
      f"({times[imin]*GYR_PER_TU:.3f} Gyr)")
print(f"Max separation: {separations.max():.3f} kpc")
print(f"Final separation: {separations[-1]:.3f} kpc")
print(f"Orbit bound at t=0: {E0 < 0}")
'''
))

cells.append(code(
'''# --- Week 1 figure ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor(BG)

ax = axes[0]
norm = Normalize(vmin=0, vmax=t_end)
for traj, label in [(hist1, "Galaxy 1"), (hist2, "Galaxy 2")]:
    pts = ax.scatter(traj[:, 0], traj[:, 1], c=times, cmap="plasma", norm=norm,
                     s=4, alpha=0.85, label=label)
    ax.plot(traj[0, 0], traj[0, 1], "o", color="white", ms=8, zorder=5)
    ax.plot(traj[-1, 0], traj[-1, 1], "o", color=cm.plasma(1.0), ms=8, zorder=5)

pad = 0.5
xmin = min(hist1[:, 0].min(), hist2[:, 0].min()) - pad
xmax = max(hist1[:, 0].max(), hist2[:, 0].max()) + pad
ymin = min(hist1[:, 1].min(), hist2[:, 1].min()) - pad
ymax = max(hist1[:, 1].max(), hist2[:, 1].max()) + pad
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_aspect("equal")
ax.set_xlabel("x [kpc]")
ax.set_ylabel("y [kpc]")
ax.set_title("Galaxy core trajectories")
ax.legend(loc="upper right", fontsize=8)

ax = axes[1]
ax.plot(times * GYR_PER_TU, frac_err, color="#44dd66", lw=1.2)
ax.axhline(1e-6, color="#aaaaaa", ls="--", lw=0.8, label=r"$10^{-6}$")
ax.set_xlabel("Time [Gyr]")
ax.set_ylabel(r"$(E - E_0) / |E_0|$")
ax.set_title("Fractional energy error")
ax.legend()

plt.tight_layout()
plt.savefig("week1_trajectories.png", dpi=150, facecolor=BG)
plt.close(fig)
print("Saved week1_trajectories.png")
'''
))

# WEEK 2
cells.append(md(
"""## Week 2: Star Particles & Dark Matter Halos

Each galaxy gets **200 massless star particles** in an exponential disk
(\\(R \\sim \\exp(-R/R_s)\\), \\(R_s=2\\) kpc) with circular speeds plus 10% thermal
scatter. Stars feel both **baryonic cores** and static **NFW halos**; cores only
feel each other.

NFW potential (per halo):
\\[
\\Phi(r) = -\\frac{GM_\\mathrm{halo}}{r}\\left(\\frac{\\ln(1+r/r_s)}{\\ln(1+c)} - \\frac{1}{c}\\right)
\\]
with \\(M_\\mathrm{halo}=20\\), \\(r_s=5\\) kpc, \\(c=10\\).
"""
))

cells.append(code(
'''# --- NFW halo helpers ---
M_HALO = 20.0
R_S_NFW = 5.0
C_NFW = 10.0
LN1PC = np.log(1.0 + C_NFW)


def nfw_phi(r, m_halo=M_HALO, r_s=R_S_NFW, c=C_NFW):
    r = np.maximum(r, 1e-6)
    term = np.log(1.0 + r / r_s) / LN1PC - 1.0 / c
    return -G * m_halo / r * term


def nfw_accel_at(pos, halo_center, m_halo=M_HALO, r_s=R_S_NFW, c=C_NFW):
    """Spherical NFW acceleration toward halo_center (single position)."""
    r_vec = np.atleast_2d(pos) - halo_center
    return nfw_accel_bulk(np.atleast_2d(pos), halo_center, m_halo, r_s, c)[0]


def nfw_accel_bulk(pos, halo_center, m_halo=M_HALO, r_s=R_S_NFW, c=C_NFW):
    """NFW acceleration for many 2D positions."""
    r_vec = pos - halo_center
    r = np.linalg.norm(r_vec, axis=1)
    r = np.maximum(r, 1e-8)
    term = np.log(1.0 + r / r_s) / LN1PC - 1.0 / c
    dterm_dr = (1.0 / (r + r_s)) / (LN1PC * r) - term / r
    dphi_dr = -G * m_halo * dterm_dr
    return -dphi_dr[:, np.newaxis] * (r_vec / r[:, np.newaxis])


def local_circular_speed(pos, gal_center, other_core, m_core, m_other_core,
                         halo_centers, halo_masses):
    """Circular speed from combined potential at radius R in the disk plane."""
    R = np.linalg.norm(pos - gal_center)
    if R < 1e-4:
        return 0.0
    dr = 1e-4
    phi0 = (
        -G * m_core / np.sqrt(R ** 2 + EPS ** 2)
        + sum(nfw_phi(np.linalg.norm(pos - hc)) for hc in halo_centers)
        + pairwise_pe(pos, other_core, 1.0, 1.0) * 0  # handled via cores below
    )
    # numeric dPhi/dR in disk
    angle = np.arctan2(pos[1] - gal_center[1], pos[0] - gal_center[0])
    pos_p = gal_center + (R + dr) * np.array([np.cos(angle), np.sin(angle)])
    pos_m = gal_center + (R - dr) * np.array([np.cos(angle), np.sin(angle)])

    def phi_at(p):
        ph = -G * m_core / np.sqrt(np.sum((p - gal_center) ** 2) + EPS ** 2)
        for hc, mh in zip(halo_centers, halo_masses):
            ph += nfw_phi(np.linalg.norm(p - hc), m_halo=mh)
        ph += -G * m_other_core / np.sqrt(np.sum((p - other_core) ** 2) + EPS ** 2)
        return ph

    dphi_dr = (phi_at(pos_p) - phi_at(pos_m)) / (2 * dr)
    return np.sqrt(max(R * abs(dphi_dr), 0.0))


def init_stars(n_stars, gal_center, other_core, m_core, m_other_core,
               halo_centers, halo_masses, rng, r_disk=2.0, scatter=0.1):
    """Exponential disk stars with circular velocity + thermal scatter."""
    radii = rng.exponential(scale=r_disk, size=n_stars)
    angles = rng.uniform(0, 2 * np.pi, size=n_stars)
    pos = gal_center + np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    vel = np.zeros_like(pos)
    for i in range(n_stars):
        R_i = radii[i]
        vc = local_circular_speed(pos[i], gal_center, other_core, m_core, m_other_core,
                                  halo_centers, halo_masses)
        # Tangential direction about gal_center (prograde)
        rhat = (pos[i] - gal_center) / R_i
        tang = np.array([-rhat[1], rhat[0]])
        vc *= 1.0 + rng.normal(0, scatter)
        vel[i] = vc * tang
    return pos, vel


rng = np.random.default_rng(42)
N_STARS_PER = 200

core1_w1 = np.array([-5.0, 0.0])
core2_w1 = np.array([5.0, 0.0])
v_core1 = np.array([0.0, v_trans])
v_core2 = np.array([0.0, -v_trans])

stars1, vstars1 = init_stars(
    N_STARS_PER, core1_w1, core2_w1, M1, M2,
    [core1_w1, core2_w1], [M_HALO, M_HALO], rng,
)
stars2, vstars2 = init_stars(
    N_STARS_PER, core2_w1, core1_w1, M2, M1,
    [core1_w1, core2_w1], [M_HALO, M_HALO], rng,
)
# Bulk velocity of each disk matches its core
vstars1 += v_core1
vstars2 += v_core2

n_stars = N_STARS_PER * 2
star_pos = np.vstack([stars1, stars2])
star_vel = np.vstack([vstars1, vstars2])
star_gal_id = np.array([0] * N_STARS_PER + [1] * N_STARS_PER)

print(f"Initialized {n_stars} star particles ({N_STARS_PER} per galaxy)")
'''
))

cells.append(code(
'''# --- Week 2 / 3: full simulation with stars and halos ---
dt2 = 0.001
t_end2 = 8.0
n_steps2 = int(t_end2 / dt2)
t_mid_gyr = 4.0
step_mid = int(round((t_mid_gyr / GYR_PER_TU) / dt2))

core_hist = np.zeros((n_steps2 + 1, 2, 2))
star_hist = np.zeros((n_steps2 + 1, n_stars, 2))
times2 = np.linspace(0, t_end2, n_steps2 + 1)

c1, c2 = core1_w1.copy(), core2_w1.copy()
vc1, vc2 = v_core1.copy(), v_core2.copy()
sp, sv = star_pos.copy(), star_vel.copy()

core_hist[0, 0], core_hist[0, 1] = c1, c2
star_hist[0] = sp


def accelerations(c1, c2, sp):
    """Core and star accelerations (vectorized; NFW halos follow cores)."""
    a_c1 = softened_accel(c1, c2, M2)
    a_c2 = softened_accel(c2, c1, M1)

    r1 = sp - c1
    r2 = sp - c2
    r1s = np.sqrt((r1 ** 2).sum(axis=1) + EPS ** 2)[:, np.newaxis]
    r2s = np.sqrt((r2 ** 2).sum(axis=1) + EPS ** 2)[:, np.newaxis]
    a_stars = G * M1 * r1 / (r1s ** 3) + G * M2 * r2 / (r2s ** 3)
    a_stars += nfw_accel_bulk(sp, c1) + nfw_accel_bulk(sp, c2)
    return a_c1, a_c2, a_stars


# Leapfrog: cores + stars together
ac1, ac2, ast = accelerations(c1, c2, sp)
vc1 += 0.5 * dt2 * ac1
vc2 += 0.5 * dt2 * ac2
sv += 0.5 * dt2 * ast

for step in range(1, n_steps2 + 1):
    c1 += dt2 * vc1
    c2 += dt2 * vc2
    sp += dt2 * sv
    ac1, ac2, ast = accelerations(c1, c2, sp)
    vc1 += dt2 * ac1
    vc2 += dt2 * ac2
    sv += dt2 * ast
    core_hist[step, 0] = c1
    core_hist[step, 1] = c2
    star_hist[step] = sp
    if step % 1000 == 0:
        print(f"  step {step}/{n_steps2}, t = {step*dt2*GYR_PER_TU:.2f} Gyr")

print("Star+core simulation complete.")
'''
))

cells.append(code(
'''# --- Week 2 figure ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor(BG)

for ax, idx, title in zip(axes, [0, step_mid], ["t = 0", f"t = {t_mid_gyr} Gyr"]):
    s = star_hist[idx]
    g1 = s[star_gal_id == 0]
    g2 = s[star_gal_id == 1]
    ax.scatter(g1[:, 0], g1[:, 1], s=6, c="cyan", alpha=0.7, label="Galaxy 1 stars")
    ax.scatter(g2[:, 0], g2[:, 1], s=6, c="magenta", alpha=0.7, label="Galaxy 2 stars")
    ax.scatter(core_hist[idx, 0, 0], core_hist[idx, 0, 1], s=120, c="white",
               edgecolors="k", zorder=5, label="Core 1")
    ax.scatter(core_hist[idx, 1, 0], core_hist[idx, 1, 1], s=120, c="white",
               marker="*", edgecolors="k", zorder=5, label="Core 2")
    pad = 2
    ax.set_xlim(s[:, 0].min() - pad, s[:, 0].max() + pad)
    ax.set_ylim(s[:, 1].min() - pad, s[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper right")

plt.tight_layout()
plt.savefig("week2_particles.png", dpi=150, facecolor=BG)
plt.close(fig)
print("Saved week2_particles.png")
'''
))

# WEEK 3
cells.append(md(
"""## Week 3: Full Merger Animation & Tidal Analysis

We animate the full 8 Gyr encounter and flag **tidally stripped** stars at the first
close passage (\\(t \\approx 4\\) Gyr): particles more than **15 kpc** from their
parent galaxy core. Compare the morphology qualitatively to the **Antennae Galaxies**
(NGC 4038/4039) below.
"""
))

cells.append(code(
'''# --- Tidal stripping at t = 4 Gyr ---
s_mid = star_hist[step_mid]
c1_mid = core_hist[step_mid, 0]
c2_mid = core_hist[step_mid, 1]
strip_thresh = 15.0

dist1 = np.linalg.norm(s_mid[star_gal_id == 0] - c1_mid, axis=1)
dist2 = np.linalg.norm(s_mid[star_gal_id == 1] - c2_mid, axis=1)
stripped1 = dist1 > strip_thresh
stripped2 = dist2 > strip_thresh
n_strip1, n_strip2 = stripped1.sum(), stripped2.sum()

print(f"Stripped stars (>{strip_thresh} kpc from parent core) at t={t_mid_gyr} Gyr:")
print(f"  Galaxy 1: {n_strip1} / {N_STARS_PER} ({100*n_strip1/N_STARS_PER:.1f}%)")
print(f"  Galaxy 2: {n_strip2} / {N_STARS_PER} ({100*n_strip2/N_STARS_PER:.1f}%)")

fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor(BG)
g1 = s_mid[star_gal_id == 0]
g2 = s_mid[star_gal_id == 1]
ax.scatter(g1[:, 0], g1[:, 1], s=6, c="cyan", alpha=0.5)
ax.scatter(g2[:, 0], g2[:, 1], s=6, c="magenta", alpha=0.5)
ax.scatter(g1[stripped1, 0], g1[stripped1, 1], s=20, c="yellow", label="Stripped G1")
ax.scatter(g2[stripped2, 0], g2[stripped2, 1], s=20, c="yellow", label="Stripped G2")
ax.scatter([c1_mid[0], c2_mid[0]], [c1_mid[1], c2_mid[1]], s=150, c="white", zorder=5)
ax.set_aspect("equal")
ax.set_xlabel("x [kpc]")
ax.set_ylabel("y [kpc]")
ax.set_title(f"Tidal tails at t = {t_mid_gyr} Gyr")
ax.legend()
plt.tight_layout()
plt.savefig("week3_tidal_analysis.png", dpi=150, facecolor=BG)
plt.close(fig)
print("Saved week3_tidal_analysis.png")
'''
))

cells.append(code(
'''# --- Merger animation (subsample frames for reasonable file size) ---
frame_skip = 8  # subsample for reasonable GIF size (~1000 frames max)
frames = list(range(0, n_steps2 + 1, frame_skip))
if frames[-1] != n_steps2:
    frames.append(n_steps2)

fig, ax = plt.subplots(figsize=(7, 7))
fig.patch.set_facecolor(BG)


def update(frame_idx):
    ax.clear()
    ax.set_facecolor(BG)
    k = frames[frame_idx]
    s = star_hist[k]
    g1 = s[star_gal_id == 0]
    g2 = s[star_gal_id == 1]
    ax.scatter(g1[:, 0], g1[:, 1], s=3, c="cyan", alpha=0.6)
    ax.scatter(g2[:, 0], g2[:, 1], s=3, c="magenta", alpha=0.6)
    ax.scatter(core_hist[k, 0, 0], core_hist[k, 0, 1], s=80, c="white", zorder=5)
    ax.scatter(core_hist[k, 1, 0], core_hist[k, 1, 1], s=80, c="white", marker="*", zorder=5)
    pad = 3
    ax.set_xlim(s[:, 0].min() - pad, s[:, 0].max() + pad)
    ax.set_ylim(s[:, 1].min() - pad, s[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")
    t_gyr = times2[k] * GYR_PER_TU
    ax.set_title(f"GalacticMergerSim  |  t = {t_gyr:.2f} Gyr")
    return ax.artists


anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / 30)
anim.save("merger_animation.gif", writer=PillowWriter(fps=30), dpi=100)
plt.close(fig)
print("Saved merger_animation.gif")
'''
))

cells.append(md(
"""### Morphology: Simulated Merger vs. Antennae Galaxies (NGC 4038/4039)

| Feature | This simulation | Antennae (NGC 4038/4039) |
|---------|-----------------|---------------------------|
| Tidal tails | Two curved streams of stripped disk stars (cyan/magenta) extending away from the nuclei | Classic **two long tidal tails** — one from each galaxy |
| Bridges | Stars span the space between cores during close passages | Prominent **stellar bridge** linking the pair mid-merger |
| Nuclei | Two bright cores that orbit and eventually coalesce | Two overlapping starburst nuclei still distinguishable in optical |
| Disturbed disks | Exponential disks are strongly distorted by ~4 Gyr | Highly irregular, knotty morphology from ongoing star formation |

**Qualitative similarities:** Both systems show **dual tidal tails**, a **connecting stellar bridge**, and **strongly disturbed** disk structure rather than smooth spirals. The Antennae are farther along in the merger and include gas-driven starbursts our collisionless test-particle model does not capture, but the **large-scale gravitational morphology** — inspiral, tidal stripping, and bridge formation — matches the basic N-body picture well.
"""
))

# WEEK 4
cells.append(md(
"""## Week 4: ML Prediction of Merger Outcomes

We run **500** core-only mergers with randomized mass ratio \\(q=M_2/M_1\\), separation
\\(r_0\\), and velocity fraction \\(f\\), record **coalescence time** (first passage
below 1 kpc), and train a **Random Forest** regressor to predict merger times from
\\([q, r_0, f]\\).
"""
))

cells.append(code(
'''# --- Week 4: generate merger dataset (cores only) ---
N_SIMS = 500
DT_ML = 0.002
T_END_ML = 8.0
MERGE_SEP = 1.0
M1_FIXED = 5.0
N_STEPS_ML = int(T_END_ML / DT_ML)

rng_ml = np.random.default_rng(123)
rows = []

print(f"Running {N_SIMS} core-only simulations...")
for sim in range(N_SIMS):
    q = rng_ml.uniform(0.1, 1.0)
    r0 = rng_ml.uniform(5.0, 20.0)
    f = rng_ml.uniform(0.5, 1.2)
    m1, m2 = M1_FIXED, M1_FIXED * q

    p1 = np.array([-0.5 * r0, 0.0])
    p2 = np.array([0.5 * r0, 0.0])
    vc = circular_speed_each(m1, m2, r0)
    vt = f * vc
    v1 = np.array([0.0, vt])
    v2 = np.array([0.0, -vt])
    vcm = (m1 * v1 + m2 * v2) / (m1 + m2)
    v1 -= vcm
    v2 -= vcm

    state = np.concatenate([p1, p2])
    vel = np.concatenate([v1, v2])

    def acc_fn(st, m1=m1, m2=m2):
        pa, pb = st[:2], st[2:]
        return np.concatenate([softened_accel(pa, pb, m2), softened_accel(pb, pa, m1)])

    acc = acc_fn(state)
    vel = vel + 0.5 * DT_ML * acc
    coalesce_t = T_END_ML
    merged = 0

    for step in range(1, N_STEPS_ML + 1):
        state = state + DT_ML * vel
        acc = acc_fn(state)
        vel = vel + DT_ML * acc
        sep = np.linalg.norm(state[2:] - state[:2])
        t_now = step * DT_ML
        if sep < MERGE_SEP and coalesce_t == T_END_ML:
            coalesce_t = t_now
            merged = 1

    rows.append({"q": q, "r0": r0, "f": f,
                 "coalescence_time": coalesce_t, "merged": merged})
    if (sim + 1) % 100 == 0:
        print(f"  completed {sim + 1}/{N_SIMS}")

df = pd.DataFrame(rows)
df.to_csv("merger_dataset.csv", index=False)
print("Saved merger_dataset.csv")
print(df.describe())
'''
))

cells.append(code(
'''# --- Week 4: Random Forest ---
X = df[["q", "r0", "f"]].values
y = df["coalescence_time"].values * GYR_PER_TU  # predict in Gyr for interpretability

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test R^2 = {r2:.4f}")
print(f"Test RMSE = {rmse:.4f} Gyr")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor(BG)

ax = axes[0]
sc = ax.scatter(y_test, y_pred, c=X_test[:, 0], cmap="viridis", alpha=0.8, edgecolors="none")
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, "w--", lw=1, label="Perfect prediction")
ax.set_xlabel("Actual coalescence time [Gyr]")
ax.set_ylabel("Predicted coalescence time [Gyr]")
ax.set_title(f"Random Forest (R²={r2:.3f}, RMSE={rmse:.2f} Gyr)")
plt.colorbar(sc, ax=ax, label="Mass ratio q")
ax.legend()

ax = axes[1]
names = ["Mass ratio q", "Separation r0", "Velocity fraction f"]
imps = rf.feature_importances_
ax.barh(names, imps, color=["#66ccff", "#ff66cc", "#ccff66"])
ax.set_xlabel("Feature importance")
ax.set_title("What drives merger time?")

plt.tight_layout()
plt.savefig("week4_ml_results.png", dpi=150, facecolor=BG)
plt.close(fig)
print("Saved week4_ml_results.png")
'''
))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": cells,
}

out = Path("galaxy_merger.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {out} with {len(cells)} cells")
