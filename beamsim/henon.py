"""
4D Henon map - basically the simplest model you can write down that still
has the physics that gives a ring a finite dynamic aperture: a linear
lattice (just a rotation) plus one sextupole kick to make it nonlinear.

one turn = rotate by 2*pi*Qx (x,px) and 2*pi*Qy (y,py), then kick:
    px -> px + k2 * (x^2 - y^2)
    py -> py - 2 * k2 * x * y

(these two come from k2*(x+iy)^2 - real/imag parts, standard sextupole
multipole expansion, not something I made up)

This is a pretty standard toy model in accelerator physics for playing
around with nonlinear dynamics without needing a full lattice code.
"""

import numpy as np


def track(x, px, y, py, qx, qy, k2, n_turns, lost_radius=10.0):
    """Push particles through n_turns of the map.

    x, px, y, py: arrays, one entry per particle.
    qx, qy, k2 can be scalars or arrays same shape as the particles -
    handy because it means you can track a bunch of different lattices
    at once instead of looping over them in python.

    Returns survived: how many turns each particle lasted before getting
    kicked out (or n_turns if it never did).
    """
    cx, sx = np.cos(2 * np.pi * qx), np.sin(2 * np.pi * qx)
    cy, sy = np.cos(2 * np.pi * qy), np.sin(2 * np.pi * qy)

    # copy so we don't clobber whatever arrays the caller passed in
    x, px, y, py = (np.array(v, dtype=float, copy=True) for v in (x, px, y, py))
    survived = np.zeros(x.shape, dtype=int)
    alive = np.ones(x.shape, dtype=bool)

    for _ in range(n_turns):
        # rotation - this is the "lattice" part, no nonlinearity yet
        x, px = cx * x + sx * px, -sx * x + cx * px
        y, py = cy * y + sy * py, -sy * y + cy * py
        # and here's the kick that actually makes this interesting
        px = px + k2 * (x * x - y * y)
        py = py - 2.0 * k2 * x * y

        r2 = x * x + px * px + y * y + py * py
        alive &= r2 < lost_radius ** 2
        survived += alive
        if not alive.any():
            break  # everyone's dead, no point tracking further
        # lost particles blow up fast (amplitude explodes within a few
        # turns), so pin them to zero or they'll eventually overflow and
        # mess with np.where downstream
        x = np.where(alive, x, 0.0)
        px = np.where(alive, px, 0.0)
        y = np.where(alive, y, 0.0)
        py = np.where(alive, py, 0.0)

    return survived


def dynamic_aperture(qx, qy, k2, n_turns=1000, n_angles=5,
                     r_max=1.5, bisection_steps=12):
    """Find the dynamic aperture for one or many lattices.

    DA here = largest launch amplitude r (particle starts at
    x=r*cos(theta), y=r*sin(theta), px=py=0) that still survives
    n_turns, minimised over a handful of angles theta. Found by
    bisecting r between 0 and r_max rather than scanning - way fewer
    track() calls needed.

    qx, qy, k2 can be arrays (one config per entry) or scalars.
    Returns an array of DA values, one per config.
    """
    qx, qy, k2 = np.broadcast_arrays(
        np.atleast_1d(np.asarray(qx, float)),
        np.atleast_1d(np.asarray(qy, float)),
        np.atleast_1d(np.asarray(k2, float)),
    )
    n_cfg = qx.shape[0]
    # avoid theta = 0 or pi/2 exactly, those are degenerate (pure x or pure y)
    thetas = np.linspace(0.05, np.pi / 2 - 0.05, n_angles)

    da_per_angle = np.empty((n_angles, n_cfg))
    for i, th in enumerate(thetas):
        lo = np.zeros(n_cfg)       # known stable
        hi = np.full(n_cfg, r_max)  # assumed unstable
        for _ in range(bisection_steps):
            mid = 0.5 * (lo + hi)
            surv = track(mid * np.cos(th), np.zeros(n_cfg),
                         mid * np.sin(th), np.zeros(n_cfg),
                         qx, qy, k2, n_turns)
            stable = surv >= n_turns
            lo = np.where(stable, mid, lo)
            hi = np.where(stable, hi, mid)
        da_per_angle[i] = lo  # after enough steps lo ~= the true boundary

    # worst-case angle sets the aperture, not the average
    return da_per_angle.min(axis=0)

