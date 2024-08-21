#!/usr/bin/env -S submit -M 2000 -m 2000 -f python -u

# This script is based on a Jupyter notebook provided by Jim Pivarski
# You can find it on GitHub: https://github.com/ErUM-Data-Hub/Challenges/blob/computing_challenge/computing/challenge.ipynb
# The markdown cells have been converted to raw comments
# and some of the LaTeX syntax has been removed for readability

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numba as nb

from utils import (
    combine_uncertaintes,
    plot_pixels,
    confidence_interval,
    wald_uncertainty,
)

# ignore deprecation warnings from numba for now
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)

@nb.jit
def is_in_mandelbrot(x, y):
    """Toirtoise and Hare approach to check if point (x,y) is in Mandelbrot set."""
    c = np.complex64(x) + np.complex64(y) * np.complex64(1j)
    z_hare = z_tortoise = np.complex64(0)  # tortoise and hare start at same point
    while True:
        z_hare = z_hare * z_hare + c
        z_hare = (
            z_hare * z_hare + c
        )  # hare does one step more to get ahead of the tortoise
        z_tortoise = z_tortoise * z_tortoise + c  # tortoise is one step behind
        if z_hare == z_tortoise:
            return True  # orbiting or converging to zero
        if z_hare.real**2 + z_hare.imag**2 > 4:
            return False  # diverging to infinity

@nb.jit
def count_mandelbrot(rng, num_samples, xmin, width, ymin, height):
    """Draw num_samples random numbers uniformly between (xmin, xmin+width)
    and (ymin, ymin+height).
    Raise `out` by one if the number is part of the Mandelbrot set.
    """
    out = np.int32(0)
    for x_norm, y_norm in rng.random((num_samples, 2), np.float32):
        x = xmin + (x_norm * width)
        y = ymin + (y_norm * height)
        out += is_in_mandelbrot(x, y)
    return out


"""
Do it inside Knill limits.
"""

# Knill limits
xmin, xmax = -2, 1
ymin, ymax = -3 / 2, 3 / 2

rng = np.random.default_rng()  # can be forked to run multiple rngs in parallel
NUM_TILES_1D = 100

numer = np.zeros((NUM_TILES_1D, NUM_TILES_1D), dtype=np.int64)
denom = np.zeros((NUM_TILES_1D, NUM_TILES_1D), dtype=np.int64)

"""
The width and height of each tile is the same:
"""

width = 3 / NUM_TILES_1D
height = 3 / NUM_TILES_1D


"""
But each tile has a different `xmin` and `ymin`.
"""

@nb.jit
def xmin(j):
    """xmin of tile in column j"""
    return -2 + width * j


@nb.jit
def ymin(i):
    """ymin of tile in row i"""
    return -3 / 2 + height * i

rngs = rng.spawn(NUM_TILES_1D * NUM_TILES_1D)

SAMPLES_IN_BATCH = 100


print("########################################################")
print("Compute Mandelbrot area per tile until target uncertainty is reached")
print("########################################################")


@nb.jit(parallel=True)
def compute_until(rngs, numer, denom, uncert, uncert_target):
    """Compute area of each tile until uncert_target is reached.
    The uncertainty is calculate with the Wald approximation in each tile.
    """
    for i in nb.prange(NUM_TILES_1D):
        for j in nb.prange(NUM_TILES_1D):
            rng = rngs[NUM_TILES_1D * i + j]

            uncert[i, j] = np.inf

            # Sample SAMPLES_IN_BATCH more points until uncert_target is reached
            while uncert[i, j] > uncert_target:
                denom[i, j] += SAMPLES_IN_BATCH
                numer[i, j] += count_mandelbrot(
                    rng, SAMPLES_IN_BATCH, xmin(j), width, ymin(i), height
                )

                uncert[i, j] = (
                    wald_uncertainty(numer[i, j], denom[i, j]) * width * height
                )


numer = np.zeros((NUM_TILES_1D, NUM_TILES_1D), dtype=np.int64)
denom = np.zeros((NUM_TILES_1D, NUM_TILES_1D), dtype=np.int64)
uncert = np.zeros((NUM_TILES_1D, NUM_TILES_1D), dtype=np.float64)

compute_until(rngs, numer, denom, uncert, 1e-5)

final_value = (np.sum((numer / denom)) * width * height).item()
print(f"\tThe total area of all tiles is {final_value}")


"""
We can use full, high-precision confidence intervals in the final result.
The implementation of this can be seen in `utils.py`.
"""

CONFIDENCE_LEVEL = 0.05

confidence_interval_low, confidence_interval_high = confidence_interval(
    CONFIDENCE_LEVEL, numer, denom, width * height
)

final_uncertainty = combine_uncertaintes(
    confidence_interval_low, confidence_interval_high, denom
)
print(f"\tThe uncertainty on the total area is {final_uncertainty}\n")


"""
Your task is to implement this on GPUs and scale it to many computers.

Your result, at the end of this exercise, may be the world's most precise estimate of a fundamental mathematical quantity.
"""
print("########################################################")
print("TASK: Implement it on GPUs and break the world record!")
print("########################################################")
