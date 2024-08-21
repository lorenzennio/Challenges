import math

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import cupy as cp

import numba.cuda
import numba as nb
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

from utils import plot_pixels


@nb.cuda.jit(device=True)
def wald_uncertainty(numer, denom):
    if numer == 0:
        numer = 1
        denom += 1
    elif numer == denom:
        denom += 1

    frac = np.float32(numer) / np.float32(denom)

    return math.sqrt(frac * (1 - frac) / denom)


@nb.cuda.jit(device=True)
def is_in_mandelbrot(x, y):
    c = np.complex64(x) + np.complex64(y) * np.complex64(1j)
    z_hare = z_tortoise = np.complex64(0)
    while True:
        z_hare = z_hare * z_hare + c
        z_hare = z_hare * z_hare + c
        z_tortoise = z_tortoise * z_tortoise + c
        if z_hare == z_tortoise:
            return True  # orbiting or converging to zero
        if z_hare.real**2 + z_hare.imag**2 > 4:
            return False  # diverging to infinity


SAMPLES_IN_BATCH = 10
NUM_BLOCKS_1D = 100
CONFIDENCE_LEVEL = 0.05

width = height = np.float32(3 / (32 * NUM_BLOCKS_1D))  # all cells have the same size


@nb.cuda.jit
def sample_mandelbrot_until(rng_states, numer, denom, uncertainty, uncertainty_target):
    i, j = nb.cuda.grid(2)
    rng_idx = (32 * NUM_BLOCKS_1D) * i + j
    xmin = np.float32(-2) + width * j
    ymin = np.float32(-3 / 2) + height * i

    uncertainty[i, j] = np.float32(np.inf)

    while uncertainty[i, j] > uncertainty_target:
        denom[i, j] += SAMPLES_IN_BATCH
        for _ in range(SAMPLES_IN_BATCH):
            x = xoroshiro128p_uniform_float32(rng_states, rng_idx) * width + xmin
            y = xoroshiro128p_uniform_float32(rng_states, rng_idx) * height + ymin
            if is_in_mandelbrot(x, y):
                numer[i, j] += 1

        uncertainty[i, j] = wald_uncertainty(numer[i, j], denom[i, j]) * width * height


rng_states = create_xoroshiro128p_states(32**2 * NUM_BLOCKS_1D**2, seed=12345)

numer = cp.zeros((32 * NUM_BLOCKS_1D, 32 * NUM_BLOCKS_1D), dtype=cp.int32)
denom = cp.zeros((32 * NUM_BLOCKS_1D, 32 * NUM_BLOCKS_1D), dtype=cp.int32)
uncertainty = cp.zeros((32 * NUM_BLOCKS_1D, 32 * NUM_BLOCKS_1D), dtype=cp.float32)

sample_mandelbrot_until[(NUM_BLOCKS_1D, NUM_BLOCKS_1D), (32, 32)](
    rng_states, numer, denom, uncertainty, 1e-8
)