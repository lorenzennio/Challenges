import math

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import cupy as cp

import numba.cuda
import numba as nb
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

from utils import (
    combine_uncertaintes,
    confidence_interval,
)


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
NUM_BLOCKS_X = 100
NUM_BLOCKS_Y = 100
CONFIDENCE_LEVEL = 0.05

width = np.float32(3 / (32 * NUM_BLOCKS_X))  # all cells have the same size
height = np.float32(1.5 / (32 * NUM_BLOCKS_Y))  # all cells have the same size


@nb.cuda.jit
def sample_mandelbrot_until(rng_states, numer, denom, uncertainty, uncertainty_target):
    i, j = nb.cuda.grid(2)
    rng_idx = (32 * NUM_BLOCKS_X) * i + j
    xmin = np.float32(-2) + width * j
    ymin = np.float32(0) + height * i

    uncertainty[i, j] = np.float32(np.inf)

    while uncertainty[i, j] > uncertainty_target:
        denom[i, j] += SAMPLES_IN_BATCH
        for _ in range(SAMPLES_IN_BATCH):
            x = xoroshiro128p_uniform_float32(rng_states, rng_idx) * width + xmin
            y = xoroshiro128p_uniform_float32(rng_states, rng_idx) * height + ymin
            if is_in_mandelbrot(x, y):
                numer[i, j] += 1

        uncertainty[i, j] = wald_uncertainty(numer[i, j], denom[i, j]) * width * height


rng_states = create_xoroshiro128p_states(32**2 * NUM_BLOCKS_X*NUM_BLOCKS_Y, seed=12345)

numer = cp.zeros((32 * NUM_BLOCKS_X, 32 * NUM_BLOCKS_Y), dtype=cp.int32)
denom = cp.zeros((32 * NUM_BLOCKS_X, 32 * NUM_BLOCKS_Y), dtype=cp.int32)
uncertainty = cp.zeros((32 * NUM_BLOCKS_X, 32 * NUM_BLOCKS_Y), dtype=cp.float32)

sample_mandelbrot_until[(NUM_BLOCKS_X, NUM_BLOCKS_Y), (32, 32)](
    rng_states, numer, denom, uncertainty, 1e-5
)

numer = numer.get()

denom = denom.get()

uncertainty = uncertainty.get()



final_value = 2*(np.sum((numer / denom)) * width * height).item()
print(f"\tThe total area of all tiles is {final_value}")

CONFIDENCE_LEVEL = 0.05

confidence_interval_low, confidence_interval_high = confidence_interval(
    CONFIDENCE_LEVEL, numer, denom, width * height
)

final_uncertainty = 2*combine_uncertaintes(
    confidence_interval_low, confidence_interval_high, denom
)
print(f"\tThe uncertainty on the total area is {final_uncertainty}\n")