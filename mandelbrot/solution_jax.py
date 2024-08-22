#!/usr/bin/env -S submit -M 2000 -m 2000 -f python -u

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import jit, lax, random
from jax import device_put
import time

start_time = time.time()

# Get the first GPU device
gpu_device = jax.devices('gpu')[0]

import numpy as np

from functools import partial
from jax.scipy.special import betainc

# Importing the utility functions
from utils import (
    combine_uncertaintes,
    plot_pixels,
    confidence_interval,
    #wald_uncertainty,
)
jax.config.update("jax_enable_x64", True)

@jax.jit
def wald_uncertainty(numer, denom):
    """Wald approximation on the uncertainty of the tile using JAX."""
    
    numer = lax.cond(numer == 0, lambda _: 1, lambda _: numer, None)
    denom = lax.cond(numer == denom, lambda _: denom + 1, lambda _: denom, None)

    frac = numer / denom

    return jnp.sqrt(frac * (1 - frac) / denom)

#
CONFIDENCE_LEVEL = 0.05
# Constants
NUM_TILES_1D = 100
SAMPLES_IN_BATCH = 1000000
width = 3 / NUM_TILES_1D
height = 3 / NUM_TILES_1D

# JAX-compatible Mandelbrot set check
@jit
def is_in_mandelbrot(x, y, max_iterations=1000):
    c = x + y * 1j
    z_hare = z_tortoise = 0 + 0j
    iter_count = 0

    def mandelbrot_step(state):
        z_hare, z_tortoise, iter_count = state
        z_hare = z_hare * z_hare + c
        z_hare = z_hare * z_hare + c  # Hare does one more step
        z_tortoise = z_tortoise * z_tortoise + c
        return z_hare, z_tortoise, iter_count + 1

    def mandelbrot_cond(state):
        z_hare, z_tortoise, iter_count = state
        return ~((jnp.real(z_hare)**2 + jnp.imag(z_hare)**2 > 4) | (iter_count >= max_iterations))

    final_state = lax.while_loop(mandelbrot_cond, mandelbrot_step, (z_hare, z_tortoise, iter_count))
    z_hare, z_tortoise, _ = final_state

    return jnp.real(z_hare)**2 + jnp.imag(z_hare)**2 <= 4

# JAX-compatible function for drawing the Mandelbrot set

# Counting points in Mandelbrot
@partial(jit, static_argnames=['num_samples'])
def count_mandelbrot(rng_key, num_samples, xmin, width, ymin, height):
    samples = random.uniform(rng_key, (num_samples, 2))  # Shape is static now
    x = xmin + samples[:, 0] * width
    y = ymin + samples[:, 1] * height
    return jnp.sum(jax.vmap(is_in_mandelbrot)(x, y))

# Estimating area
xmin, xmax = -2, 1
ymin, ymax = 0, 3 / 2
rng_key = random.PRNGKey(0)

@jax.jit
def tile_xmin(j):
    return -2 + width * j

@jax.jit
def tile_ymin(i):
    return 0 + height * i

# Compute Mandelbrot area per tile in parallel with JAX
@jax.jit
def compute_until(rngs, numer, denom, uncert, uncert_target):
    def process_tile(idx, state):
        numer, denom, uncert = state
        i, j = divmod(idx, NUM_TILES_1D)
        rng = rngs[NUM_TILES_1D * i + j]

        def cond(state):
            numer_ij, denom_ij, uncert_ij, _ = state
            return uncert_ij > uncert_target

        def body(state):
            numer_ij, denom_ij, uncert_ij, rng = state
            denom_ij += SAMPLES_IN_BATCH
            numer_ij += count_mandelbrot(
                rng, SAMPLES_IN_BATCH, tile_xmin(j), width, tile_ymin(i), height
            )
            uncert_ij = (
                wald_uncertainty(numer_ij, denom_ij) * width * height
            )
            return numer_ij, denom_ij, uncert_ij, rng

        numer_ij, denom_ij, uncert_ij, _ = lax.while_loop(
            cond, body, (numer[i, j], denom[i, j], jnp.inf, rng)
        )

        # Update the arrays
        numer = numer.at[i, j].set(numer_ij)
        denom = denom.at[i, j].set(denom_ij)
        uncert = uncert.at[i, j].set(uncert_ij)
        
        return numer, denom, uncert

    # Run the loop over all tiles
    numer, denom, uncert = jax.lax.fori_loop(
        0, NUM_TILES_1D * NUM_TILES_1D, process_tile, (numer, denom, uncert)
    )

    return numer, denom, uncert

# Set up arrays and compute until uncertainty target is reached
rngs = random.split(rng_key, NUM_TILES_1D * NUM_TILES_1D)
numer = jnp.zeros((NUM_TILES_1D, NUM_TILES_1D), dtype=jnp.int64)
denom = jnp.zeros((NUM_TILES_1D, NUM_TILES_1D), dtype=jnp.int64)
uncert = jnp.full((NUM_TILES_1D, NUM_TILES_1D), jnp.inf)

print("########################################################")
print("Compute Mandelbrot area per tile until target uncertainty is reached")
print("########################################################")
numer, denom, uncert = compute_until(rngs, numer, denom, uncert, 1e-5)

# Calculating the final area and uncertainty
final_value = (jnp.sum((numer / denom)) * width * height * 2).item()
print(f"\tThe total area of all tiles is {final_value}")

confidence_interval_low, confidence_interval_high = confidence_interval(
    CONFIDENCE_LEVEL, numer, denom, width * height
)

final_uncertainty = 2*combine_uncertaintes(
    confidence_interval_low, confidence_interval_high, denom
)
print(f"\tThe uncertainty on the total area is {final_uncertainty}\n")

print("########################################################")
print("TASK: Implement it on GPUs and break the world record!")
print("########################################################")

end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")