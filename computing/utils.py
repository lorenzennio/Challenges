# Some utility functions for the challenge
import matplotlib.pyplot as plt


def plot_pixels(pixels, figsize=(7, 7), dpi=300, extend=[-2, 1, -3 / 2, 3 / 2]):
    fig, ax = plt.subplots(figsize=figsize, dpi=300, layout="constrained")
    p = ax.imshow(pixels, extent=extend)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    return fig, ax, p
