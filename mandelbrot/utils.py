# Some utility functions for the challenge
import matplotlib.pyplot as plt
from scipy.stats import beta
import numpy as np
import numba as nb


def plot_pixels(pixels, figsize=(7, 7), dpi=300, extend=[-2, 1, -3 / 2, 3 / 2]):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, layout="constrained")
    p = ax.imshow(pixels, extent=extend)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    return fig, ax, p


def confidence_interval(confidence_level, numerator, denominator, area):
    """Calculate confidence interval based on Clopper-Pearson.
    `beta.ppf` is the Percent Point function of the Beta distribution.
    Check out
    https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Clopper%E2%80%93Pearson_interval
    """
    # low, high = (
    #     beta.ppf(
    #         [confidence_level / 2, 1 - confidence_level / 2],
    #         [numerator, numerator + 1],
    #         [denominator - numerator + 1, denominator - numerator],
    #     )
    #     * area
    # )

    low = (
        np.nan_to_num(
            beta.ppf(confidence_level / 2, numerator, denominator - numerator + 1),
            nan=0,
        )
        * area
    )
    high = (
        np.nan_to_num(
            beta.ppf(1 - confidence_level / 2, numerator + 1, denominator - numerator),
            nan=1,
        )
        * area
    )
    # catch nan cases
    low = np.nan_to_num(np.asarray(low), nan=0)
    high = np.nan_to_num(np.asarray(high), nan=area)

    return low, high


r"""
(Disclaimer: It's to complicated to remove the LaTeX code here, put it in here (https://latexeditor.lagrida.com) to display.)
Wald approximation:

$$ \mbox{uncertainty} \approx \left\{\begin{array}{c l}
\displaystyle\sqrt{\frac{\frac{n + 1}{d + 1} \left(1 - \frac{n + 1}{d + 1}\right)}{d + 1}} & \mbox{if } n = 0 \\
\displaystyle\sqrt{\frac{\frac{n}{d + 1} \left(1 - \frac{n}{d + 1}\right)}{d + 1}} & \mbox{if } n = d \\
\displaystyle\sqrt{\frac{\frac{n}{d} \left(1 - \frac{n}{d}\right)}{d}} & \mbox{otherwise} \\
\end{array}\right. $$

where n is `numer` and d is `denom`. (This prevents the uncertainty from being zero if n = 0 or n = d by imagining that if we had taken one more sample, it would have broken the perfect streak. This is ad-hoc, but it's the right scale, which is what we need to know to decide whether more samples are needed.)

"""


@nb.jit
def wald_uncertainty(numer, denom):
    """Wald approximation on the uncertainty of the tile."""
    if numer == 0:
        numer = 1
        denom += 1
    elif numer == denom:
        denom += 1

    frac = numer / denom

    return np.sqrt(frac * (1 - frac) / denom)


def combine_uncertaintes(
    confidence_interval_low, confidence_interval_high, denominator
):
    """
    See the section on stratified sampling in http://www.ff.bg.ac.rs/Katedre/Nuklearna/SiteNuklearna/bookcpdf/c7-8.pdf
    for how to combine uncertainties in each cell into a total uncertainty.
    """
    final_uncertainty = (
        np.sum(confidence_interval_high - confidence_interval_low)
        / np.sqrt(4 * np.sum(denominator))
    ).item()

    return final_uncertainty
