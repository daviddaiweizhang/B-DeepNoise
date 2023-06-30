import numpy as np
from scipy import stats
import tensorflow_probability as tfp
tfd = tfp.distributions


def normal_inv_sf(x, mean, scale):
    return stats.norm.isf(x, loc=mean, scale=scale)


def normal_inv_cdf(x, mean, scale):
    return stats.norm.ppf(x, loc=mean, scale=scale)


def normal_log_den(x, mean=0.0, scale=1.0):
    return (
            (-0.5) * np.log(2 * np.pi * scale**2)
            + (-0.5) * (x - mean)**2 / scale**2)


def normal_cdf_approx(x, mean=0.0, scale=1.0, log=False):
    """ Reference:
        'A logistic approximation to the cumulative normal distribution'
        by Bowling et al. (2009)
    """
    a = 0.07056
    b = 1.5976
    z = (x - mean) / scale
    if log:
        u = -np.log1p(np.exp(-a*z*z*z - b*z))
    else:
        u = 1 / (1 + np.exp(-a*z*z*z - b*z))
    return u


def normal_inv_cdf_approx_lower(u):
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]
    t = np.sqrt(-2.0*np.log(u))
    z = (
            ((c[2]*t + c[1])*t + c[0])
            / (((d[2]*t + d[1])*t + d[0])*t + 1.0)
            - t)
    return z


def normal_inv_cdf_approx(u, mean=0.0, scale=1.0):
    """ Reference:
        1. https://www.johndcook.com/blog/normal_cdf_inverse/
        2. 'Handbook of Mathematical Functions'
           by Abramowitz and Stegun (1965),
           Formula 26.2.23.
    """
    z = (
            normal_inv_cdf_approx_lower(u) * (u < 0.5)
            - normal_inv_cdf_approx_lower(1-u) * (u >= 0.5))
    x = z * scale + mean
    return x


# TODO: improve efficiency by using sf only when cdf fails
def normal_log_prob_safe(
        lower, upper, mean=0.0, scale=1.0,
        return_lower=False, return_upper=False):
    """ Safe computation of log of normal CDF of an interval.
        Avoids numeric underflow.
        The returned value is
        `log(F(upper; mean, scale) - F(lower; mean, scale))`,
        where `F` is the standard normal CDF.
    """
    upper_standardized = (upper - mean) / scale
    lower_standardized = (lower - mean) / scale
    cdf_log_upper = stats.norm.logcdf(upper_standardized)
    sf_log_upper = stats.norm.logsf(upper_standardized)
    del upper_standardized
    cdf_log_lower = stats.norm.logcdf(lower_standardized)
    sf_log_lower = stats.norm.logsf(lower_standardized)
    del lower_standardized
    if not (
            (cdf_log_lower != cdf_log_upper)
            + (sf_log_lower != sf_log_upper)
            + (lower == upper)
            ).all():
        raise ValueError('CDF and SF underflow')

    prob_log_middle_cdf = cdf_log_upper + log1mexp(
            cdf_log_upper - cdf_log_lower)
    if not return_lower:
        del cdf_log_lower
    if not return_upper:
        del cdf_log_upper
    prob_log_middle_sf = sf_log_lower + log1mexp(
            sf_log_lower - sf_log_upper)
    if not return_lower:
        del sf_log_lower
    if not return_upper:
        del sf_log_upper
    # Underflow causes log_prob_cdf or log_prob_sf to be zero
    prob_log_middle = np.minimum(
            prob_log_middle_cdf, prob_log_middle_sf)
    prob_log_middle[lower == upper] = -np.inf

    if return_lower:
        prob_log_lower_cdf = cdf_log_lower
        prob_log_lower_sf = log1mexp(-sf_log_lower)
        log_prob_lower = np.minimum(
                prob_log_lower_cdf, prob_log_lower_sf)
        del cdf_log_lower, sf_log_lower, prob_log_lower_cdf, prob_log_lower_sf
    else:
        log_prob_lower = None
    if return_upper:
        prob_log_upper_sf = sf_log_upper
        prob_log_upper_cdf = log1mexp(-cdf_log_upper)
        log_prob_upper = np.minimum(
                prob_log_upper_cdf, prob_log_upper_sf)
        del cdf_log_upper, sf_log_upper, prob_log_upper_cdf, prob_log_upper_sf
    else:
        log_prob_upper = None
    return prob_log_middle, log_prob_lower, log_prob_upper


def log1pexp(x):
    """
    Computation of log(1 + exp(x)).

    Details can be found at:
    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    result = np.zeros_like(x)
    result[x <= -37] = np.exp(x[x <= -37])
    result[(x > -37) & (x <= 18)] = np.log1p(np.exp(x[(x > -37) & (x <= 18)]))
    result[(x > 18) & (x <= 33.3)] = x[(x > 18) & (x <= 33.3)] + \
        np.exp(-x[(x > 18) & (x <= 33.3)])
    result[x > 33.3] = x[x > 33.3]

    return result


def log1mexp(x):
    """
    Computation of log(1 - exp(-x)).

    Details can be found at:
    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    if np.any(np.exp(-x) > 1.):
        raise ValueError("Invalid input x: exists element such that "
                         "1 - exp(-x) is negative.")
    result = np.zeros_like(x)
    result[(x > 0) & (x <= 0.693)
           ] = np.log(-(np.expm1(-x[(x > 0) & (x <= 0.693)])))
    result[x > 0.693] = np.log1p(-np.exp(-x[x > 0.693]))
    return result


# TODO: replace with plain computation
def invgamma_log_den(x, concentration, scale):
    dist = tfd.InverseGamma(concentration, scale)
    pdf = dist.log_prob(x).numpy()
    return pdf
