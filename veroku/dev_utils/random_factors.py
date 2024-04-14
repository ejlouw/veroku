import numpy as np

from veroku.factors.experimental.gaussian_mixture import GaussianMixture
from veroku.factors.gaussian import Gaussian


def get_random_gaussian(cov_coeff, mean_coeff=1.0, seed=None):
    """
    A test helper function that generates random Gaussian factors.

    :param cov_coeff: The scale coefficient for the uniform distribution that the variance parameter is drawn from.
    :param mean_coeff:  The scale coefficient for the uniform distribution that the mean parameter is drawn from.
    :return: a random Gaussian factor
    """
    if seed is not None:
        np.random.seed(seed)
    cov = np.random.rand() * cov_coeff
    mean = np.random.rand() * mean_coeff
    weight = np.random.rand()
    random_gaussian = Gaussian(cov=cov, mean=mean, log_weight=np.log(weight), var_names=["a"])
    return random_gaussian


def get_random_gaussian_mixture(cov_coeff=1.0, mean_coeff=1.0, num_components=3, seed=0):
    """
    A test helper function that generates random Gaussian factors.

    :param n: the number of components.
    :param cov_coeff: The scale coefficient for the uniform distribution that the variance parameter is drawn from.
    :param mean_coeff:  The scale coefficient for the uniform distribution that the mean parameter is drawn from.
    :return: a random Gaussian factor
    """
    assert seed >= 0
    random_gaussians = []
    for i in range(num_components):
        comp_seed = (seed + 1) * i
        random_gaussians.append(get_random_gaussian(cov_coeff, mean_coeff, seed=comp_seed))
    return GaussianMixture(random_gaussians)
