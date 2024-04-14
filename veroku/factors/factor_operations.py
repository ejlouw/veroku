#def general_multiplication(factor_a, factor_b):
#    if isinstance(factor_a
import numpy as np

from veroku.factors.normal_gamma import NormalGamma, normal_gamma_log_norm_constant


def normal_gamma_prior_batch_data_update(ng_prior, data):
    """
    Update this (prior) distribution and get the posterior given observed data.
    :param data:
    :return:
    """
    if len(data) == 0:
        return NormalGamma(mu_0=ng_prior.mu_0,
                           kappa_0=ng_prior.kappa_0,
                           alpha=ng_prior.alpha,
                           beta=ng_prior.beta,
                           log_weight=ng_prior.log_weight)
    if isinstance(data, np.ndarray):
        assert len(data.shape) == 1
    # ref: https://www.cs.ubc.ca/~murphyk/Teaching/CS340-Fall07/reading/NG.pdf (eqs: 21-24)
    # confirmed here as well: https://en.wikipedia.org/wiki/Normal-gamma_distribution
    n = float(len(data))
    x = data
    mu_0, kappa_0, alpha_0, beta_0 = ng_prior.mu_0, ng_prior.kappa_0, ng_prior.alpha, ng_prior.beta

    data_data_mean_diffs = np.square(x - np.mean(x))
    beta_n_term_0 = beta_0
    beta_n_term_1 = 0.5*np.sum(data_data_mean_diffs)
    beta_n_term_2 = (kappa_0 * n * np.square(np.mean(x) - mu_0)) / (2. * (kappa_0 + n))

    mu_n = (kappa_0 * mu_0 + n * np.mean(x)) / (kappa_0 + n)
    kappa_n = kappa_0 + n
    alpha_n = alpha_0 + n / 2.
    beta_n = beta_n_term_0 + beta_n_term_1 + beta_n_term_2
    updated_log_norm_constant = normal_gamma_log_norm_constant(alpha=alpha_n, beta=beta_n, kappa0=kappa_n)
    half_n_log_2pi = (n / 2) * np.log(2 * np.pi)
    log_weight = updated_log_norm_constant + ng_prior.log_weight_over_norm_const - half_n_log_2pi
    return NormalGamma(mu_0=mu_n,
                       kappa_0=kappa_n,
                       alpha=alpha_n,
                       beta=beta_n,
                       log_weight=log_weight,
                       var_names=ng_prior.var_names)