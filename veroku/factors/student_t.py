import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.special import gamma

from veroku.factors._factor import Factor


class StudentT(Factor):
    def __init__(self, mu, v, sigma):
        self.mu = mu
        self.v = v
        self.sigma = sigma

    def potential(self, x):
        # ref: https://www.cs.ubc.ca/~murphyk/Teaching/CS340-Fall07/reading/NG.pdf
        #p = np.power(1 + (1/self.v)*np.square((x-self.mu)/self.sigma), -(self.v+1)/2)
        #c = (gamma(self.v/2 + 0.5)/gamma(self.v/2))/(np.sqrt(self.v*np.pi*self.sigma))
        return np.exp(self.log_potential(x))

    def log_potential(self, x):
        log_p = (-(self.v + 1) / 2) * np.log(1 + (1 / self.v) * np.square((x - self.mu) / self.sigma))
        log_c = np.log(gamma(self.v / 2 + 0.5)) - np.log(gamma(self.v / 2)) - np.log(np.sqrt(self.v * np.pi) * self.sigma)
        return log_c + log_p

    def sample(self, n):
        samples = scipy.stats.t.rvs(loc=self.mu, df=self.v, scale=self.sigma, size=n)
        return samples

    def __repr__(self):
        return f"\nmu = {self.mu:.3f}, \nv = {self.v:.3f}, \nsigma = {self.sigma:.3f}"

    @property
    def variance(self):
        assert self.v > 2
        return (self.v*self.sigma**2)/(self.v - 2)

    def get_standard_limits(self):
        limits = [self.mu - 5*np.sqrt(self.variance), self.mu + 5*np.sqrt(self.variance)]
        return limits

    def plot(self, limits=None, ax=None, *args, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        if limits is None:
            limits = self.get_standard_limits()
        x = np.linspace(*limits, 100)
        ax.plot(x, self.potential(x), *args, **kwargs)



