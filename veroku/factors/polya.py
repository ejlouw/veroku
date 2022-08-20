from veroku.factors._factor import Factor
import numpy as np

from veroku.factors._factor_utils import make_column_vector
from veroku.factors.dirichlet import Dirichlet
from veroku.factors.categorical import Categorical
from scipy.special import gamma
import copy

#TODO: change this to be a factorised factor type distribution.


class Polya(Factor):
    def __init__(self, alphas, p_var_names=None, a_var_names=None):
        if p_var_names is None:
            p_var_names = ["data"]
        if a_var_names is None:
            a_var_names = [f"alpha_{i}" for i in range(len(alphas))]
        assert len(p_var_names) == 1
        assert len(a_var_names) == len(alphas)
        self.p_var_names = p_var_names
        self.a_var_names = copy.deepcopy(a_var_names)
        super().__init__(var_names=p_var_names + a_var_names)
        self.alphas = make_column_vector(alphas)

    def __repr__(self):
        s = f"Polya Factor" \
            f"\nalphas: {np.round(self.alphas.ravel(), 3)}"
        return s

    def copy(self):
        return Polya(alphas=self.alphas,
                     p_var_name=self.p_var_names,
                     a_var_names=self.a_var_names)

    #TODO: consider removing - should rather marginalize and then sample
    def sample_categorical(self):
        d = Dirichlet(self.alphas)
        probs = d.sample(num_samples=1).ravel()
        probs_table = {(assignment,): prob for assignment, prob in enumerate(probs)}
        cardinalities = [len(probs)]
        return Categorical(var_names=self.p_var_names, cardinalities=cardinalities,
                           probs_table=probs_table)

    # TODO: consider removing - one would expect to sample x and theta values here
    def sample(self, num_samples):
        data_samples = []
        for _ in range(num_samples):
            categorical = self.sample_categorical()
            data_sample = categorical.sample(1)
            data_samples.append(data_sample)
        col_vec_samples = np.array(data_samples).T
        return col_vec_samples

    def reduce(self, vrs, values):
        # TODO: handle different variable orderings
        assert vrs == self.p_var_names
        assert len(values) == 1
        inc_alpha_index = values[0]
        alphas = self.alphas.copy()
        alphas[inc_alpha_index] += 1
        return Dirichlet(alphas=alphas, var_names=self.a_var_names)

    def absorb(self, factor):
        if not isinstance(factor, Dirichlet):
            raise NotImplementedError()
        assert factor.var_names == self.a_var_names

        result_alphas = self.alphas + factor.alphas - 1 # TODO: check this
        return Polya(alphas=result_alphas,
                     p_var_names=self.p_var_names,
                     a_var_names=self.a_var_names)

    def marginalize(self, vrs, keep=True):
        # TODO: handle different variable orderings
        vars_to_keep = super().get_marginal_vars(vrs, keep)
        if vars_to_keep == self.p_var_names:
            sum_alphas = np.sum(self.alphas)
            weight = gamma(sum_alphas) / (gamma(sum_alphas + 1))
            cardinalities = [len(self.alphas)]
            probs_table = {(assignment,): weight * prob for assignment, prob in
                           enumerate(self.alphas)}
            return Categorical(var_names=self.p_var_names,
                               cardinalities=cardinalities,
                               probs_table=probs_table)

        elif vars_to_keep == self.a_var_names:
            return Dirichlet(alphas=self.alphas, var_names=self.a_var_names)
        else:
            raise NotImplementedError()
