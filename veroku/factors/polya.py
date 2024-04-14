from veroku.factors._factor import Factor
import numpy as np

from veroku.factors._factor_utils import make_column_vector
from veroku.factors.dirichlet import Dirichlet
from veroku.factors.categorical import Categorical
from scipy.special import gamma
import copy

#TODO: change this to be a factorised factor type distribution.


class Polya(Factor):
    def __init__(self,
                 alphas,
                 log_weight=0.0,
                 x_var_names=None,
                 theta_var_names=None):
        """

        :param alphas: The alpha parameter values.
        :param log_weight: The log weight parameter.
        :param x_var_names: The conditional variable representing data samples.
        :param theta_var_names: The conditional theta variable names.
        """

        if x_var_names is None:
            x_var_names = [f"x_{i}" for i in range(len(alphas))]
        if theta_var_names is None:
            theta_var_names = [f"theta_{i}" for i in range(len(alphas))]

        assert len(theta_var_names) == len(alphas)
        assert len(theta_var_names) == len(alphas)
        self.x_var_names = x_var_names
        self.theta_var_names = theta_var_names
        self.a_var_names = copy.deepcopy(a_var_names)
        super().__init__(var_names=x_var_names + a_var_names)
        self.alpha_vec = make_column_vector(alphas)
        self.log_weight = log_weight

    def __repr__(self):
        s = f"Polya Factor" \
            f"\nalphas: {np.round(self.alpha_vec.ravel(), 3)}"
        return s

    def copy(self):
        return Polya(alphas=self.alpha_vec,
                     log_weight=self.log_weight,
                     x_var_names=self.x_var_names,
                     theta_var_name=self.theta_var_names,
                     a_var_names=self.a_var_names)

    def reorder_alphas(self, new_order_a_var_names):
        new_order_alphas = []
        current_alphas = self.alpha_vec.ravel()
        for new_a in new_order_a_var_names:
            index_in_current_order = self.a_var_names.index(new_a)
            new_order_alphas.append(current_alphas[index_in_current_order])
        return make_column_vector(new_order_alphas)

    def _var_names_equal_same_order(self, other):
        if not self.x_var_names == other.x_var_names:
            return False
        if not self.theta_var_names == other.theta_var_names:
            return False
        if not self.a_var_names == other.a_var_names:
            return False
        return True

    def _var_name_sets_equal(self, other):
        if not set(self.x_var_names) == set(other.x_var_names):
            return False
        if not set(self.theta_var_names) == set(other.theta_var_names):
            return False
        if not set(self.a_var_names) == set(other.a_var_names):
            return False
        return True

    def equals_up_to_scale(self, other):
        if not isinstance(other, Polya):
            return False
        if not self._var_name_sets_equal(other):
            return False
        other_ordered_alpha_vec = other.reorder_alphas(self.a_var_names)
        if not np.all_close(self.alpha_vec, other_ordered_alpha_vec):
            return False

    def equals(self, other):
        if not self.equals_up_to_scale(self, other):
            return False
        if not np.isclose(self.log_weight, other.log_weight):
            return False
        return True

    #TODO: consider removing - should rather marginalize and then sample
    def sample_categorical(self):
        d = Dirichlet(self.alpha_vec)
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
        assert vrs == self.theta_var_names
        assert len(values) == 1
        inc_alpha_index = values[0]
        alphas = self.alpha_vec.copy()
        alphas[inc_alpha_index] += 1
        return Dirichlet(alphas=alphas, var_names=self.a_var_names)

    def absorb(self, factor):
        if not isinstance(factor, Dirichlet):
            raise NotImplementedError()
        assert factor.var_names == self.a_var_names
        result_alphas = self.alpha_vec + factor.alphas - 1 # TODO: check this
        result_log_weight = np.linalg.special.logsumexp([self.log_weight, factor.log_weight])
        return Polya(alphas=result_alphas,
                     log_weight=result_log_weight,
                     x_var_names=self.x_var_names,
                     theta_var_names=self.theta_var_names,
                     a_var_names=self.a_var_names)

    def marginalize(self, vrs, keep=True):
        # TODO: handle different variable orderings
        vars_to_keep = super().get_marginal_vars(vrs, keep)
        if vars_to_keep == self.theta_var_names:

            sum_alphas = np.sum(self.alpha_vec)
            weight = gamma(sum_alphas) / (gamma(sum_alphas + 1))
            cardinalities = [len(self.alpha_vec)]
            probs_table = {(assignment,): weight * prob for assignment, prob in
                           enumerate(self.alpha_vec)}
            return Categorical(var_names=self.theta_var_names,
                               cardinalities=cardinalities,
                               probs_table=probs_table)

        elif vars_to_keep == self.a_var_names:
            return Dirichlet(alphas=self.alpha_vec, var_names=self.a_var_names)
        else:
            raise NotImplementedError()
