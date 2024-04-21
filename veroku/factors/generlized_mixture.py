"""
A module for factorised factor functionality
"""
import numpy as np

from veroku.factors._factor import Factor
from veroku.factors._factor_utils import get_subset_evidence
from veroku._constants import DEFAULT_FACTOR_RTOL, DEFAULT_FACTOR_ATOL

# pylint: disable=protected-access


class GeneralizedMixture(Factor):
    """
    A factorised factor class that allows factors to be used in factorised form - a product of other factors.
    This is useful for improving efficiency between factor multiplication and marginalisation. It also allows the
    for a efficient and intuitive representation of independent factors (especially factors with disjoint scopes).
    """

    # TODO: see how this is used in practice and investigate what the best way is to treat the variable name parameters.
    def __init__(self, factors):
        """
        The initializer.

        :param factors: The list of factors that will be implicitly multiplied.
        """
        assert all([not isinstance(factor, GeneralizedMixture) for factor in factors])
        var_names = factors[0].var_names
        for factor in factors:
            assert set(var_names) == set(factor.var_names)
        super().__init__(var_names=var_names)
        self.factors = []
        for factor in factors:
            if isinstance(factor, GeneralizedMixture):
                self.factors += [f.copy() for f in factor.factors]
            else:
                self.factors.append(factor.copy())

    def copy(self):
        """
        Copy this factorised factor.

        :return: the copied factor
        :rtype: GeneralizedMixture
        """
        factor_copies = []
        for factor in self.factors:
            factor_copies.append(factor.copy())
        return GeneralizedMixture(factor_copies)

    def __iter__(self):
        yield from self.factors

    def __repr__(self):
        s = "\n\n".join([f.__repr__() for f in self])
        return s

    def multiply(self, factor):
        """
        Multiply by another factor.

        :param factor: any type of factor.
        :return: the resulting factor
        :rtype: GeneralizedMixture
        """

        if isinstance(factor, GeneralizedMixture):
            other_mixture_factors = factor
        else:
            other_mixture_factors = [factor]
        result_mixture_factors = []
        for other_factor_component in other_mixture_factors:
            for factor_component in self.factors:
                product_factor = other_factor_component.multiply(factor_component)
                result_mixture_factors.append(product_factor)
        return GeneralizedMixture(result_mixture_factors)

    def divide(self, factor):
        """
        Divide out a general factor.

        :param factor: any type of factor.
        :return: the resulting factor
        :rtype: GeneralizedMixture
        """
        raise NotImplementedError()

    @property
    def is_vacuous(self):
        """
        Check if the factor is vacuous (i.e uniform).

        :return: The result of the check.
        :rtype: Bool
        """
        all_vacuous = all([factor._is_vacuous for factor in self.factors])
        if all_vacuous:
            return True
        none_vacuous = all([not factor._is_vacuous for factor in self.factors])
        if none_vacuous:
            return False
        # TODO: implement this
        raise NotImplementedError()

    def distance_from_vacuous(self):
        """
        Get the Kullback-Leibler (KL) divergence between this factor and a uniform copy of it.

        :return: The KL divergence.
        :rtype: float
        """
        raise NotImplementedError()

    def kl_divergence(self, factor):
        """
        Get the KL-divergence D_KL(self || factor) between a normalized version of this factor and another factor.

        :param factor: The other factor
        :type factor: Factor
        :return: The Kullback-Leibler divergence
        :rtype: float
        """
        raise NotImplementedError()

    @property
    def num_factors(self):
        """
        Get the number of factors in the factorised factor.

        :return: The number of factors.
        :rtype: int
        """
        return len(self.factors)

    def add_log_weight(self, log_weight):
        split_log_weight = log_weight - np.log(len(self.factors))
        for factor_i in self.factors:
            factor_i.log_weight += split_log_weight

    def normalize(self):
        """
        Normalize the factor.

        :return: The normalized factor.
        """
        # TODO: check this
        combined_log_weight = 0
        for factor_i in self.factors:
            combined_log_weight += factor_i.log_weight
        factor_copies = [f.copy() for f in self.factors]
        normalized_factor = GeneralizedMixture(factor_copies)
        normalized_factor.add_log_weight(combined_log_weight)
        return normalized_factor

    def marginalize(self, vrs, keep=True):
        """
        Marginalize out a subset of the variables in this factor's scope.

        :param list vrs: the variable names
        :param bool keep: whether to keep or sum (or integrate) out these variables.
        :return: the resulting marginal
        :rtype: FactorisedFactor
        """
        vars_to_keep = super().get_marginal_vars(vrs, keep)
        vars_to_integrate_out_set = set(self.var_names) - set(vars_to_keep)
        factor_marginals = []
        for factor in self.factors:
            factor_vars_to_integrate_out_set = set(factor.var_names).intersection(vars_to_integrate_out_set)
            if factor_vars_to_integrate_out_set == set(factor.var_names):
                raise NotImplementedError()
            factor_vars_to_integrate_out = list(factor_vars_to_integrate_out_set)
            factor_marginal = factor.marginalize(factor_vars_to_integrate_out, keep=False)
            factor_marginals.append(factor_marginal)
        if len(factor_marginals) == 1:
            return factor_marginals[0]
        marginal_factor = GeneralizedMixture(factor_marginals)
        return marginal_factor

    def reduce(self, vrs, values):
        """
        Observe a subset of the variables in the scope of this Gaussian and return the resulting factor.

        :param vrs: the names of the observed variable (list)
        :type vrs: str list
        :param values: the values of the observed variables
        :type values: vector-like
        :return: the resulting Gaussian
        :rtype: GeneralizedMixture
        """
        all_evidence_dict = dict(zip(vrs, values))
        reduced_factors = []
        for factor in self.factors:
            factor_var_names = factor.var_names
            if set(vrs) == set(factor_var_names):
                raise NotImplementedError()
            elif len(set(vrs).intersection(set(factor_var_names))) > 0:
                subset_vrs, subset_values = get_subset_evidence(
                    all_evidence_dict=all_evidence_dict, subset_vars=factor.var_names
                )
                reduced_factor = factor.reduce(subset_vrs, subset_values)
                reduced_factors.append(reduced_factor)
            else:
                reduced_factors.append(factor.copy())
        return GeneralizedMixture(reduced_factors)

    def equals(self, factor, rtol=DEFAULT_FACTOR_RTOL, atol=DEFAULT_FACTOR_ATOL):
        """
        Check if this factor is the same as another factor.

        :param factor: The other factor to compare to.
        :type factor: GeneralizedMixture
        :param float rtol: The relative tolerance to use for factor equality check.
        :param float atol: The absolute tolerance to use for factor equality check.
        :return: The result of the comparison.
        :rtype: bool
        """
        if not isinstance(factor, GeneralizedMixture):
            # TODO: find a better solution here
            return self.equals(GeneralizedMixture([factor]))
        if set(factor.var_names) != set(self.var_names):
            return False
        if len(self.factors) != len(factor.factors):
            return False

        num_matches = 0
        for self_factor_i in self.factors:
            for other_factor_j in factor.factors:
                if self_factor_i.equals(other_factor_j):
                        num_matches += 1
        if num_matches == len(self.factors):
            return True
        return False

    def show(self):
        """
        Print this factorised factor.
        """
        for i, factor in self.factors:
            print(f"\n factor {i}/{len(self.factors)}:")
            factor.show()
