"""
A module for factorised factor functionality
"""

from veroku.factors._factor import Factor
from veroku.factors._factor_utils import get_subset_evidence

# TODO: This class in very much a work in progress. This needs to be completed. We must think especially carefully about
#  how we multiply factors when we have special factors such as nonlinear Gaussian factors in the factorised factor.


class FactorisedFactor(Factor):
    """
    A factorised factor class that allows factors to be used in factorised form - a product of other factors.
    This is useful for improving efficiency between factor multiplication and marginalisation. It also allows the
    for a efficient and intuitive representation of independent factors (especially factors with disjoint scopes).
    """

    # TODO: see how this is used in practice and investigate what the best way is to treat the variable name parameters.
    def __init__(self, factors):
        """
        The initialiser.
        :param factors: The list of factors that will be implicitly multiplied.
        """
        assert all([not isinstance(factor, FactorisedFactor) for factor in factors])
        factors_var_name_lists = ([factor.var_names for factor in factors])
        factors_var_name = list(set([var_name for sublist in factors_var_name_lists for var_name in sublist]))
        super().__init__(var_names=factors_var_name)
        self.factors = []
        for factor in factors:
            if isinstance(factor, FactorisedFactor):
                self.factors += [f.copy() for f in factor.factors]
            else:
                self.factors.append(factor.copy())

    @property
    def joint_distribution(self):
        """
        compute the joint distribution.
        :return:
        """
        if len(self.factors) == 0:
            raise ValueError('FactorisedFactor has no factors.')
        joint_distribution = self.factors[0].copy()
        for factor in self.factors[1:]:
            joint_distribution = joint_distribution.multiply(factor)
        return joint_distribution

    def copy(self):
        """
        Copy this factorised factor
        :return: the copied factor
        """
        factor_copies = []
        for factor in self.factors:
            factor_copies.append(factor.copy())
        return FactorisedFactor(factor_copies)

    def multiply(self, factor):
        """
        Multiply in a general factor
        :param factor: any type of factor.
        :return: the resulting Factorised Factor
        """
        # TODO: add preference for factors with conditioning scope? Not sure if practically necessary, but would be best.
        factorised_factor_copy = self.copy()
        index = factorised_factor_copy.first_factor_with_vars_index(factor.var_names)
        if index is not None:
            factorised_factor_copy.factors[index] = factorised_factor_copy.factors[index].multiply(factor)
        else:
            factorised_factor_copy.factors.append(factor)
        return factorised_factor_copy

    def divide(self, factor):
        """
        Divide out a general factor
        :param factor: any type of factor.
        :return: the resulting Factorised Factor
        """
        factorised_factor_copy = self.copy()
        index = factorised_factor_copy.first_factor_with_vars_index(factor.var_names)

        for i, factorised_comp in enumerate(self.factors):
            if factorised_comp.equals(factor):
                if len(self.factors) > 1:
                    del factorised_factor_copy.factors[i]
                    return factorised_factor_copy
                # TODO: add else to make general vacuous factor (wont necessarily be Gaussian)
        if index is not None:
            factorised_factor_copy.factors[index] = factorised_factor_copy.factors[index].divide(factor)
        else:
            raise Exception('Error: Cannot divide factor with disjoint scope.')
        return factorised_factor_copy

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
        # TODO: improve this
        return self.joint_distribution.distance_from_vacuous()

    def kl_divergence(self, factor):
        # TODO: improve this
        return self.joint_distribution.kl_divergence(factor)

    @property
    def num_factors(self):
        """
        Get the number of factors in the factorised factor.
        :return: (int) The number of factors.
        """
        return len(self.factors)

    #def marginalize(self, vrs, keep=False):
    #    """
    #    Marginalise out a subset of the variables in this factor's scope.
    #    :param vrs: (list) the variable names
    #    :param keep: (bool) whether to keep or sum (or integrate) out these variables.
    #    :return: (FactorisedFactor) the resulting marginal
    #    """
    #    cg = ClusterGraph(self.factors, make_animation_gif=False, debug=False, disable_tqdm=True)
    #    cg.process_graph()
    #    #vars_to_keep = super().get_marginal_vars(vrs, keep)
    #    #marginal = cg.get_marginal(vars_to_keep)
    #    posterior_joint = cg.get_posterior_joint()
    #    marginal = posterior_joint.marginalize(vrs, keep=keep)
    #    return marginal

    def marginalize(self, vrs, keep=False):
        """
        Marginalise out a subset of the variables in this factor's scope.
        :param vrs: (list) the variable names
        :param keep: (bool) whether to keep or sum (or integrate) out these variables.
        :return: (FactorisedFactor) the resulting marginal
        """
        # TODO: resolve this: Non-linear Gaussian factors can cause a problem here when there is enough information
        # that they all should be well defined, but the information resides in only one of them for instance. This
        # information takes the form of the observed evidence and the conditioning factors.

        vars_to_keep = super().get_marginal_vars(vrs, keep)
        vars_to_integrate_out_set = set(self.var_names) - set(vars_to_keep)
        factors_in_keep_scope_indices = self.all_intersecting_factors_indices(vars_to_keep)
        if not factors_in_keep_scope_indices:
            raise Exception('cannot marginalize out all variables from FactorisedFactor.')

        factor_marginals = []
        self_copy = self.copy()
        additional_log_weight = 0.0
        self_copy.merge_factors_with_shared_scope(vars_to_integrate_out_set)
        for factor in self_copy.factors:
            factor_vars_to_integrate_out_set = set(factor.var_names).intersection(vars_to_integrate_out_set)
            if factor_vars_to_integrate_out_set == set(factor.var_names):
                additional_log_weight += factor.get_log_weight()
            else:
                factor_vars_to_integrate_out = list(factor_vars_to_integrate_out_set)
                factor_marginal = factor.marginalize(factor_vars_to_integrate_out, keep=False)
                factor_marginals.append(factor_marginal)
        factor_marginals[0]._add_log_weight(additional_log_weight)
        if len(factor_marginals) == 1:
            return factor_marginals[0]
        marginal_factor = FactorisedFactor(factor_marginals)
        return marginal_factor

    # TODO: remove this
    def marginalise_OLD(self, vrs, keep=False):
        """
        Marginalise out a subset of the variables in this factor's scope.
        :param vrs: (list) the variable names
        :param keep: (bool) whether to keep or sum (or integrate) out these variables.
        :return: (FactorisedFactor) the resulting marginal
        """
        vars_to_keep = super().get_marginal_vars(vrs, keep)
        factors_in_keep_scope_indices = self.all_intersecting_factors_indices(vars_to_keep)
        factors_in_nokeep_scope_indices = list(set(range(self.num_factors)) - set(factors_in_keep_scope_indices))
        if not factors_in_keep_scope_indices:
            raise Exception('cannot marginalize out all variables from FactorisedFactor.')

        keep_scope_factor_product = self.factors[factors_in_keep_scope_indices[0]].copy()
        for index in factors_in_keep_scope_indices[1:]:
            factor_component = self.factors[index]
            keep_scope_factor_product = keep_scope_factor_product.multiply(factor_component)

        # TODO: this can potentially be improved by checking if all the keep scope factors maybe have
        #  set(vars)==vars_to_keep -> then we do not even need to multiply them together.
        if isinstance(keep_scope_factor_product, FactorisedFactor):
            raise TypeError
        keep_scope_factor_product_marginal = keep_scope_factor_product.marginalize(vars_to_keep, keep=True)

        joint_for_full_integration = self.factors[factors_in_nokeep_scope_indices[0]].copy()
        #TODO: here we can also optimise be integrating factors with disjoint variable scopes separately.
        for index in factors_in_nokeep_scope_indices[1:]:
            joint_for_full_integration = joint_for_full_integration.multiply(self.factors[index])
        additional_log_weight = joint_for_full_integration.get_log_weight()
        keep_scope_factor_product_marginal.add_log_weight(additional_log_weight)

        return keep_scope_factor_product_marginal

    #TODO: make these more efficient by storing joint_distribution
    def get_cov(self):
        return self.joint_distribution.get_cov()

    def get_mean(self):
        return self.joint_distribution.get_mean()

    # TODO: uncomment and test
    #def merge_dependent_factors(self):
    #    """
    #    Merge all factors (by multiplying) that have overlapping scopes.
    #    This is useful for?
    #    :return:
    #    """
    #    merged_factors = []
    #    already_merged_factor_indices = []
    #    for i, factor_i in enumerate(self.factors):
    #        factor_i_merged = factor_i.copy()
    #        for j in range(i+1,len(self.factors)):
    #            factor_j = self.factors[j]
    #            if j not in already_merged_factor_indices:
    #                if set(factor_i.var_names).intersection(set(factor_j.varnames)) > 0:
    #                    factor_i_merged = factor_i_merged.multiply(factor_j)
    #                    already_merged_factor_indices.append(j)
    #        merged_factors.append(factor_i_merged)
    #    return FactorisedFactor(merged_factors)

    def merge_factors_with_shared_scope(self, vrs_set):
        """
        Merge all factors in groups that all contain variables in vrs and have overlap within these subsets.
        This function will typically be used to ensure that all factors containing variables that need to be marginalised
        out are merged together into disjoint (in terms of the integrand variables) factors, so that these larger factors
        can be marginalised independently.
        :param vrs_set: (string list) The common scope variables to be considered to determine 'dependent' factors.
        """
        observed_factor_vars_list = []

        for f in self.factors:
            observed_factor_vars = set(f.var_names).intersection(vrs_set)
            observed_factor_vars_list.append(observed_factor_vars)

        merged_factors = []
        merged_factors_indices = []
        for i, observed_factor_i_vars in enumerate(observed_factor_vars_list):
            if i not in merged_factors_indices:
                merged_factor = self.factors[i]
                for j in range(i + 1, len(observed_factor_vars_list)):
                    observed_factor_j_vars = observed_factor_vars_list[j]
                    if j not in merged_factors_indices:
                        if len(observed_factor_i_vars.intersection(observed_factor_j_vars)) > 0:
                            merged_factor = merged_factor.multiply(self.factors[j])
                            merged_factors_indices.append(j)
                merged_factors.append(merged_factor)
        self.factors = merged_factors

    def reduce(self, vrs, values):
        """
        Observe certain values of the variables in vrs.
        :param vrs: (list) The observed variables.
        :param values: (list) Their values.
        :return: (FactorisedFactor) the reduced factor
        """
        all_evidence_dict = dict(zip(vrs, values))
        reduced_factors = []
        additional_log_weight = 0.0
        for factor in self.factors:
            factor_var_names = factor.var_names
            if set(vrs) == set(factor_var_names):
                additional_log_weight += factor.log_potential(x_val=values, vrs=vrs)
            elif len(set(vrs).intersection(set(factor_var_names))) > 0:
                subset_vrs, subset_values = get_subset_evidence(all_evidence_dict=all_evidence_dict,
                                                                subset_vars=factor.var_names)
                reduced_factor = factor.reduce(subset_vrs, subset_values)
                reduced_factors.append(reduced_factor)
            else:
                reduced_factors.append(factor.copy())
        adjusted_weight_factor_0 = reduced_factors[0]
        adjusted_weight_factor_0._add_log_weight(additional_log_weight)
        reduced_factors[0] = adjusted_weight_factor_0
        return FactorisedFactor(reduced_factors)

    def first_factor_with_vars_index(self, var_names):
        """
        returns index of the first factor in self_factors with var_names in its scope.
        """
        # TODO: add check for equals variable names - will be faster for multiplication
        for i, factor in enumerate(self.factors):
            if set(var_names).issubset(set(factor.var_names)):
                return i
        return None

    def all_intersecting_factors_indices(self, var_names):
        """
        return indices of factors in self_factors which has overlap with var_names.
        """
        indices = []
        # TODO: add check for equals variable names - will be faster for multiplication
        for i, factor in enumerate(self.factors):
            if len(set(var_names).intersection(set(factor.var_names))) > 0:
                indices.append(i)
        return indices

    def equals(self, factor):
        """
        Check if this factor is the same as another factor.

        :param factor: The other factor to compare to.
        :type factor: FactorisedFactor
        :return: The result of the comparison.
        :rtype: bool
        """
        if not isinstance(factor, FactorisedFactor):
            # TODO: find a better solution here
            return self.equals(FactorisedFactor([factor]))
        if set(factor.var_names) != set(self.var_names):
            return False
        if len(self.factors) != len(factor.factors):
            return self.joint_distribution.equals(factor.joint_distribution)

        self_comps_with_cpoies_in_factor = []
        for i, self_factor_i in enumerate(self.factors):
            if i not in self_comps_with_cpoies_in_factor:
                for j, other_factor_j in enumerate(factor.factors):
                    if self_factor_i.equals(other_factor_j):
                        self_comps_with_cpoies_in_factor.append(i)
        if len(self_comps_with_cpoies_in_factor) == len(self.factors):
            return True
        return False

    def show(self):
        """
        Display this factorised factor
        """
        for i, factor in self.factors:
            print(f'\n factor {i}/{len(self.factors)}:')
            factor.show()
