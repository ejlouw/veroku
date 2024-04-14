"""
A module for instantiating sparse tables with log probabilities.
"""

# Standard imports
import copy
import itertools
import operator

# Third-party imports
import numpy as np
import scipy.special
from scipy import special
import pandas as pd

# Local imports
from veroku.factors._factor import Factor
from veroku._constants import DEFAULT_FACTOR_RTOL, DEFAULT_FACTOR_ATOL
from veroku.factors.experimental.gaussian_mixture import GaussianMixture
from veroku.factors.gaussian import Gaussian
from veroku.factors.generlized_mixture import GeneralizedMixture
from veroku.factors.constant_factor import ConstantFactor
from veroku.factors.sparse_categorical import SparseCategorical


# pylint: disable=protected-access

#TODO: remove duplicate code here (copied from SparseCategorical)

#TODO: replace _get_nested_sorted_probs with this and add example here
def _get_nested_sorted_categorical_factor(
        new_variables_order_outer, new_variables_order_inner, old_variable_order, old_assignments
):
    """
    Reorder variables to a new order (and sort assignments) and then convert the probs dictionary to a hierarchical
    dictionary with certain variables (or rather their corresponding assignments) in the outer dictionary and the rest
    in the inner dictionaries.

    :param new_variables_order_outer: The variables (and their order) for the outer scope.
    :type new_variables_order_outer: str list
    :param new_variables_order_inner: The variables (and their order) for the inner scope.
    :type new_variables_order_inner: str list
    :param old_variable_order: The initial variable order corresponding to old_assign_probs/
    :type old_variable_order: str list
    :param old_assignments: A dictionary of assignment and corresponding probabilities.
    :type old_assignments: dict
    """
    new_variable_order = new_variables_order_outer + new_variables_order_inner
    new_order_indices = [new_variable_order.index(var) for var in old_variable_order]
    new_assign_probs = dict()
    for old_assign_i, old_prob_i in old_assignments.items():
        new_row_assignment = [None] * len(old_assign_i)
        for old_i, new_i in enumerate(new_order_indices):
            new_row_assignment[new_i] = old_assign_i[old_i]
        l1_assign = tuple(new_row_assignment[: len(new_variables_order_outer)])
        if l1_assign not in new_assign_probs:
            new_assign_probs[l1_assign] = dict()
        assign_l2 = tuple(new_row_assignment[len(new_variables_order_outer):])
        new_assign_probs[l1_assign][assign_l2] = old_prob_i
    return new_assign_probs, new_variable_order


def _same_scope_binary_operation(probs_table_a, probs_table_b, func):
    """
    Apply a mathematical operation between the two factors with the same variable scope.
    NB: this function assumes that the variables corresponding to the keys in the two different dicts have the same
     order.

    :param dict probs_table_a: The probs dictionary for factor (typically sub factor) A.
    :param dict probs_table_b: The probs dictionary for factor (typically sub factor) B.
    """
    result_common_sub_dict = dict()
    all_common_assignments = set([*probs_table_a.keys()] + [*probs_table_b.keys()])
    for assignment in all_common_assignments:
        a_val = probs_table_a[assignment]
        b_val = probs_table_b[assignment]
        r_val = func(a_val, b_val)
        result_common_sub_dict[assignment] = r_val
    return result_common_sub_dict


def _flatten_ntd(ntd):
    """
    Flatten a two level nested table dictionary.
    """
    flattened_ntd = dict()
    for assign_outer in ntd.keys():
        for assign_inner in ntd[assign_outer].keys():
            flattened_ntd[assign_outer + assign_inner] = ntd[assign_outer][assign_inner]
    return flattened_ntd


def _apply_any_scope_binary_opp(ntd_a,
                                ntd_b,
                                symmetric_difference_combinations,
                                func):
    """
    Apply a binary operation using the two nested tables, a list of all the outer assignments that need to be process
     and default inner sub tables for each. This function loops through the outer assignments in the
    symmetric_difference_combinations, and gets the inner sub-table for each nested distribution dictionary (if the
    assignment is not present, the respective default sub-tables are used). The binary operation applied using these
    sub-table pairs and the result, finally, forms a part of the resulting final table.

    :param dict ntd_a: The one nested dictionary representing the distribution.
    :param dict ntd_b: The other nested dictionary representing the distribution.
    :param iterable symmetric_difference_combinations: The list of outer scope assignments that need to be processed.
    :param func: The binary operation.
    :return: The resulting distribution dictionary.
    :rtype: dict
    """
    resulting_factor_table = dict()
    for sd_assign_a, sd_assign_b in symmetric_difference_combinations:
        joined_sd_assignment = sd_assign_a + sd_assign_b

        # TODO: Add special, pre-computed result if both 0.
        common_vars_subtable_a = ntd_a[sd_assign_a]
        common_vars_subtable_b = ntd_b[sd_assign_b]

        # TODO: extend _same_scope_binary_operation functionality to allow table completion given flag.
        subtable_result = _same_scope_binary_operation(
            common_vars_subtable_a, common_vars_subtable_b, func
        )
        if subtable_result:
            resulting_factor_table[joined_sd_assignment] = subtable_result
    return resulting_factor_table


def _any_scope_bin_opp_none(ntd_a, ntd_b, outer_inner_cards_a, outer_inner_cards_b, func):
    """
    Apply a binary operation between categorical tables that have the same, disjoint or overlapping variable scopes.
    This function assumes that no combination of default or non-default values is guaranteed to result in a default
    value.
    :param ntd_a: The one nested dictionary representing the distribution.
    :param ntd_b: The other nested dictionary representing the distribution.
    ::param outer_inner_cards_a: The cardinalities of the inner and outer variables respectively in ntd_a
     (i.e [[2],[2,3]]).
    :param outer_inner_cards_b: The cardinalities of the inner and outer variables respectively in ntd_b
     (i.e [[2],[2,3]]).
    :return: The resulting distribution dictionary.
    :rtype: dict
    """
    # we have to check everything - any combination of {value, default_value} could result in non-default value.
    ntd_a_sd_assignments = list(itertools.product(*[range(c) for c in outer_inner_cards_a[0]]))
    ntd_b_sd_assignments = list(itertools.product(*[range(c) for c in outer_inner_cards_b[0]]))
    symmetric_difference_combinations = itertools.product(*[ntd_a_sd_assignments, ntd_b_sd_assignments])
    resulting_table = _apply_any_scope_binary_opp(ntd_a,
                                                  ntd_b,
                                                  symmetric_difference_combinations,
                                                  func)
    return resulting_table



def _fast_copy_assignments_table(table):
    """
    Copy a dictionary representation of a probability table faster than the standard deepcopy.

    :param dict table: A dictionary with the tuples of ints as keys and floats as values.
    :return: The copied table.
    """
    table_copy = {}
    for assign, assignment_entry in table.items():
        if isinstance(assignment_entry, Factor):
            assignment_entry_copy = copy.deepcopy(assignment_entry)
        else:
            assignment_entry_copy = assignment_entry
        table_copy[tuple(assign)] = assignment_entry_copy
    return table_copy

def get_mixture_marginal(factors_table, assignment_vars_to_sum_out):
    marginal_factor_sum_list = []
    for assignment_factor in factors_table.values():
        factor_vars_to_sum_out = list(
            set(assignment_factor.var_names).intersection(assignment_vars_to_sum_out)
        )

        if len(factor_vars_to_sum_out) > 0:
            assignment_factor_marginal = assignment_factor.marginalize(
                factor_vars_to_sum_out, keep=False
            )
        else:
            assignment_factor_marginal = assignment_factor.copy()
        marginal_factor_sum_list.append(assignment_factor_marginal)
    return GeneralizedMixture(marginal_factor_sum_list)

# Note: this is based on the SparseCategorical class, in terms of the dictionary structure of the,
# table, but only supports dense tables.

class GeneralizedCategorical(Factor):
    """
    A class for instantiating sparse tables with log probabilities.
    """

    def __init__(self, var_names, cardinalities, factors_table=None):
        """
        Construct a sparse categorical factor. Either log_probs_table or probs_table should be supplied.

        :param var_names: The categorical variable names - the factors (factors table dictionary
        keys will have their own variable names).
        :type var_names: str list
        :param cardinalities: The cardinalities of the variables (i.e, for three binary variables: [2,2,2])
        :type cardinalities: int list
        :param factors_table: A dictionary with assignments (tuples) as keys and assignment factors as values.
        """
        # TODO: add check that assignment lengths are consistent with var_names
        # TODO: add check that cardinalities are consistent with assignments

        all_assignment_entry_factors_var_names = []
        for assignment in factors_table.keys():
            if not (isinstance(assignment, tuple) or isinstance(assignment, list)):
                raise ValueError()
        for f in factors_table.values():
            all_assignment_entry_factors_var_names += f.var_names
        self.continuous_var_names = list(set(all_assignment_entry_factors_var_names))
        self.categorical_var_names = var_names
        all_var_names = self.continuous_var_names + self.categorical_var_names

        super().__init__(var_names=all_var_names)
        if len(cardinalities) != len(self.categorical_var_names):
            raise ValueError("The cardinalities and var_names lists should be the same length.")
        if factors_table is None:
            raise ValueError("probs_table must be specified")
        self.factors_table = _fast_copy_assignments_table(factors_table)
        self.var_cards = dict(zip(var_names, cardinalities))
        self.cardinalities = cardinalities
        self._assert_dense()

    def _assert_dense(self):
        """
        Assert that a factor is a dense factor.
        """
        sorted_var_cards = [range(self.var_cards[v]) for v in self.categorical_var_names]
        dense_assignments = itertools.product(*sorted_var_cards)
        for assign in dense_assignments:
            if assign not in self.factors_table:
                raise ValueError("Not dense")

    def _all_non_default_equal(self, factor):
        """
        Check that all non default values in this factor are the same as the corresponding values in factor, where
        the two factors have the same variable scope.

        :param factor: The other factor
        :return: The result of the check.
        :rtype: bool
        """

        for assign, self_assignment in self.factors_table.items():
            if assign not in factor.factors_table:
                if self_assignment != factor.default_factor:
                    return False
            else:
                other_assignment = factor.factors_table[assign]
                if self_assignment != other_assignment:
                    return False
        return True

    def equals(self, factor, rtol=DEFAULT_FACTOR_RTOL, atol=DEFAULT_FACTOR_ATOL):
        """
        Check if this factor is the same as another factor.

        :param factor: The other factor to compare to.
        :type factor: GeneralizedCategorical
        :param float rtol: The relative tolerance to use for factor equality check.
        :param float atol: The absolute tolerance to use for factor equality check.
        :return: The result of the comparison.
        :rtype: bool
        """
        factor_ = factor
        if not isinstance(factor_, GeneralizedCategorical):
            raise TypeError(f"factor must be of GeneralisedSparseCategorical type but has type {type(factor)}")

        if set(self.var_names) != set(factor_.var_names):
            return False

        # var sets are the same
        if self.var_names != factor.var_names:
            factor_ = factor.reorder(self.var_names)
        # factors now have same variable order

        # everywhere that self has non default values, factor has the same values.
        if not self._all_non_default_equal(factor_):
            return False
        # TODO: improve efficiency here (there could be a lot of duplication with the above loop)
        #   Check the values for every non-default assignment of factor
        if not factor_._all_non_default_equal(self):
            return False

        # If all possible assignments have not been checked - check that the default values are the same
        if not self._is_dense() and not factor._is_dense():
            if self.default_factor != factor.default_factor:
                return False

        return True

    def copy(self):
        """
        Make a copy of this factor.

        :return: The copy of this factor.
        :rtype: GeneralizedCategorical
        """
        return GeneralizedCategorical(
            var_names=self.categorical_var_names.copy(),
            factors_table=_fast_copy_assignments_table(self.factors_table),
            cardinalities=copy.deepcopy(self.cardinalities)
        )

    def marginalize(self, vrs, keep=True):
        """
        Sum out variables from this factor.

        :param vrs: (list) a subset of variables in the factor's scope
        :param keep: Whether to keep or sum out vrs
        :return: The resulting factor.
        :rtype: GeneralizedCategorical
        """

        vars_to_keep = super().get_marginal_vars(vrs, keep)
        vars_to_sum_out = [v for v in self.var_names if v not in vars_to_keep]
        categorical_vars_to_keep = list(set(vars_to_keep).intersection(self.categorical_var_names))
        categorical_vars_to_sum_out = list(set(vars_to_sum_out).intersection(self.categorical_var_names))

        nested_table, _ = _get_nested_sorted_categorical_factor(
            new_variables_order_outer=categorical_vars_to_keep,
            new_variables_order_inner=categorical_vars_to_sum_out,
            old_variable_order=self.categorical_var_names,
            old_assignments=self.factors_table,
        )
        result_table = dict()
        for l1_assign, factors_group_table in nested_table.items():
            l1_marginal_factors = []
            for a_factor in factors_group_table.values():
                a_factor_vars_to_sum = set(vars_to_sum_out).intersection(a_factor.var_names)
                a_marginal = a_factor.marginalize(a_factor_vars_to_sum, keep=False)
                l1_marginal_factors.append(a_marginal)
            if all([isinstance(f, ConstantFactor) for f in l1_marginal_factors]):
                log_constants = [f.log_constant_value for f in l1_marginal_factors]
                mixture_log_constant = scipy.special.logsumexp(log_constants)
                l1_mixture = ConstantFactor(mixture_log_constant)
            #elif all([isinstance(f, Gaussian) for f in l1_marginal_factors]):
            #    gaussians = [f for f in l1_marginal_factors]
            #    l1_mixture = GaussianMixture(gaussians)
            else:
                l1_mixture = GeneralizedMixture(l1_marginal_factors)
            result_table[l1_assign] = l1_mixture

        if len(result_table) == 1:
            assert len(categorical_vars_to_keep) == 0
            # all categorical variables have been summed out.
            marginal_factor = list(result_table.values())[0]
            return marginal_factor

        # remove cardinalities of summed out categorical variables
        result_var_cards = copy.deepcopy(self.var_cards)
        for var in categorical_vars_to_sum_out:
            del result_var_cards[var]

        cardinalities = [self.var_cards[v] for v in categorical_vars_to_keep]
        if all([isinstance(v, ConstantFactor) for v in result_table.values()]):
            marginal_log_probs_table = {a:f.log_constant_value for a,f in result_table.items()}
            marginal = SparseCategorical(
                cardinalities=result_var_cards,
                log_probs_table=marginal_log_probs_table,
                var_names=categorical_vars_to_keep
            )
        else:
            marginal = GeneralizedCategorical(
                var_names=categorical_vars_to_keep, factors_table=result_table, cardinalities=cardinalities
            )
        return marginal
    def marginalize_OLD(self, vrs, keep=True):
        """
        Sum out variables from this factor.

        :param vrs: (list) a subset of variables in the factor's scope
        :param keep: Whether to keep or sum out vrs
        :return: The resulting factor.
        :rtype: GeneralizedCategorical
        """

        vars_to_keep = super().get_marginal_vars(vrs, keep)
        vars_to_sum_out = [v for v in self.var_names if v not in vars_to_keep]
        categorical_vars_to_keep = set(vars_to_keep).intersection(self.categorical_var_names)
        categorical_vars_to_sum_out = set(vars_to_sum_out).intersection(self.categorical_var_names)
        continuous_vars_to_keep = set(vars_to_keep).intersection(self.continuous_var_names)
        continuous_vars_to_sum_out = set(vars_to_sum_out).intersection(self.continuous_var_names)
        marginal_factor_sum_list = []
        if len(categorical_vars_to_keep) == 0:
            mixture_marginal = get_mixture_marginal(self.factors_table, continuous_vars_to_sum_out)
            return mixture_marginal
        if len(continuous_vars_to_keep) == 0:
            raise NotImplementedError()
        nested_table, _ = _get_nested_sorted_categorical_factor(
            new_variables_order_outer=categorical_vars_to_keep,
            new_variables_order_inner=categorical_vars_to_sum_out,
            old_variable_order=self.categorical_vars_names,
            old_assignments=self.factors_table,
        )
        result_table = dict()
        for l1_assign, log_probs_table in nested_table.items():
            prob = special.logsumexp(list(log_probs_table.values()))
            result_table[l1_assign] = prob

        result_var_cards = copy.deepcopy(self.var_cards)
        for var in vars_to_sum_out:
            del result_var_cards[var]

        cardinalities = [self.var_cards[v] for v in vars_to_keep]

        resulting_factor = GeneralizedCategorical(
            var_names=vars_to_keep, factors_table=result_table, cardinalities=cardinalities
        )

        return resulting_factor

    def reduce(self, vrs, values):
        """
        Observe variables to have certain values and return reduced table.

        :param vrs: (list) The variables.
        :param values: (tuple or list) Their values
        :return: The resulting factor.
        :rtype: GeneralizedCategorical
        """

        vars_unobserved = [var_name for var_name in self.var_names if var_name not in vrs]
        nested_table, _ = _get_nested_sorted_categorical_factor(
            new_variables_order_outer=vrs,
            new_variables_order_inner=vars_unobserved,
            old_variable_order=self.var_names,
            old_assignments=self.factors_table,
        )
        lp_table = nested_table[tuple(values)]
        result_var_cards = copy.deepcopy(self.var_cards)
        for var in vrs:
            del result_var_cards[var]

        cards = list(result_var_cards.values())
        var_names = vars_unobserved
        resulting_factor = GeneralizedCategorical(var_names=var_names, factors_table=lp_table, cardinalities=cards)

        return resulting_factor

    def _assert_consistent_cardinalities(self, factor):
        """
        Assert that the variable cardinalities are consistent between two factors.

        :param GeneralizedCategorical factor: The factor to compare with.
        """
        for var in self.var_names:
            if var in factor.var_cards:
                error_msg = f"Error: inconsistent variable cardinalities: {factor.var_cards}, {self.var_cards}"
                assert self.var_cards[var] == factor.var_cards[var], error_msg

    def multiply(self, factor):
        """
        Multiply this factor with another factor and return the result.

        :param factor: The factor to multiply with.
        :type factor: GeneralizedCategorical
        :return: The factor product.
        :rtype: GeneralizedCategorical
        """
        if not isinstance(factor, GeneralizedCategorical):
            raise TypeError(f"factor must be of GeneralisedSparseCategorical type but has type {type(factor)}")
        return self._apply_binary_operator(factor, operator.mul)

    def cancel(self, factor):
        """
        Almost like divide, but with a special rule that ensures that division of zeros by zeros results in zeros.

        :param factor: The factor to divide by.
        :type factor: GeneralizedCategorical
        :return: The factor quotient.
        :rtype: GeneralizedCategorical
        """
        if not isinstance(factor, GeneralizedCategorical):
            raise TypeError(f"factor must be of GeneralisedSparseCategorical type but has type {type(factor)}")
        return self._apply_binary_operator(factor, operator.truediv)

    def divide(self, factor):
        """
        Divide this factor by another factor and return the result.

        :param factor: The factor to divide by.
        :type factor: GeneralizedCategorical
        :return: The factor quotient.
        :rtype: GeneralizedCategorical
        """
        return self._apply_binary_operator(factor, operator.sub)

    def argmax(self):
        """
        Get the Categorical assignment (vector value) that maximises the factor potential.

        :return: The argmax assignment.
        :rtype: int list
        """
        return max(self.factors_table.items(), key=operator.itemgetter(1))[0]

    def _apply_to_probs(self, func, include_assignment=False):
        """
        Apply a function to the log probs of the factor.

        :param func: The function to apply to the log probs in this factor.
        :param include_assignment: Whether or not to pass the assignment to the function as well
            (along with the log probs).
        """
        for assign, prob in self.factors_table.items():
            if include_assignment:
                self.factors_table[assign] = func(prob, assign)
            else:
                self.factors_table[assign] = func(prob)

    def _apply_binary_operator(self, factor, operator_function):
        """
        Apply a binary operator function f(self.factor, factor) and return the result

        :param factor: The other factor to use in the binary operation.
        :type factor: GeneralizedSparseCategorical
        If this parameter is not specified, 'none' will be used to ensure correct, albeit slower computation.
        :return: The resulting factor.
        :rtype: GeneralizedCategorical
        """
        # pylint: disable=too-many-locals
        if not isinstance(factor, GeneralizedCategorical):
            raise TypeError(f"factor must be of GeneralisedSparseCategorical type but has type {type(factor)}")
        self._assert_consistent_cardinalities(factor)
        intersection_vars = list(set(self.categorical_var_names).intersection(set(factor.categorical_var_names)))
        intersection_vars = sorted(intersection_vars)

        remaining_a_vars = list(set(self.categorical_var_names) - set(intersection_vars))
        ntd_a, _ = _get_nested_sorted_categorical_factor(
            new_variables_order_outer=remaining_a_vars,
            new_variables_order_inner=intersection_vars,
            old_variable_order=self.categorical_var_names,
            old_assignments=self.factors_table,
        )

        remaining_b_vars = list(set(factor.categorical_var_names) - set(intersection_vars))
        ntd_b, _ = _get_nested_sorted_categorical_factor(
            new_variables_order_outer=remaining_b_vars,
            new_variables_order_inner=intersection_vars,
            old_variable_order=factor.categorical_var_names,
            old_assignments=factor.factors_table,
        )

        def vars_to_cards(factor, var_names):
            return [factor.var_cards[v] for v in var_names]

        outer_inner_cards_a = [vars_to_cards(self, remaining_a_vars), vars_to_cards(self, intersection_vars)]
        outer_inner_cards_b = [
            vars_to_cards(factor, remaining_b_vars),
            vars_to_cards(factor, intersection_vars),
        ]
        result_ntd = _any_scope_bin_opp_none(
            ntd_a, ntd_b, outer_inner_cards_a, outer_inner_cards_b, operator_function
        )

        flattened_result_table = _flatten_ntd(result_ntd)
        result_var_names = remaining_a_vars + remaining_b_vars + intersection_vars

        result_var_cards = {**self.var_cards, **factor.var_cards}

        result_cardinalities = [result_var_cards[v] for v in result_var_names]
        resulting_factor = GeneralizedCategorical(
            var_names=result_var_names,
            factors_table=flattened_result_table,
            cardinalities=result_cardinalities
        )

        return resulting_factor

    def normalize(self):
        """
        Normalize the factor.

        :return: The normalized factor.
        :rtype: GeneralizedCategorical
        """

        factor_copy = self.copy()
        logz = special.logsumexp(list(factor_copy.factors_table.values()))
        for assign, prob in factor_copy.factors_table.items():
            factor_copy.factors_table[assign] = prob - logz
        return factor_copy

    def _is_dense(self):
        """
        Check if all factor assignments have non-default values.

        :return: The result of the check.
        :rtype: Bool
        """
        num_assignments = len(self.factors_table)
        max_possible_assignments = np.prod(self.cardinalities)
        if num_assignments == max_possible_assignments:
            return True
        return False

    @staticmethod
    def _raw_kld(log_p, log_q):
        """
        Get the raw numerically calculated kld (which could result in numerical errors causing negative KLDs).

        :param log_p: The log probability distribution tensor of distribution P.
        :param log_q: The log probability distribution tensor of distribution Q.
        :return: The KL-divergence
        :rtype: float
        """
        raise NotImplementedError()

    def kl_divergence(self, factor, normalize_factor=True):
        """
        Get the KL-divergence D_KL(P || Q) = D_KL(self||factor) between a normalized version of this factor and another
        factor.

        :param factor: The other factor
        :type factor: GeneralizedCategorical
        :param normalize_factor: Whether or not to normalize the other factor before computing the KL-divergence.
        :type normalize_factor: bool
        :return: The Kullback-Leibler divergence
        :rtype: float
        """
        raise NotImplementedError()

    def distance_from_vacuous(self):
        """
        Get the Kullback-Leibler (KL) divergence between this factor and a uniform copy of it.

        :return: The KL divergence.
        :rtype: float
        """
        raise NotImplementedError()

    def potential(self, vrs, assignment):
       raise NotImplementedError()

    @property
    def dense_distribution_array(self):
        raise NotImplementedError()

    @property
    def weight(self):
        """
        An array containing all the discrete assignments and the corresponding probabilities.
        """
        log_weight = special.logsumexp(list(self.factors_table.values()))
        weight_ = np.exp(log_weight)
        return weight_

    def _to_df(self):
        """
        Convert the factor to a dataframe representation.

        :return: The dataframe representation.
        :rtype: pandas.DataFrame
        """
        log_probs_table = self.factors_table
        var_names = self.var_names
        factor_df = pd.DataFrame.from_dict(log_probs_table.items()).rename(columns={0: "assignment", 1: "log_prob"})
        factor_df[var_names] = pd.DataFrame(factor_df["assignment"].to_list())
        factor_df.drop(columns=["assignment"], inplace=True)
        # Correct column order
        var_cols = [c for c in factor_df.columns if c != "log_prob"]
        return factor_df[var_cols + ["log_prob"]]

    def show(self):
        """
        Print the factor's string representation.
        """
        print(self.__repr__())

    def __repr__(self):
        """
        Get the string representation for the factor.

        :return: The representation string
        :rtype: str
        """
        tabbed_spaced_var_names = "\t".join(self.var_names) + "\tprob\n"
        repr_str = tabbed_spaced_var_names
        for assignment, assignment_entry in self.factors_table.items():
            repr_str += assignment_entry.__repr__() + "\n"
        return repr_str

    def reorder(self, new_var_names_order):
        """
        Reorder categorical table variables to a new order and reorder the associated probabilities
        accordingly.

        :param new_var_names_order: The new categorical variable order.
        :type new_var_names_order: str list
        :return: The factor with new order.

        Example:
            old_variable_order = [a, b]
            new_variable_order = [b, a]

            a b P(a,b)  return    b a P(a,b)
            0 0  pa0b0            0 0  pa0b0
            0 1  pa0b1            0 1  pa1b0
            1 0  pa1b0            1 0  pa0b1
            1 1  pa1b1            1 1  pa1b1
        """
        new_order_indices = [self.categorical_var_names.index(var) for var in new_var_names_order]
        new_log_probs_table = dict()
        for assignment, value in self.factors_table.items():
            reordered_assignment = tuple(assignment[i] for i in new_order_indices)
            new_log_probs_table[reordered_assignment] = value
        reordered_cardinalities = [self.cardinalities[i] for i in new_order_indices]

        return GeneralizedCategorical(
            var_names=new_var_names_order,
            factors_table=new_log_probs_table,
            cardinalities=reordered_cardinalities,
        )
