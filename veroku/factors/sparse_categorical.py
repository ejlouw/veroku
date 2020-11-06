"""
A module for instantiating sparse tables with log probabilities.
"""

# System imports
import copy
import operator

# Third-party imports
import numpy as np
from scipy import special
import pandas as pd
import itertools

# Local imports
from veroku.factors._factor import Factor
from veroku.factors._factor_template import FactorTemplate


def _make_dense(factor):
    """
    Convert a factor to a dense factor by instanciating all of the implicit default value assignments.
    :param factor: The factor to convert.
    :type factor: SparseCategorical
    :return: The dense factor version.
    :rtype: SparseCategorical
    """
    factor_copy = factor.copy()
    dense_assingments = itertools.product(*[range(factor_copy.var_cards[v]) for v in factor_copy.var_names])
    for assign in dense_assingments:
        if assign not in factor_copy.log_probs_table:
            factor_copy.log_probs_table[assign] = factor_copy.default_log_prob
    return factor_copy


def _make_dense_default_probs_dict(cardinalities, default_value):
    """
    Make a dictionary representing a categorical probability table that contains all assignments, as specified by the
    given cardinalities (starting from 0).
    :param cardinalities: The given cardinalities.
    :type cardinalities: list
    :param default_value: The default value to assign the missing values.
    :type default_value: float
    :returns: The dense representation.
    :rtype: dict
    Example:
    >>> _make_dense_default_probs_dict(cardinalities=[2,3], default_value=0.0)
    {(0,0):0.0,
    (0,1):0.0,
    (0,2):0.0,
    (1,0):0.0,
    (1,1):0.0,
    (1,2):0.0}
    """
    dense_assingments = itertools.product(*[range(c) for c in cardinalities])
    dense_default_dict = dict(zip(dense_assingments, [default_value] * np.product(cardinalities)))
    return dense_default_dict


def _make_inner_dense(sparse_nested_table, outer_inner_cards, default_value):
    """
    Convert n sparse nested table dictionary's implicit default values in the innetables to real entries so that
    all possible assignments are present in the inner sub tables.
    :param sparse_nested_table: The nested dictionary representing a sparse table.
    :type sparse_nested_table: dictionary of dictionaries
    :param outer_inner_cards: The lists of the cardinalities for the outer and inner part of the nested table respectively
        i.e [[2,2], [3, 4]] if the table dict is {(0,0):{(2,3):0.3},....} where the outer variables has cardinalities [2,2]
        and the inner ones [3,4]
    """
    sparse_nested_inner_dense_table = copy.deepcopy(sparse_nested_table)
    inner_cards = outer_inner_cards[1]

    all_inner_table_assignments = itertools.product(*[range(c) for c in inner_cards])
    for outer_assign in sparse_nested_inner_dense_table.keys():
        for inner_assign in all_inner_table_assignments:
            if inner_assign not in sparse_nested_inner_dense_table[outer_assign]:
                sparse_nested_inner_dense_table[outer_assign][inner_assign] = default_value
    return sparse_nested_inner_dense_table


def _get_nested_sorted_probs(new_variables_order_outer,
                             new_variables_order_inner,
                             old_variable_order, old_assign_probs):
    """
    Reorder probs to a new order and sort assignments.
    :params new_variables_order_outer:
    :params new_variables_order_inner:
    :params old_variable_order:
    :params old_assign_probs: A dictionary of assignment and coresponding probabilities.
    Example:
    old_variable_order = [a, b]
    new_variables_order_outer = [b]

      a  b  c   P(a,b)     return:       b    a  c  P(b,a)
    {(0, 0, 0): pa0b0c0                {(0):{(0, 0): pa0b0,
     (0, 1, 0): pa0b1c0                      (1, 0): pa1b0}
     (1, 0, 1): pa1b0c1                 (1):{(0, 1): pa0b1,
     (1, 1, 1): pa1b1c1}                     (1, 1): pa1b1}}
    """
    # TODO: Complete and improve docstring.
    new_variable_order = new_variables_order_outer + new_variables_order_inner
    new_order_indices = [new_variable_order.index(var) for var in old_variable_order]
    new_assign_probs = dict()
    for old_assign_i, old_prob_i in old_assign_probs.items():
        new_row_assignment = [None] * len(old_assign_i)
        for old_i, new_i in enumerate(new_order_indices):
            new_row_assignment[new_i] = old_assign_i[old_i]
        l1_assign = tuple(new_row_assignment[:len(new_variables_order_outer)])
        if l1_assign not in new_assign_probs:
            new_assign_probs[l1_assign] = dict()
        assign_l2 = tuple(new_row_assignment[len(new_variables_order_outer):])
        new_assign_probs[l1_assign][assign_l2] = old_prob_i
    return new_assign_probs, new_variable_order


def same_scope_binary_operation(a, b, func, default):
    """
    NB: this function assumes that the variables corresponding to the keys in the two different dicts have the same order.
    :param a: The dictionary for factor (typically sub factor) A.
    :param b: The dictionary for factor (typically sub factor) B.
    """
    result_common_sub_dict = dict()
    all_common_assignments = set([*a.keys()] + [*b.keys()])
    new_default = func(default, default)
    for assignment in all_common_assignments:
        a_val = a.get(assignment, default)
        b_val = b.get(assignment, default)
        r_val = func(a_val, b_val)
        if r_val != new_default:
            result_common_sub_dict[assignment] = r_val
    return result_common_sub_dict


def flatten_ntd(ntd):
    """
    Flatten a two level nested table dictionary.
    """
    flattened_ntd = dict()
    for assign_outer in ntd.keys():
        for assign_inner in ntd[assign_outer].keys():
            flattened_ntd[assign_outer + assign_inner] = ntd[assign_outer][assign_inner]
    return flattened_ntd


def any_scope_binary_operation(ntd_a, outer_inner_cards_a,
                               ntd_b, outer_inner_cards_b,
                               func, default, default_rules='none'):
    """
    Apply a binary operation between categorical tables that have the same, disjoint or overlapping variable scopes.

    :param ntd_a: The nested table dictionary for factor A.
    :param ntd_b: The nested table dictionary for factor B.
    :param default: The default value for ntd_a and ntd_b (the values for the missing values).
    :param default_rules: The rules for when a calculation results will result in a default value (optional).
        This can help speed up this function. The possible values are as follows:
            'any' : If either ntd_a or ntd_b has a default value, the result will always be default.
            'both': Only if both ntd_a and ntd_b has a default value, the result will always be default.
            'none': No combination of default or non-default values is guarenteed to result in a default value
        If this parameter is not specified, 'none' will be used to ensure correct, albeit slower computation.
    :returns: The nested table dictionary for the resulting factor.
    """

    # TODO: this will not work for tables with the same scope, as this will mess up the nestedness(?).
    #  Seems like it works - confirm.

    resulting_factor_table = dict()
    ntd_a_sd_assignments = ntd_a.keys()
    ntd_b_sd_assignments = ntd_b.keys()
    # Note: sd: symmetric_difference

    full_sd_a_default_sub_table = None
    full_sd_b_default_sub_table = None

    # Calculate the symmetric_difference_combinations
    # The symmetric_difference_combinations are the combinations of the sub-assignments
    # that are not common to both factors. Depending on the value of default_rules, these
    # assignments could only include assignments actually specified in the nested table
    # dictionaries, or some or all of the missing assignments (if default values in either)
    # table does not necessarily guarantee default results.
    if default_rules == 'none':

        # we have to check everything - any combination of {value, default_value} could result in non-default value.
        ntd_a_sd_assignments = list(itertools.product(*[range(c) for c in outer_inner_cards_a[0]]))
        ntd_b_sd_assignments = list(itertools.product(*[range(c) for c in outer_inner_cards_b[0]]))

        full_sd_a_default_sub_table = _make_dense_default_probs_dict(outer_inner_cards_a[1], default)
        full_sd_b_default_sub_table = _make_dense_default_probs_dict(outer_inner_cards_b[1], default)

        # Having only dense default table is not good enough, as this will still not cause the missing
        # default values in the sub tables to be used to compute potentially none default result values.
        # we therefore need to make these sub tables dense.
        ntd_a = _make_inner_dense(ntd_a, outer_inner_cards_a, default)
        ntd_b = _make_inner_dense(ntd_b, outer_inner_cards_b, default)

    elif default_rules == 'both':
        # The result will only be default if the values in both ntd_a are default. This means
        # that we have to extend the assignments in the symmetric_difference_combinations to
        # those the full set of all possible assignments where either ntd_a is not default or
        # ntd_b is not default, because we need to calculate these results explicitly.
        # For example:
        # ntd_a:
        #    a b
        #    0 0  v
        #   (1 0) d
        # ntd_b:
        #    a c
        #    0 0  v
        #    1 0  v
        # The full set of resulting operations would be:
        # a b c
        # 0 0 0 vv
        # 0 0 1 vd
        # 0 1 0 dv
        # 0 1 1 dd = d
        # 1 0 0 dv
        # 1 0 1 dd = d
        # 1 1 0 dv
        # 1 1 1 dd = d
        # But because a:1 is not in ntd_a, this will not be calculated automatically and we therefore need to add this
        # entry. We do this by adding default sub tables below.
        outer_assignments_a = set(ntd_a.keys())
        outer_assignments_b = set(ntd_b.keys())
        ntd_a_sd_assignments = list(outer_assignments_a.union(outer_assignments_b))
        ntd_b_sd_assignments = copy.deepcopy(ntd_a_sd_assignments)

        full_sd_a_default_sub_table = _make_dense_default_probs_dict(outer_inner_cards_a[1], default)
        full_sd_b_default_sub_table = _make_dense_default_probs_dict(outer_inner_cards_b[1], default)

    # TODO: This is strange - improve.
    # TODO: See if any performance gains are possible with 'left' and 'right' flags and add them if so.
    elif default_rules == 'any':
        pass  # no further processing necessary
    symmetric_difference_combinations = itertools.product(*[ntd_a_sd_assignments, ntd_b_sd_assignments])

    for sd_assign_a, sd_assign_b in symmetric_difference_combinations:
        joined_sd_assignment = sd_assign_a + sd_assign_b

        # TODO: Add special, pre-computed result if both 0.
        common_vars_subtable_a = ntd_a.get(sd_assign_a, full_sd_a_default_sub_table)
        common_vars_subtable_b = ntd_b.get(sd_assign_b, full_sd_b_default_sub_table)

        # TODO: extend same_scope_binary_operation functionality to allow table completion given flag.
        subtable_result = same_scope_binary_operation(common_vars_subtable_a,
                                                      common_vars_subtable_b,
                                                      func, default)
        resulting_factor_table[joined_sd_assignment] = subtable_result
    return resulting_factor_table


class SparseCategorical(Factor):
    """
    A class for instantiating sparse tables with log probabilities.
    """

    def __init__(self, var_names, cardinalities, log_probs_table=None, probs_table=None, default_log_prob=-np.inf):
        """
        Construct a SparseLogTable. Either log_probs_table or probs_table should be supplied.

        :param var_names: The variable names.
        :type var_names: str list
        :param log_probs_table: A dictionary with assignments (tuples) as keys and probabilities as values.
            Missing assignments are assumed to have zero probability.
        :type
        :param log_probs_table: A dictionary with assignments (tuples) as keys and log probabilities as values.
            Missing assignments are assumed to have -inf log-probability (zero probability).
        Example:
            >>> var_names = ['rain','slip']
            >>> probs_table = {(0,0):0.8,
            >>>                (0,1):0.2,
            >>>                (1,0):0.4,
            >>>                (1,1):0.6}
            >>> cardinalities = [2,2]
            >>> table = SparseCategorical(log_probs_table=log_probs_table,
            >>>                           var_names=var_names,
            >>>                           cardinalities=cardinalities)
        """
        # TODO: add check that assignment lengths are consistent with var_names
        # TODO: add check that cardinalities are consistent with assignments
        super().__init__(var_names=var_names)
        if len(cardinalities) != len(var_names):
            raise ValueError('The cardinalities and var_names lists should be the same length.')
        if (log_probs_table is None) and (probs_table is None):
            raise ValueError('Either log_probs_table or probs_table must be specified')
        if log_probs_table is None:
            log_probs_table = {assignment: np.log(prob) for assignment, prob in probs_table.items()}
        self.log_probs_table = copy.deepcopy(log_probs_table)
        self.var_cards = dict(zip(var_names, cardinalities))
        self.cardinalities = cardinalities
        self.default_log_prob = default_log_prob

    # TODO: Improve this to take missing assignments into account. Alternatively: add functionality to sparsify factor
    #  when probs turn to 0.
    # TODO: Add variable order sorting
    def equals(self, factor):
        """
        Check if this factor is the same as another factor.

        :param factor: The other factor to compare to.
        :type factor: SparseCategorical
        :return: The result of the comparison.
        :rtype: bool
        """
        factor_ = factor
        if not isinstance(factor_, SparseCategorical):
            raise ValueError(f'factor must be of SparseLogTable type but has type {type(factor)}')
            raise ValueError(f'factor must be of SparseLogTable type but has type {type(factor)}')

        if set(self.var_names) != set(factor_.var_names):
            return False

        # var sets are the same
        if self.var_names != factor.var_names:
            factor_ = factor.reorder(self.var_names)
        # factors now have same variable order

        # Check the values for every non-default assignment of self
        for assign, self_log_prob in self.log_probs_table.items():

            if assign not in factor_.log_probs_table:
                if self_log_prob != factor_.default_log_prob:
                    return False
            else:
                factor_log_prob = factor_.log_probs_table[assign]
                if not np.isclose(factor_log_prob, self_log_prob):
                    return False
        # everywhere that self has non default values, factor has the same values.

        # TODO: improve efficiency here (there could be a lot of duplication with the above loop)
        # Check the values for every non-default assignment of factor
        for assign, factor_log_prob in factor_.log_probs_table.items():

            if assign not in self.log_probs_table:
                if factor_log_prob != self.default_log_prob:
                    return False
            else:
                self_log_prob = self.log_probs_table[assign]
                if not np.isclose(self_log_prob, factor_log_prob):
                    return False

        # If all possible assignments have not been checked - check that the default values are the same
        if not self._is_dense() and not factor._is_dense():
            if self.default_log_prob != factor.default_log_prob:
                return False

        return True

    def copy(self):
        """
        Make a copy of this factor.
        :return: The copy of this factor.
        :rtype: SparseCategorical
        """
        return SparseCategorical(var_names=self.var_names.copy(),
                                 log_probs_table=copy.deepcopy(self.log_probs_table),
                                 cardinalities=copy.deepcopy(self.cardinalities),
                                 default_log_prob=self.default_log_prob)

    @staticmethod
    def _get_shared_order_smaller_vars(smaller_vars, larger_vars):
        """
        larger_vars = ['a', 'c', 'd', 'b']
        smaller_vars = ['c', 'e', 'b']
        return ['c', 'b']
        """
        shared_vars = [v for v in smaller_vars if v in larger_vars]
        remaining_smaller_vars = list(set(larger_vars) - set(shared_vars))
        smaller_vars_new_order = shared_vars + remaining_smaller_vars
        return smaller_vars_new_order

    @staticmethod
    def _intersection_has_same_order(larger_vars, smaller_vars):
        """
        Check if the intersection of two lists has the same order in both lists.
        Will return true if either list is empty? SHOULD THIS BE THE CASE?
        """
        indices_of_smaller_in_larger = [larger_vars.index(v) for v in smaller_vars if v in larger_vars]
        if sorted(indices_of_smaller_in_larger) == indices_of_smaller_in_larger:
            return True
        return False

    # TODO: change back to log form
    def marginalize(self, vrs, keep=False):
        """
        Sum out variables from this factor.
        :param vrs: (list) a subset of variables in the factor's scope
        :param keep: Whether to keep or sum out vrs
        :return: The resulting factor.
        :rtype: SparseCategorical
        """

        vars_to_keep = super().get_marginal_vars(vrs, keep)
        vars_to_sum_out = [v for v in self.var_names if v not in vars_to_keep]
        nested_table, nested_table_vars = _get_nested_sorted_probs(new_variables_order_outer=vars_to_keep,
                                                                   new_variables_order_inner=vars_to_sum_out,
                                                                   old_variable_order=self.var_names,
                                                                   old_assign_probs=self.log_probs_table)
        result_table = dict()
        for l1_assign, log_probs_table in nested_table.items():
            prob = special.logsumexp(list(log_probs_table.values()))
            result_table[l1_assign] = prob

        result_var_cards = copy.deepcopy(self.var_cards)
        for v in vars_to_sum_out:
            del result_var_cards[v]
        cardinalities = list(result_var_cards.values())
        return SparseCategorical(var_names=vars_to_keep, log_probs_table=result_table,
                                 cardinalities=cardinalities)

    def reduce(self, vrs, values):
        """
        Observe variables to have certain values and return reduced table.
        :param vrs: (list) The variables.
        :param values: (tuple or list) Their values
        :return: The resulting factor.
        :rtype: SparseCategorical
        """

        vars_unobserved = [v for v in self.var_names if v not in vrs]
        nested_table, nested_table_vars = _get_nested_sorted_probs(new_variables_order_outer=vrs,
                                                                   new_variables_order_inner=vars_unobserved,
                                                                   old_variable_order=self.var_names,
                                                                   old_assign_probs=self.log_probs_table)
        result_table = nested_table[tuple(values)]
        result_var_cards = copy.deepcopy(self.var_cards)
        for v in vrs:
            del result_var_cards[v]

        cardinalities = list(result_var_cards.values())
        return SparseCategorical(var_names=vars_unobserved, log_probs_table=result_table,
                                 cardinalities=cardinalities)

    def _assert_consistent_cardinalities(self, factor):
        """
        Assert that the variable cardinalities are consistent between two factors.

        :param factor:
        """
        for var in self.var_names:
            if var in factor.var_cards:
                error_msg = f'Error: inconsistent variable cardinalities: {factor.var_cards}, {self.var_cards}'
                assert self.var_cards[var] == factor.var_cards[var], error_msg

    def multiply(self, factor):
        """
        Multiply this factor with another factor and return the result.

        :param factor: The factor to multiply with.
        :type factor: SparseCategorical
        :return: The factor product.
        :rtype: SparseCategorical
        """
        return self._apply_binary_operator(factor, operator.add, default_rules='any')

    def cancel(self, factor):
        """
        Almost like divide, but with a special rule that ensures that division of zeros by zeros results in zeros.
        :param factor: The factor to divide by.
        :type factor: SparseCategorical
        :return: The factor quotient.
        :rtype: SparseCategorical
        """

        def special_divide(a, b):
            if (a == -np.inf) and (b == -np.inf):
                return -np.inf
            else:
                return a - b

        return self._apply_binary_operator(factor, special_divide, default_rules='any')

    def divide(self, factor):
        """
        Divide this factor by another factor and return the result.

        :param factor: The factor to divide by.
        :type factor: SparseCategorical

        :return: The factor quotient.
        :rtype: SparseCategorical
        """
        return self._apply_binary_operator(factor, operator.sub, default_rules='none')

    def argmax(self):
        """
        Get the Categorical assignment (vector value) that maximises the factor potential.

        :return: The argmax assignment.
        :rtype: int list
        """
        return max(self.log_probs_table.items(), key=operator.itemgetter(1))[0]

    def _apply_to_probs(self, func, include_assignment=False):
        for assign, prob in self.log_probs_table.items():
            if include_assignment:
                self.log_probs_table[assign] = func(prob, assign)
            else:
                self.log_probs_table[assign] = func(prob)

    def _apply_binary_operator(self, factor, operator_function, default_rules='none'):
        """
        Apply a binary operator function f(self.factor, factor) and return the result

        :param factor: The other factor to use in the binary operation.
        :type factor: SparseCategorical
        :param default_rules: The rules for when a calculation results will result in a default value (optional).
            This can help speed up this function. The possible values are as follows:
                'left' : If ntd_a has a default value, the result will always be default.
                'right' : If ntd_b has a default value, the result will always be default.
                'any' : If either ntd_a or ntd_b has a default value, the result will always be default.
                'both': Only if both ntd_a and ntd_b has a default value, the result will always be default.
                'none': No combination of default or non-default values is guarenteed to result in a default value
        If this parameter is not specified, 'none' will be used to ensure correct, albeit slower computation.
        :return: The resulting factor.
        :rtype: SparseCategorical
        """
        if not isinstance(factor, SparseCategorical):
            raise ValueError(f'factor must be of SparseLogTable type but has type {type(factor)}')
        self._assert_consistent_cardinalities(factor)
        intersection_vars = list(set(self.var_names).intersection(set(factor.var_names)))
        intersection_vars = sorted(intersection_vars)

        remaining_a_vars = list(set(self.var_names) - set(intersection_vars))
        ntd_a, _ = _get_nested_sorted_probs(new_variables_order_outer=remaining_a_vars,
                                            new_variables_order_inner=intersection_vars,
                                            old_variable_order=self.var_names,
                                            old_assign_probs=self.log_probs_table)

        remaining_b_vars = list(set(factor.var_names) - set(intersection_vars))
        ntd_b, _ = _get_nested_sorted_probs(new_variables_order_outer=remaining_b_vars,
                                            new_variables_order_inner=intersection_vars,
                                            old_variable_order=factor.var_names,
                                            old_assign_probs=factor.log_probs_table)

        # TODO: Add this functionality
        if self.default_log_prob != factor.default_log_prob:
            error_msg = 'Cases where self.default_value and factor.default_value differ are not yet supported.'
            raise NotImplementedError(error_msg)
        default_log_prob = self.default_log_prob

        def vars_to_cards(factor, var_names):
            return [factor.var_cards[v] for v in var_names]

        outer_inner_cards_a = [vars_to_cards(self, remaining_a_vars),
                               vars_to_cards(self, intersection_vars)]
        outer_inner_cards_b = [vars_to_cards(factor, remaining_b_vars),
                               vars_to_cards(factor, intersection_vars)]

        result_ntd = any_scope_binary_operation(ntd_a, outer_inner_cards_a,
                                                ntd_b, outer_inner_cards_b,
                                                operator_function, default_log_prob,
                                                default_rules=default_rules)
        flattened_result_table = flatten_ntd(result_ntd)
        result_var_names = remaining_a_vars + remaining_b_vars + intersection_vars
        result_var_cards = {**self.var_cards, **factor.var_cards}
        result_cardinalities = [result_var_cards[v] for v in result_var_names]
        result_default_log_prob = operator_function(default_log_prob, default_log_prob)
        return SparseCategorical(var_names=result_var_names, log_probs_table=flattened_result_table,
                                 cardinalities=result_cardinalities, default_log_prob=result_default_log_prob)

    def normalize(self):
        """
        Return a normalized copy of the factor.
        :return: The normalized factor.
        """

        factor_copy = self.copy()
        logz = special.logsumexp(list(factor_copy.log_probs_table.values()))
        for assign, prob in factor_copy.log_probs_table.items():
            factor_copy.log_probs_table[assign] = prob - logz
        return factor_copy

    def _is_dense(self):
        """
        Check if all factor assignments have non-default values.
        :return: The result of the check.
        :rtype: Bool
        """
        num_assignments = len(self.log_probs_table)
        max_possible_assignments = np.prod(self.cardinalities)
        if num_assignments == max_possible_assignments:
            return True
        return False

    @property
    def is_vacuous(self):
        """
        Check if this factor is vacuous (i.e uniform).
        :return: Whether the factor is vacuous or not.
        :rtype: bool
        """
        if self.distance_from_vacuous() < 1e-10:
            return True
        return False

    def kl_divergence(self, factor, normalize_factor=True):
        """
        Get the KL-divergence D_KL(P||Q) = D_KL(self||factor) between a normalized version of this factor and another factor.
        Reference https://infoscience.epfl.ch/record/174055/files/durrieuThiranKelly_kldiv_icassp2012_R1.pdf, page 1.

        :param factor: The other factor
        :type factor: Gaussian
        :param normalize_factor: Whether or not to normalize the other factor before computing the KL-divergence.
        :type normalize_factor: bool
        :return: The Kullback-Leibler divergence
        :rtype: float
        """

        def kld_from_log_elements(log_pi, log_qi):
            if log_pi == -np.inf:
                # lim_{p->0} p*log(p/q) = 0
                # lim_{p->0} p*log(p/q) = 0, with q = p
                # So if p_i = 0, kld_i = 0
                return 0.0
            else:
                kld_i = np.exp(log_pi) * (log_pi - log_qi)
                return kld_i

        log_P = self.normalize()
        log_Q = factor
        if normalize_factor:
            log_Q = factor.normalize()

        kld_factor = SparseCategorical._apply_binary_operator(log_P,
                                                              log_Q,
                                                              kld_from_log_elements,
                                                              default_rules='none')  # TODO: check this

        klds = list(kld_factor.log_probs_table.values())
        kld = np.sum(klds)

        if kld < 0.0:
            if np.isclose(kld, 0.0, atol=1e-5):
                #  this is fine (numerical error)
                return 0.0
            print('\nnormalize_factor = ', log_P)
            print('self = ')
            self.show()
            print('normalized_self = ')
            log_P.show()
            print('\nfactor = ')
            log_Q.show()
            raise ValueError(f'Negative KLD: {kld}')
        return kld

    def distance_from_vacuous(self):
        """
        Get the Kullback-Leibler (KL) divergence between this factor and a uniform copy of it.

        :return: The KL divergence.
        :rtype: float
        """
        # make uniform copy
        uniform_factor = self.copy()
        cards = list(uniform_factor.var_cards.values())
        uniform_log_prob = -np.log(np.product(cards))
        uniform_factor._apply_to_probs(lambda x: uniform_log_prob)
        kl = self.kl_divergence(uniform_factor, normalize_factor=False)
        if kl < 0.0:
            raise ValueError(f"kl ({kl}) < 0.0")
            self.show()
        return kl

    def potential(self, vrs, assignment):
        """
        Get the value of the factor for a specific assignment.

        :param assignment: The assignment
        :return: The value
        """
        assert set(vrs) == set(self.var_names), 'variables (vrs) do not match factor variables.'
        vrs_to_var_names_indices = [self.var_names.index(v) for v in vrs]
        var_names_order_assignments = tuple([assignment[i] for i in vrs_to_var_names_indices])
        return self.log_probs_table[var_names_order_assignments]

    def _to_df(self):
        log_probs_table = self.log_probs_table
        var_names = self.var_names
        df = pd.DataFrame.from_dict(log_probs_table.items()).rename(columns={0: 'assignment', 1: 'log_prob'})
        df[var_names] = pd.DataFrame(df['assignment'].to_list())
        df.drop(columns=['assignment'], inplace=True)
        return df

    def show(self, exp_log_probs=True):
        """
        Print the factor.

        :param exp_log_probs: Whether or no to exponentiate the log probabilities (to display probabilities instead of
        log-probabilities)
        :type exp_log_probs: bool
        """
        prob_string = 'log(prob)'
        if exp_log_probs:
            prob_string = 'prob'
        print(self.var_names, ' ', prob_string)
        for assignment, log_prob in self.log_probs_table.items():
            prob = log_prob
            if exp_log_probs:
                prob = np.exp(prob)
            print(assignment, ' ', prob)

    def reorder(self, new_var_names_order):
        """
        Reorder categorical table variables to a new order and reorder the associated probabilities
        accordingly.
        :param new_var_names_order: The new variable order.
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

        new_order_indices = [self.var_names.index(var) for var in new_var_names_order]
        new_log_probs_table = dict()
        for assignment, value in self.log_probs_table.items():
            reordered_assignment = tuple(assignment[i] for i in new_order_indices)
            new_log_probs_table[reordered_assignment] = value
        reordered_cardinalities = [self.cardinalities[i] for i in new_order_indices]

        return SparseCategorical(var_names=new_var_names_order,
                                 log_probs_table=new_log_probs_table,
                                 cardinalities=reordered_cardinalities)


class SparseCategoricalTemplate(FactorTemplate):

    def __init__(self, log_probs_table, var_templates):
        """
        Create a Categorical factor template.

        :param log_probs_table: The log_probs_table that specifies the assignments and values for the template.
        :type log_probs_table: tuple:float dict
        :param var_templates: A list of formattable strings.
        :type var_templates: str list

        log_probs_table example:
        {(0, 0): 0.1,
         (0, 1): 0.3,
         (1, 0): 0.1,
         (1, 1): 0.5}
        """
        # TODO: Complete and improve docstring.
        super().__init__(var_templates=var_templates)
        self.log_probs_table = copy.deepcopy(log_probs_table)

    def make_factor(self, format_dict=None, var_names=None):
        """
        Make a factor with var_templates formatted by format_dict to create specific var names.

        :param format_dict: The dictionary to be used to format the var_templates strings.
        :type format_dict: str dict
        :return: The instantiated factor.
        :rtype: SparseCategorical
        """
        if format_dict is not None:
            assert var_names is None
            var_names = [vt.format(**format_dict) for vt in self._var_templates]
        cardinalities = list(self.var_cards.values())
        return SparseCategorical(log_probs_table=copy.deepcopy(self.log_probs_table),
                                 var_names=var_names, cardinalities=cardinalities)
