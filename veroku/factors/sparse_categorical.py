"""
A module for instantiating sparse tables with log probabilities.
"""

# System imports
import copy
import operator

# Third-party imports
import numpy as np
from scipy import special
import itertools

# Local imports
from veroku.factors._factor import Factor
from veroku.factors._factor_template import FactorTemplate


def check_input_rules(values, rules):
    """
    A helper function for special rules for complex table operations.
    :param values: The list of values (left, right)
    :param rules: The list of rules (left_match_rule, right_match_rule), where each rule can either be a function or a constant
    """
    results = [False, False]

    for i in range(2):
        rule = rules[i]
        actual_value = values[i]
        if callable(rule):
            results[i] = rule(actual_value)
        else:
            expected_value = rule
            results[i] = expected_value == actual_value
    return all(results)


class SparseCategorical(Factor):
    """
    A class for instantiating sparse tables with log probabilities.
    """
    def __init__(self, var_names, cardinalities, log_probs_table=None, probs_table=None, default_value=-np.inf):
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
        #if not isinstance(cardinalities, list):
        #    raise ValueError('cardinalies should have type list')
        if len(cardinalities) != len(var_names):
            raise ValueError('The cardinalities and var_names lists should be the same length.')
        if (log_probs_table is None) and (probs_table is None):
            raise ValueError('Either log_probs_table or probs_table must be specified')
        if log_probs_table is None:
            log_probs_table = {assignment: np.log(prob) for assignment, prob in probs_table.items()}
        self.log_probs_table = copy.deepcopy(log_probs_table)
        self.var_cards = dict(zip(var_names, cardinalities))
        self.cardinalities = cardinalities
        self.default_value = default_value

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

        if set(self.var_names) != set(factor_.var_names):
            return False

        # var sets are the same
        if self.var_names != factor.var_names:
            factor_ = factor.reorder(self.var_names)
        # factors now have same variable order

        # Check the values for every non-default assignment of self
        for assign, self_prob in self.log_probs_table.items():
            if assign not in factor_.log_probs_table:
                if self_prob != factor_.default_value:
                    return False
            elif not np.isclose(factor_.log_probs_table[assign], self_prob):
                return False
        # everywhere that self has non default values, factor has the same values.

        # TODO: improve efficiency here (there could be a lot of duplication with the above loop)
        # Check the values for every non-default assignment of factor
        for assign, factor_prob in factor_.log_probs_table.items():
            if assign not in self.log_probs_table:
                if factor_prob != self.default_value:
                    return False
            elif not np.isclose(self.log_probs_table[assign], factor_prob):
                return False

        # If all possible assignments have not been checked - check that the default values are the same
        if not self._is_dense() and not factor._is_dense():
            if self.default_value != factor.default_value:
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
                                 cardinalities=copy.deepcopy(self.cardinalities))

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
        nested_table, nested_table_vars = SparseCategorical._get_nested_sorted_probs(new_variables_order_outer=vars_to_keep,
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
        nested_table, nested_table_vars = SparseCategorical._get_nested_sorted_probs(new_variables_order_outer=vrs,
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
        return self._complex_table_operation(factor, operator.add)

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

        return self._complex_table_operation(factor, special_divide)

    def divide(self, factor):
        """
        Divide this factor by another factor and return the result.

        :param factor: The factor to divide by.
        :type factor: SparseCategorical
        :return: The factor quotient.
        :rtype: SparseCategorical
        """
        return self._complex_table_operation(factor, operator.sub)

    def argmax(self):
        """
        Get the Categorical assignment (vector value) that maximises the factor potential.

        :return: The argmax assignment.
        :rtype: int list
        """
        return max(self.log_probs_table.items(), key=operator.itemgetter(1))[0]

    @staticmethod
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

    #TODO: make this more general / robust
    def _complex_table_operation(self, factor_b, func):
        """
        Operate on a pair of tables which can be sparse and have any combination of overlapping or disjoint variable sets.
        :param factor_a:
        :param factor_b:
        :param func: The function to apply on pairs of corresponding probabilities in the two tables.
        :return:
        """
        # NOTE: This function will skip operations where one of the factors has default values and is therefore only
        # suitable for operators such as multiply, where if either of the factors have a default value, the result will
        # also be default, or cancel, where it is typically the case that the larger factor (the cluster potentials)
        # will have default values at assignments corresponding to that of the smaller factor (the message).
        # TODO: Investigate the above note further.

        factor_a = self.copy()
        # TODO: Complete and improve docstring.
        if not isinstance(factor_b, SparseCategorical):
            raise ValueError(f'factor_b must be of SparseLogTable type but has type {type(factor)}')
        factor_a._assert_consistent_cardinalities(factor_b)

        factor_a_table = factor_a.log_probs_table
        factor_b_table = factor_b.log_probs_table
        factor_a_vars = factor_a.var_names
        factor_b_vars = factor_b.var_names

        shared_vars_factor_b_order = [v for v in factor_a_vars if v in factor_b_vars]
        remaining_factor_b_vars = list(set(factor_b_vars) - set(shared_vars_factor_b_order))
        remaining_factor_b_var_cards = [factor_b.cardinalities[factor_b_vars.index(v)] for v in remaining_factor_b_vars]

        new_order_factor_b_vars = remaining_factor_b_vars + shared_vars_factor_b_order
        nested_factor_b_table, factor_b_vars = SparseCategorical._get_nested_sorted_probs(remaining_factor_b_vars,
                                                                                          shared_vars_factor_b_order,
                                                                                          factor_b_vars, factor_b_table)
        factor_b_vars = copy.deepcopy(new_order_factor_b_vars)

        # use the nested dictionary (of sub-assignment prob dictionaries)
        result_table = dict()
        for assign_l1, factor_b_sub_table in nested_factor_b_table.items():
            result_l2_table = SparseCategorical._basic_table_operation(factor_a_vars, factor_a_table,
                                                                       factor_b_vars, factor_b_sub_table,
                                                                       func, switched=False)
            for results_assign, prob in result_l2_table.items():
                result_table[assign_l1 + results_assign] = prob
        result_vars = remaining_factor_b_vars + factor_a_vars
        result_cardinalities = remaining_factor_b_var_cards + factor_a.cardinalities
        return SparseCategorical(var_names=result_vars, log_probs_table=result_table, cardinalities=result_cardinalities)

    @staticmethod
    def _basic_table_operation(larger_table_vars,
                               larger_table,
                               smaller_table_vars,
                               smaller_table,
                               _operator,
                               switched):
        """
        Efficiently perform operations on corresponding (as indicted by the assignments and variables names)
        elements between larger_table_probs and smaller_table_probs. The smaller table variables must only
        contain variables that is also in the larger and must be sorted to have the same order as in the larger
        (although there can be variables in between in the larger). Also both can be sparse.

        Note: the variables will have the same order as larger_table_vars and a new variable name list is therefore not
              returned.

        :param larger_table: A dictionary of assignment tuples and corresponding probabilities.
        :type larger_table: SparseCategorical
        :param smaller_table: A dictionary of assignment tuples and corresponding probabilities.
        :type smaller_table: SparseCategorical

        :Example:
        larger_table_vars = ['a', 'b', 'c']
        larger_table = {(0, 0, 0): 0.5,
                        (0, 1, 1): 0.2,
                        (1, 1, 0): 0.3}
        smaller_table_vars = ['a', c']
        smaller_table = {(0, 0): 0.2,
                        (0, 1): 0.5,
                        (1, 1): 0.1}
        """

        if not SparseCategorical._intersection_has_same_order(larger_table_vars, smaller_table_vars):
            raise ValueError('Variables must have same relative order.')
        shared_indices_in_larger = [larger_table_vars.index(var) for var in smaller_table_vars if var in larger_table_vars]

        result_probs = dict()
        for lt_assign, lt_prob in larger_table.items():
            assign_smaller = tuple([lt_assign[i] for i in shared_indices_in_larger])
            if assign_smaller in smaller_table:
                rt_prob = smaller_table[assign_smaller]
            else:
                # use default zero prob (-inf log prob)
                rt_prob = -np.inf
            if not switched:
                result_prob = _operator(lt_prob, rt_prob)
            else:
                result_prob = _operator(rt_prob, lt_prob)
            #  TODO: add this and update tests.
            if result_prob != -np.inf:
                result_probs[lt_assign] = result_prob
        return result_probs

    def _apply_to_probs(self, func, include_assignment=False):
        for assign, prob in self.log_probs_table.items():
            if include_assignment:
                self.log_probs_table[assign] = func(prob, assign)
            else:
                self.log_probs_table[assign] = func(prob)

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

        # TODO: check that this is correct. esp with zeroes.
        kld_factor = SparseCategorical._complex_table_operation(log_P,
                                                                log_Q,
                                                                kld_from_log_elements)

        klds = list(kld_factor.log_probs_table.values())

        # get assignments plog(p/q) where p is not default, but q is. These have not been accounted for above.
        # TODO: find a better solution for this (see TOOD in _complex_table_operation)

        #TODO: fix this
        #q_assignments = list(log_P.log_probs_table.keys())
        #for p_assign in log_P.log_probs_table.keys():
        #    p_to_q_translation_indices = [self.var_names.index(v) for v in factor.var_names]
        #    p_assign_in_q_order = [p_assign[i] for i in p_to_q_translation_indices]
        #    if p_assign_in_q_order not in q_assignments:
        #        klds.append(np.inf)

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