"""
A module for instantiating sparse tables with log probabilities.
"""

# System imports
import copy
import operator

# Third-party imports
import numpy as np
import scipy.misc
import itertools

# Local imports
from veroku.factors.factor import Factor
from veroku.factors.factor_template import FactorTemplate


# TODO: consider removing some unused functions
class Categorical(Factor):
    """
    A class for instantiating sparse tables with log probabilities.
    """
    def __init__(self, var_names, log_probs_table, cardinalities):
        """
        Construct a SparseLogTable
        :param var_names: (list) The variable names.
        :param probs: (dictionary) A dictionary with assignments (tuples) as keys and log probabilities as values.
                      Missing assignments are assumed to have zero probability.
        """
        # TODO: add check that assignment lengths are consistent with var_names
        # TODO: add check that cardinalities are consistent with assignments
        super().__init__(var_names=var_names)
        self.log_probs_table = copy.deepcopy(log_probs_table)
        self.var_cards = dict(zip(var_names, cardinalities))

    # TODO: improve this to take missing assignments into account. Alternatively: add functionality to sparsify factor when probs turn to 0
    # TODO: Add variable order sorting
    def equals(self, factor):
        if not isinstance(factor, Categorical):
            raise ValueError(f'factor must be of SparseLogTable type but has type {type(factor)}')
        if self.var_names != factor.var_names:
            return False
        for assign, prob in self.log_probs_table.items():
            if assign not in factor.log_probs_table:
                return False
            elif not np.isclose(factor.log_probs_table[assign], prob):
                return False
        return True

    @classmethod
    def from_table_factor(cls, table_factor):
        """
        Construct an equivalent SparseLogTable from a table factor
        :param table_factor:
        :return:
        """
        factor_df = table_factor.get_table_as_dataframe()
        log_probs_table = dict()
        for assign, prob_ in factor_df.iterrows():
            prob = prob_.values[0]
            if prob != 0.0:
                log_prob = np.log(prob)
                log_probs_table[assign] = log_prob
        cards = factor_df.index.levshape
        var_names = factor_df.index.names
        return cls(var_names=var_names, log_probs_table=log_probs_table, cardinalities=cards)

    def copy(self):
        """
        Make a copy of this factor.
        :return: (SparseLogTable) The copy.
        """
        return Categorical(var_names=self.var_names.copy(),
                           log_probs_table=copy.deepcopy(self.log_probs_table),
                           cardinalities=self.var_cards.values())

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
    def marginalise(self, vrs, keep=False):
        """
        Sum out variables from this factor.
        :param vrs: (list) a subset of variables in the factor's scope
        :param keep: Whether to keep or sum out vrs
        :return: (SparseLogTable) the resulting factor.
        """

        vars_to_keep = super().get_marginal_vars(vrs, keep)
        vars_to_sum_out = [v for v in self.var_names if v not in vars_to_keep]
        nested_table, nested_table_vars = Categorical.get_nested_sorted_probs(new_variables_order_outer=vars_to_keep,
                                                                              new_variables_order_inner=vars_to_sum_out,
                                                                              old_variable_order=self.var_names,
                                                                              old_assign_probs=self.log_probs_table)
        result_table = dict()
        for l1_assign, log_probs_table in nested_table.items():
            prob = scipy.misc.logsumexp(list(log_probs_table.values()))
            result_table[l1_assign] = prob

        result_var_cards = copy.deepcopy(self.var_cards)
        for v in vars_to_sum_out:
            del result_var_cards[v]

        return Categorical(var_names=vars_to_keep, log_probs_table=result_table,
                           cardinalities=result_var_cards.values())

    def observe(self, vrs, values):
        """
        Observe variables to have certain values and return reduced table.
        :param vrs: (list) The variables.
        :param values: (tuple or list) Their values
        :return: (SparseLogTable) The resulting factor.
        """

        vars_unobserved = [v for v in self.var_names if v not in vrs]
        nested_table, nested_table_vars = Categorical.get_nested_sorted_probs(new_variables_order_outer=vrs,
                                                                              new_variables_order_inner=vars_unobserved,
                                                                              old_variable_order=self.var_names,
                                                                              old_assign_probs=self.log_probs_table)
        result_table = nested_table[tuple(values)]
        result_var_cards = copy.deepcopy(self.var_cards)
        for v in vrs:
            del result_var_cards[v]

        return Categorical(var_names=vars_unobserved, log_probs_table=result_table,
                           cardinalities=result_var_cards.values())

    def assert_consistent_cardinalities(self, factor):
        """
        Assert that the variable cardinalities are consistent between two factors.
        :param factor:
        """
        for var in self.var_names:
            if var in factor.var_cards:
                error_msg = f'Error: inconsistent variable cardinalities: {factor.var_cards}, {self.var_cards}'
                assert self.var_cards[var] == factor.var_cards[var], error_msg

    def absorb(self, factor):
        """
        Multiply this factor with `factor` and return the result.
        :param factor: (SparseLogTable) The factor to multiply with.
        :return: (SparseLogTable) The resulting factor.
        """
        if not isinstance(factor, Categorical):
            raise ValueError(f'factor must be of SparseLogTable type but has type {type(factor)}')
        self.assert_consistent_cardinalities(factor)
        result_table, result_vars = Categorical.complex_table_operation(self.var_names, self.log_probs_table,
                                                                        factor.var_names, factor.log_probs_table,
                                                                        operator.add)
        result_var_cards = copy.deepcopy(self.var_cards)
        result_var_cards.update(factor.var_cards)
        return Categorical(var_names=result_vars, log_probs_table=result_table,
                           cardinalities=result_var_cards.values())

    def cancel(self, factor):
        """
        Divide this factor by `factor` and return the result.
        :param factor: (SparseLogTable) The factor to divide by.
        :return: (SparseLogTable) The resulting factor.
        """
        if not isinstance(factor, Categorical):
            raise ValueError(f'factor must be of SparseLogTable type but has type {type(factor)}')
        self.assert_consistent_cardinalities(factor)
        result_table, result_vars = Categorical.complex_table_operation(self.var_names, self.log_probs_table,
                                                                        factor.var_names, factor.log_probs_table,
                                                                        operator.sub)
        result_var_cards = copy.deepcopy(self.var_cards)
        result_var_cards.update(factor.var_cards)
        return Categorical(var_names=result_vars, log_probs_table=result_table,
                           cardinalities=result_var_cards.values())

    def argmax(self):
        return max(self.log_probs_table.items(), key=operator.itemgetter(1))[0]

    @staticmethod
    def get_nested_sorted_probs(new_variables_order_outer,
                                new_variables_order_inner,
                                old_variable_order, old_assign_probs):
        """
        Reorder probs to a new order and sort assignments.
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

    @staticmethod
    def complex_table_operation(vars_a, table_a, vars_b, table_b, func):
        """
        Operate on a pair of tables which can be sparse and have any combination of overlapping or disjoint variable sets.
        :param vars_a:
        :param table_a:
        :param vars_b:
        :param table_b:
        :param func: The function to apply on pairs of corresponding probabilities in the two tables.
        :return:
        """
        larger_table = table_a
        smaller_table = table_b
        larger_table_vars = vars_a
        smaller_table_vars = vars_b
        switched = False
        if len(table_a) < len(table_b):
            larger_table = table_b
            smaller_table = table_a
            larger_table_vars = vars_b
            smaller_table_vars = vars_a
            switched = True

        shared_order_smaller_vars = [v for v in larger_table_vars if v in smaller_table_vars]
        remaining_smaller_vars = list(set(smaller_table_vars) - set(shared_order_smaller_vars))

        new_order_smaller_table_vars = remaining_smaller_vars + shared_order_smaller_vars
        smaller_table, smaller_table_vars = Categorical.get_nested_sorted_probs(remaining_smaller_vars,
                                                                                shared_order_smaller_vars,
                                                                                smaller_table_vars, smaller_table)
        smaller_table_vars = copy.deepcopy(new_order_smaller_table_vars)

        # use the nested dictionary (of sub-assignment prob dictionaries)
        result_table = dict()
        for assign_l1, l2_table in smaller_table.items():
            result_l2_table = Categorical.basic_table_operation(larger_table_vars, larger_table,
                                                                smaller_table_vars, l2_table, func)
            for results_assign, prob in result_l2_table.items():
                # TODO: get better solution than this:
                if (func == operator.sub) and switched:
                    prob = -prob
                result_table[assign_l1 + results_assign] = prob
        result_vars = remaining_smaller_vars + larger_table_vars
        return result_table, result_vars

    @staticmethod
    def basic_table_operation(larger_table_vars,
                              larger_table,
                              smaller_table_vars,
                              smaller_table,
                              _opperator):
        """
        Efficiently perform operations on corresponding (as indicted by the assignments and variables names)
        elements between larger_table_probs and smaller_table_probs. The smaller table variables must only
        contain variables that is also in the larger and must be sorted to have the same order as in the larger
        (although there can be variables in between in the larger). Also both can be sparse.

        Note: the variables will have the same order as larger_table_vars and a new variable name list is therefore not
              returned.

        :param larger_table: a dictionary of assignment tuples and corresponding probabilities
        :param smaller_table: a dictionary of assignment tuples and corresponding probabilities

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

        if not Categorical._intersection_has_same_order(larger_table_vars, smaller_table_vars):
            raise ValueError('Variables must have same relative order.')
        shared_indices_in_larger = [larger_table_vars.index(var) for var in smaller_table_vars if var in larger_table_vars]

        result_probs = dict()

        for lt_assign, lt_prob in larger_table.items():
            assign_smaller = tuple([lt_assign[i] for i in shared_indices_in_larger])
            if assign_smaller in smaller_table:
                rt_prob = smaller_table[assign_smaller]
                result_probs[lt_assign] = _opperator(lt_prob, rt_prob)

        return result_probs

    def apply_to_probs(self, func, include_assignment=False):
        for assign, prob in self.log_probs_table.items():
            if include_assignment:
                self.log_probs_table[assign] = func(prob, assign)
            else:
                self.log_probs_table[assign] = func(prob)

    def normalise(self):
        """
        Return a normalised copy of the factor.
        :return: The normalised factor.
        """

        factor_copy = self.copy()
        logz = scipy.misc.logsumexp(list(factor_copy.log_probs_table.values()))
        for assign, prob in factor_copy.log_probs_table.items():
            factor_copy.log_probs_table[assign] = prob - logz
        return factor_copy

    def kl_divergence(self, factor, normalise_factor=True):
        """
        Get D_KL(self||factor).
        D_KL(P|Q) = sum(P*log(P/Q))
        :param factor: The other factor
        :return: The Kullback-Leibler divergence
        """
        normalised_self = self.normalise()
        normalised_factor = factor
        if normalise_factor:
            normalised_factor = factor.normalise()
        # TODO: check that this is correct. esp with zeroes.
        logPdivQ = normalised_self.cancel(normalised_factor)
        normalised_self.apply_to_probs(np.exp)

        PlogPdivQ_table, _ = Categorical.complex_table_operation(normalised_self.var_names,
                                                                 normalised_self.log_probs_table,
                                                                 logPdivQ.var_names,
                                                                 logPdivQ.log_probs_table,
                                                                 operator.mul)
        kld = np.sum(list(PlogPdivQ_table.values()))
        if kld < 0.0:
            if np.isclose(kld, 0.0, atol=1e-5):
                #  this is fine (numerical error)
                return 0.0
            print('\nnormalise_factor = ', normalise_factor)
            print('self = ')
            self.show()
            print('\nfactor = ')
            normalised_factor.show()
            raise ValueError(f'Negative KLD: {kld}')
        return kld

    def distance_from_vacuous(self):
        """
        Get the Kullback-Leibler divergence between this factor and a uniform copy of it.
        :return:
        """
        # make uniform copy
        uniform_factor = self.copy()
        cards = list(uniform_factor.var_cards.values())
        uniform_log_prob = -np.log(np.product(cards))
        uniform_factor.apply_to_probs(lambda x: uniform_log_prob)

        return self.kl_divergence(uniform_factor, normalise_factor=False)

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
        Print the factor
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


class SparseLogTableTemplate(FactorTemplate):

    def __init__(self, log_probs_table, var_templates):
        """
        log_probs_table example:
        {(0, 0): 0.1,
         (0, 1): 0.3,
         (1, 0): 0.1,
         (1, 1): 0.5}
        """
        super().__init__(var_templates=var_templates)
        self.log_probs_table = copy.deepcopy(log_probs_table)

    def make_factor(self, format_dict=None, var_names=None):
        """
        Make a factor with var_templates formatted by format_dict to create specific var names.
        :param format_dict:
        :return:
        """
        if format_dict is not None:
            assert var_names is None
            var_names = [vt.format(**format_dict) for vt in self._var_templates]
        return Categorical(log_probs_table=copy.deepcopy(self.log_probs_table),
                           var_names=var_names, cardinalities=self.var_cards.values())


# Note: this function is old and still uses the list (instead of dictionary) format for probs
def reorder_probs(new_variable_order, old_variable_order, old_assign_probs):
    """
    Reorder probs to a new order and sort assignments.

    Example:
    old_variable_order = [a, b]
    new_variable_order = [b, a]

    a b P(a,b)  return    b a P(a,b)
    0 0  pa0b0            0 0  pa0b0
    0 1  pa0b1            0 1  pa1b0
    1 0  pa1b0            1 0  pa0b1
    1 1  pa1b1            1 1  pa1b1
    """
    new_order_indices = [new_variable_order.index(var) for var in old_variable_order]
    new_assign_probs = []
    for old_assign_prob_i in old_assign_probs:
        old_row_assignment = old_assign_prob_i[0]
        prob = old_assign_prob_i[1]
        new_row_assignment = [None] * len(old_row_assignment)
        for old_i, new_i in enumerate(new_order_indices):
            new_row_assignment[new_i] = old_row_assignment[old_i]
        new_assign_probs.append([tuple(new_row_assignment), prob])

    return sorted(new_assign_probs, key=lambda x: x[0])
