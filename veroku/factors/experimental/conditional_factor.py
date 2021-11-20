"""
A module for instantiating sparse tables with log probabilities.
"""

# Standard imports
import copy
import operator
from numbers import Number

# Third-party imports
import numpy as np
from scipy import special

# Local imports

from veroku.factors import _factor_utils
from veroku.factors.sparse_categorical import SparseCategorical, _get_nested_sorted_probs, _any_scope_binary_operation, _flatten_ntd
from veroku.factors.experimental.mixture_factor import MixtureFactor

# pylint: disable=protected-access


def _fast_copy_factor_table(table):
    """
    Copy a dictionary representation of a probability table faster than the standard deepcopy.

    :param dict table: A dictionary with the tuples of ints as keys and floats as values.
    :return: The copied table.
    """
    table_copy = {tuple(assign): value for assign, value in table.items()}
    return table_copy


#TODO: consider swapping this around so that SparseCategorical rather inherits from this class.
class ConditionalFactor(SparseCategorical):
    """
    A class for instantiating sparse tables with log probabilities.
    """

    def __init__(self, conditioning_var_names, cardinalities, factor_table=None, default_factor=None):
        """


        Construct a sparse categorical factor. Either log_probs_table or probs_table should be supplied.

        :param conditioning_var_names: The variable names.
        :type conditioning_var_names: str list
        :param cardinalities: The cardinalities of the variables (i.e, for three binrary variables: [2,2,2])
        :type cardinalities: int list
        :param factor_table: A dictionary with assignments (tuples) as keys and a factor corresponding to each key.
            Missing assignments are assumed to have the provided default log probability (with -inf default value).
        :type factor_table: dict
        :param default_factor: The default factor that missing values are assumed to have.
        """
        # TODO: add check that assignment lengths are consistent with var_names
        # TODO: add check that cardinalities are consistent with assignments

        all_factors_are_constants = True
        for _, value in factor_table.items():
            if not isinstance(value, Number):
                all_factors_are_constants = False
        if all_factors_are_constants:
            return SparseCategorical()

        self.default_factor = None
        conditional_var_names = list(factor_table.values())[0].var_names
        for _, factor in factor_table.items():
            assert factor.var_names == conditional_var_names

        if len(cardinalities) != len(conditioning_var_names):
            raise ValueError("The cardinalities and var_names lists should be the same length.")
        if default_factor is None:
            assert np.product(cardinalities) == len(factor_table), "factor table is not consistent with provided cardinalities."
        else:
            assert default_factor.var_names == conditional_var_names
            self.default_factor = default_factor.copy()
        assert len(set(conditioning_var_names).intersection(set(conditional_var_names))) == 0
        self.conditional_var_names = conditional_var_names
        var_names = conditioning_var_names + conditional_var_names

        # From Factor class
        if len(set(var_names)) != len(var_names):
            raise ValueError("duplicate variables in var_names: ", var_names)

        self._var_names = copy.deepcopy(var_names)
        if not isinstance(var_names, list):
            self._var_names = [var_names]

        self._dim = len(var_names)


        self.factor_table = _fast_copy_factor_table(factor_table)
        self.var_cards = dict(zip(conditioning_var_names, cardinalities))
        self.cardinalities = copy.deepcopy(cardinalities)


        self.conditioning_var_names = conditioning_var_names

    def _all_non_default_equal(self, other):
        """
        Check that all non default values in this factor are the same as the corresponding values in factor, where
        the two factors have the same variable scope.

        :param other: The other factor
        :return: The result of the check.
        :rtype: bool
        """

        for assign, self_factor_comp in self.factor_table.items():

            if assign not in other.log_probs_table:
                # TODO: make this `not is_close` and simplify this method
                if self_factor_comp != other.default_log_prob:
                    return False
            else:
                other_factor_comp = other.log_probs_table[assign]
                if not other_factor_comp.equals(self_factor_comp):
                    return False
        return True

    def equals(self, factor):
        """
        Check if this factor is the same as another factor.

        :param factor: The other factor to compare to.
        :type factor: ConditionalFactor
        :param float rtol: The relative tolerance to use for factor equality check.
        :param float atol: The absolute tolerance to use for factor equality check.
        :return: The result of the comparison.
        :rtype: bool
        """
        factor_ = factor
        if not isinstance(factor_, ConditionalFactor):
            raise TypeError(f"factor must be of ConditionalFactor type but has type {type(factor)}")

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
        :rtype: ConditionalFactor
        """
        return ConditionalFactor(
            conditioning_var_names=self.var_names.copy(),
            factor_table=self.factor_table,
            cardinalities=copy.deepcopy(self.cardinalities),
            default_factor=self.default_factor,
        )

    def copy_default_factor(self):
        if self.default_factor is None:
            return None
        return self.default_factor.copy()

    def marginalize(self, vrs, keep=True):
        """
        Sum out variables from this factor.

        :param vrs: (list) a subset of variables in the factor's scope
        :param keep: Whether to keep or sum out vrs
        :return: The resulting factor.
        :rtype: ConditionalFactor
        """
        vars_to_keep = super().get_marginal_vars(vrs, keep)
        vars_to_sum_out = [v for v in self.var_names if v not in vars_to_keep]

        discrete_vars_to_keep = list(set(vars_to_keep).intersection(self.conditioning_var_names))
        discrete_vars_to_sum_out = list(set(vars_to_sum_out).intersection(self.conditioning_var_names))
        conditional_vars_to_sum_out = list(set(vars_to_sum_out).intersection(self.conditional_var_names))
        marginal_default_factor = None
        if self.default_factor is not None:
            marginal_default_factor = self.copy_default_factor()

        if len(discrete_vars_to_keep) == 0:
            number_of_default_factors = np.product(self.cardinalities) - len(self.factor_table)
            default_factors = [self.copy_default_factor() for _ in range(number_of_default_factors)]
            all_conditional_factors = list(self.factor_table.values()) + default_factors
            mixture_factor_marginal = MixtureFactor(all_conditional_factors)
            if len(conditional_vars_to_sum_out) > 0:
                mixture_factor_marginal = mixture_factor_marginal.marginalize(vrs=conditional_vars_to_sum_out,
                                                                              keep=False)
            return mixture_factor_marginal

        assert len(discrete_vars_to_sum_out) > 0
        nested_table, _ = _get_nested_sorted_probs(
            new_variables_order_outer=discrete_vars_to_keep,
            new_variables_order_inner=discrete_vars_to_sum_out,
            old_variable_order=self.var_names,
            old_assign_probs=self.factor_table
        )
        result_table = dict()
        for l1_assign, factor_table in nested_table.items():
            factors_to_sum = list(factor_table.values())
            #
            mixture_factor_marginal = MixtureFactor(factors_to_sum)
            if len(conditional_vars_to_sum_out) > 0:
                mixture_factor_marginal = mixture_factor_marginal.marginalize(vrs=conditional_vars_to_sum_out,
                                                                              keep=False)
            result_table[l1_assign] = mixture_factor_marginal
        if len(conditional_vars_to_sum_out) > 0:
            if self.default_factor is not None:
                marginal_default_factor = self.default_factor.marginalise(vrs=conditional_vars_to_sum_out,
                                                                          keep=False)
            else:
                marginal_default_factor = None

        result_var_cards = copy.deepcopy(self.var_cards)
        for var in discrete_vars_to_sum_out:
            del result_var_cards[var]

        cardinalities = [self.var_cards[v] for v in vars_to_keep]

        resulting_factor = ConditionalFactor(conditioning_var_names=discrete_vars_to_keep,
                                             cardinalities=cardinalities,
                                             default_factor=marginal_default_factor,
                                             factor_table=result_table,)
        return resulting_factor

    def reduce(self, vrs, values):
        """
        Observe variables to have certain values and return reduced table.

        :param vrs: (list) The variables.
        :param values: (tuple or list) Their values
        :return: The resulting factor.
        :rtype: ConditionalFactor
        """
        continuous_vars_observed = list(set(self.conditional_var_names).intersection(set(vrs)))
        discrete_vars_observed = list(set(self.conditioning_var_names).intersection(set(vrs)))

        discrete_var_observed_values = [val for vr, val in zip(vrs, values) if vr in discrete_vars_observed]
        continuous_var_observed_values = [val for vr, val in zip(vrs, values) if vr in continuous_vars_observed]

        discrete_vars_unobserved = list(set(self.conditioning_var_names) - set(discrete_vars_observed))
        print("self.conditioning_var_names = ", self.conditioning_var_names)
        print("discrete_vars_observed = ", discrete_vars_observed)
        print("discrete_vars_unobserved = ", discrete_vars_unobserved)
        reduced_default_factor = self.copy_default_factor()
        if len(discrete_vars_observed) > 0:
            #print(self.factor_table)

            if len(discrete_vars_unobserved) > 0:
                nested_table, _ = _get_nested_sorted_probs(
                    new_variables_order_outer=discrete_vars_observed,
                    new_variables_order_inner=discrete_vars_unobserved,
                    old_variable_order=self.var_names,
                    old_assign_probs=self.factor_table,
                )
                lp_table = nested_table[tuple(discrete_var_observed_values)]
                result_var_cards = copy.deepcopy(self.var_cards)
                for var in discrete_vars_observed:
                    del result_var_cards[var]
            else:
                vrs_to_var_names_indices = [self.var_names.index(v) for v in discrete_vars_observed]
                var_names_order_assignments = tuple([discrete_var_observed_values[i] for i in vrs_to_var_names_indices])
                reduced_to_continuous_factor = self.factor_table[var_names_order_assignments]
                if len(continuous_vars_observed) > 0:
                    reduced_to_continuous_factor = reduced_to_continuous_factor.reduce(continuous_vars_observed,
                                                                                       continuous_var_observed_values)
                return reduced_to_continuous_factor
        else:
            lp_table = copy.deepcopy(self.factor_table) # TODO: remove this copy (safely)
            result_var_cards = copy.deepcopy(self.var_cards)
        if len(continuous_vars_observed) > 0:
            for assignment, factor in lp_table.items():
                print("continuous_vars_observed = ", continuous_vars_observed)
                print("continuous_var_observed_values = ", continuous_var_observed_values)
                lp_table[assignment] = factor.reduce(continuous_vars_observed,
                                                     continuous_var_observed_values)
            if self.default_factor is not None:
                print("if self.default_factor is not None: = ")

                reduced_default_factor = reduced_default_factor.reduce(continuous_vars_observed,
                                                                       continuous_var_observed_values)
            else:
                reduced_default_factor = None

        cards = list(result_var_cards.values())
        print("\n\ndiscrete_vars_unobserved = ", discrete_vars_unobserved)
        print("cards = ", cards)
        print("lp_table = ", lp_table)

        resulting_factor = ConditionalFactor(conditioning_var_names=discrete_vars_unobserved,
                                             cardinalities=cards,
                                             factor_table=lp_table,
                                             default_factor=reduced_default_factor)

        return resulting_factor

    def multiply(self, factor):
        """
        Multiply this factor with another factor and return the result.

        :param factor: The factor to multiply with.
        :type factor: ConditionalFactor
        :return: The factor product.
        :rtype: ConditionalFactor
        """
        if not isinstance(factor, ConditionalFactor):
            raise TypeError(f"factor must be of SparseCategorical type but has type {type(factor)}")
        return self._apply_binary_operator(factor, operator.mul, default_rules="any")

    def divide(self, factor):
        """
        Divide this factor by another factor and return the result.

        :param factor: The factor to divide by.
        :type factor: ConditionalFactor
        :return: The factor quotient.
        :rtype: ConditionalFactor
        """
        return self._apply_binary_operator(factor, operator.truediv, default_rules="none")

    def argmax(self):
        """
        Get the Categorical assignment (vector value) that maximises the factor potential.

        :return: The argmax assignment.
        :rtype: int list
        """
        raise NotImplementedError()

    def _apply_to_probs(self, func, include_assignment=False):
        """
        Apply a function to the log probs of the factor.

        :param func: The function to apply to the log probs in this factor.
        :param include_assignment: Whether or not to pass the assignment to the function as well
            (along with the log probs).
        """
        for assign, prob in self.factor_table.items():
            if include_assignment:
                self.factor_table[assign] = func(prob, assign)
            else:
                self.factor_table[assign] = func(prob)

    #  TODO: (modify and) use almost identical SparseCategorical function instead of this one.
    def _apply_binary_operator(self, factor, operator_function, default_rules="none"):
        """
        Apply a binary operator function f(self.factor, factor) and return the result

        :param factor: The other factor to use in the binary operation.
        :type factor: ConditionalFactor
        :param default_rules: The rules for when a calculation results will result in a default value (optional).
            This can help speed up this function. The possible values are as follows:
                'left' : If ntd_a has a default value, the result will always be default.
                'right' : If ntd_b has a default value, the result will always be default.
                'any' : If either ntd_a or ntd_b has a default value, the result will always be default.
                'both': Only if both ntd_a and ntd_b has a default value, the result will always be default.
                'none': No combination of default or non-default values is guarenteed to result in a default value
        If this parameter is not specified, 'none' will be used to ensure correct, albeit slower computation.
        :return: The resulting factor.
        :rtype: ConditionalFactor
        """
        # pylint: disable=too-many-locals
        if not isinstance(factor, ConditionalFactor):
            raise TypeError(f"factor must be of SparseCategorical type but has type {type(factor)}")
        self._assert_consistent_cardinalities(factor)
        intersection_vars = list(set(self.var_names).intersection(set(factor.var_names)))
        intersection_vars = sorted(intersection_vars)

        remaining_a_vars = list(set(self.var_names) - set(intersection_vars))
        ntd_a, _ = _get_nested_sorted_probs(
            new_variables_order_outer=remaining_a_vars,
            new_variables_order_inner=intersection_vars,
            old_variable_order=self.var_names,
            old_assign_probs=self.factor_table,
        )

        remaining_b_vars = list(set(factor.var_names) - set(intersection_vars))
        ntd_b, _ = _get_nested_sorted_probs(
            new_variables_order_outer=remaining_b_vars,
            new_variables_order_inner=intersection_vars,
            old_variable_order=factor.var_names,
            old_assign_probs=factor.factor_table,
        )

        # TODO: Add this functionality
        if self.default_log_prob != factor.default_log_prob:
            error_msg = (
                "Cases where self.default_value and factor.default_value differ are not yet supported."
            )
            raise NotImplementedError(error_msg)
        default_log_prob = self.default_log_prob

        def vars_to_cards(factor, var_names):
            return [factor.var_cards[v] for v in var_names]

        outer_inner_cards_a = [vars_to_cards(self, remaining_a_vars), vars_to_cards(self, intersection_vars)]
        outer_inner_cards_b = [
            vars_to_cards(factor, remaining_b_vars),
            vars_to_cards(factor, intersection_vars),
        ]

        result_ntd = _any_scope_binary_operation(
            ntd_a,
            outer_inner_cards_a,
            ntd_b,
            outer_inner_cards_b,
            operator_function,
            default_log_prob,
            default_rules=default_rules,
        )
        flattened_result_table = _flatten_ntd(result_ntd)
        result_var_names = remaining_a_vars + remaining_b_vars + intersection_vars

        result_var_cards = {**self.var_cards, **factor.var_cards}

        result_cardinalities = [result_var_cards[v] for v in result_var_names]
        result_default_log_prob = operator_function(default_log_prob, default_log_prob)
        resulting_factor = ConditionalFactor(
            conditioning_var_names=result_var_names,
            factor_table=flattened_result_table,
            cardinalities=result_cardinalities,
            default_factor=result_default_log_prob)

        return resulting_factor

    def normalize(self):
        """
        Normalize the factor.

        :return: The normalized factor.
        :rtype: ConditionalFactor
        """
        factor_copy = self.copy()
        log_weights = [factor.log_weight for factor in factor_copy.factor_table.values()]

        number_of_default_factors = np.product(self.cardinalities) - len(self.factor_table)
        log_weights += [self.default_factor.log_weight]*number_of_default_factors
        total_log_weight = special.logsumexp(log_weights)

        for assign, factor in factor_copy.log_probs_table.items():
            factor.add_log_weight(-total_log_weight)  #TODO: add add_log_weight function to all continuous function classes and make abstract method in Factor class
            factor_copy.log_probs_table[assign] = factor
        return factor_copy

    @staticmethod
    def _raw_kld(log_p, log_q):
        raise NotImplementedError()

    def kl_divergence(self, factor, normalize_factor=True):
        raise NotImplementedError()

    def potential(self, vrs, assignment):
        """
        Get the value of the factor for a specific assignment.

        :param assignment: The assignment
        :return: The value
        """
        assert set(vrs) == set(self.var_names), "variables (vrs) do not match factor variables."
        vrs_to_var_names_indices = [self.var_names.index(v) for v in vrs]
        var_names_order_assignments = tuple([assignment[i] for i in vrs_to_var_names_indices])
        return np.exp(self.factor_table[var_names_order_assignments])

    @property
    def weight(self):
        """
        An array containing all the discrete assignments and the corresponding probabilities.
        """
        log_weight = special.logsumexp([factor.log_weight for factor in self.factor_table.values()])
        weight_ = np.exp(log_weight)
        return weight_

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

        for assignment, factor in self.factor_table.items():
            line = "\n" + str(assignment) + "\t\t" + factor.__repr__()
            repr_str += line
        return repr_str

    def reorder(self, new_conditioning_var_names_order):
        """
        Reorder categorical table variables to a new order and reorder the associated probabilities
        accordingly.

        :param new_conditioning_var_names_order: The new variable order.
        :type new_conditioning_var_names_order: str list
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
        assert set(new_conditioning_var_names_order) == set(self.conditioning_var_names)
        new_order_indices = [self.conditioning_var_names.index(var) for var in new_conditioning_var_names_order]
        new_log_probs_table = dict()
        for assignment, factor in self.factor_table.items():
            reordered_assignment = tuple(assignment[i] for i in new_order_indices)
            new_log_probs_table[reordered_assignment] = factor
        reordered_cardinalities = [self.cardinalities[i] for i in new_order_indices]

        return ConditionalFactor(
            conditioning_var_names=new_conditioning_var_names_order,
            cardinalities=reordered_cardinalities,
            factor_table=new_log_probs_table,
            default_factor=self.default_factor
        )

