"""
A module for instantiating sparse tables with log probabilities.
"""

# System imports
import copy
import operator
import warnings

# Third-party imports
import numpy as np
from numpy import unravel_index
from scipy import special
from itertools import product

# Local imports
from veroku.factors._factor import Factor
from veroku.factors._factor_template import FactorTemplate

# special operator rules
LOG_SUBTRACT_CANCEL_RULES = {(-np.inf, operator.sub, -np.inf): -np.inf}
LOG_SUBTRACT_KL_RULES = {(-np.inf, operator.sub, -np.inf): 0.0}


# TODO: consider removing some unused functions

# TODO: Currently variable values should be the same as numpy indices and therefore start at 0 with increments of 1.
#       We probably want to relax this requirement.


class Categorical(Factor):
    """
    A class for instantiating sparse tables with log probabilities.
    """

    def __init__ (self, var_names, cardinalities=None, probs_table=None, log_probs_tensor=None):
        """
        Construct a SparseLogTable. Either log_probs_table or probs_table should be supplied.

        :param var_names: The variable names.
        :type var_names: str list
        :param probs_table: A dictionary with assignments (tuples) as keys and probabilities as values.
            Missing assignments are assumed to have zero probability.
        :type probs_table: dict

        Example:
            >>> var_names = ['rain','slip']
            >>> probs_table = {(0,0):0.8,
            >>>                (0,1):0.2,
            >>>                (1,0):0.4,
            >>>                (1,1):0.6}
            >>>var_cardinalities = [2,2]
            >>>table = Categorical(log_probs_table=log_probs_table,
            >>>                    var_names=var_names,
            >>>                    cardinalities=var_cardinalities)
        """
        # TODO: add check that assignment lengths are consistent with var_names
        # TODO: add check that cardinalities are consistent with assignments
        super().__init__(var_names=var_names)

        if cardinalities is None:
            if log_probs_tensor is None:
                raise ValueError('numpy array type log_probs_tensor is expected cardinalities are not supplied.\n' +
                                 'Alternatively, provide cardinalities with dict type probs_table.')
            cardinalities = log_probs_tensor.shape
        elif len(cardinalities) != len(var_names):
            raise ValueError('The cardinalities and var_names lists should be the same length.')
        if (log_probs_tensor is None) and (probs_table is None):
            raise ValueError('Either log_probs_tensor or probs_table must be specified')
        if log_probs_tensor is None:
            probs_tensor = np.zeros(shape=cardinalities)
            for assignment, prob in probs_table.items():
                try:
                    probs_tensor[assignment] = prob
                except IndexError:
                    error_message = f'assignment {assignment} is not consistent with the cardinalities ({cardinalities}) provided'
                    raise IndexError(error_message)
            with warnings.catch_warnings():
                # prevents 'divide by zero encountered in log' from displaying
                warnings.simplefilter("ignore")
                self.log_probs_tensor = np.log(probs_tensor)
        else:
            if len(log_probs_tensor.shape) != len(var_names):
                raise ValueError('log_probs_tensor should have dimension equal to the number of variables.')
            self.log_probs_tensor = log_probs_tensor.copy()
        # TODO: remove this
        self.var_cards = dict(zip(var_names, cardinalities))
        self.cardinalities = cardinalities

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
        if new_var_names_order == self.var_names:
            return self.copy()
        if set(new_var_names_order) != set(self.var_names):
            raise ValueError('The new_var_names_order set must be the same as the factor variables.')
        vars_new_order_indices = [self.var_names.index(v) for v in new_var_names_order]
        log_probs_tensor = self.log_probs_tensor.transpose(vars_new_order_indices)
        return Categorical(var_names=new_var_names_order, log_probs_tensor=log_probs_tensor)

    def equals(self, factor):
        """
        Check if this factor is the same as another factor.

        :param factor: The other factor to compare to.
        :type factor: Categorical
        :return: The result of the comparison.
        :rtype: bool
        """
        if not isinstance(factor, Categorical):
            raise ValueError(f'factor must be of Categorical type but has type {type(factor)}')
        if set(self.var_names) != set(factor.var_names):
            return False
        factor_copy = factor.reorder(self.var_names)
        if not np.allclose(self.log_probs_tensor, factor_copy.log_probs_tensor):
            return False
        return True

    def copy(self):
        """
        Make a copy of this factor.

        :return: The copy of this factor.
        :rtype: Categorical
        """
        return Categorical(var_names=self.var_names.copy(),
                           log_probs_tensor=self.log_probs_tensor.copy())

    @staticmethod
    def _get_shared_order_smaller_vars(smaller_vars, larger_vars):
        """

        :param smaller_vars:
        :param larger_vars:
        :return:

            Example:
            >>> larger_vars = ['a', 'c', 'd', 'b']
            >>> smaller_vars = ['c', 'e', 'b']
            >>> _get_shared_order_smaller_vars(smaller_vars, larger_vars)
            >>> ['c', 'b']
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

    @staticmethod
    def tensor_operation(tensor_a, tensor_b, tensor_a_var_names, tensor_b_var_names, func):
        """
        Apply a element wise function to two tensors with named indices.

        :param numpy.ndarray tensor_a: The first tensor.
        :param numpy.ndarray tensor_b: The second tensor.
        :param tensor_a_var_names: The first tensor's variable names corresponding to the indices.
        :type tensor_a_var_names: string list
        :param tensor_b_var_names: The second tensor's variable names corresponding to the indices.
        :type tensor_b_var_names: string list
        :param func: The function to apply elementwise to the two tensors (func(a,b))
        :return: The resulting tensor and corresponding variable names
        :rtype: string list
        """
        if tensor_a_var_names == tensor_b_var_names:
            result_tensor = func(tensor_a, tensor_b)
            return result_tensor, tensor_a_var_names
        common_vars = set(tensor_a_var_names).intersection(tensor_b_var_names)

        remaining_f1_vars = [v for v in tensor_a_var_names if v not in common_vars]
        remaining_f2_vars = [v for v in tensor_b_var_names if v not in common_vars]
        # sort according to f1_vars so that only f2_tensor can be transposed if sets are the same
        common_vars = [v for v in tensor_a_var_names if v in common_vars]
        f1_vars_new_order = [tensor_a_var_names.index(v) for v in remaining_f1_vars]
        f1_vars_new_order += [tensor_a_var_names.index(v) for v in common_vars]

        f1_tensor_new_shape = tensor_a.transpose(f1_vars_new_order)
        f1_dim_expansion_axis = list(range(len(remaining_f2_vars)))
        f1_tensor_std_shape = np.expand_dims(f1_tensor_new_shape, axis=f1_dim_expansion_axis)
        #  now f1 has var order / dim order: [[1]*num_f2_vars, rest_f1_vars_dims, common_vars]

        f2_vars_new_order = [tensor_b_var_names.index(v) for v in remaining_f2_vars]
        f2_vars_new_order += [tensor_b_var_names.index(v) for v in common_vars]
        f2_tensor_new_shape = tensor_b.transpose(f2_vars_new_order)
        # now f2 has var order / dim order:  [common_vars_dim, [1]*num_f2_vars]

        f2_dim_expansion_axis = list(range(len(remaining_f2_vars), len(remaining_f2_vars) + len(remaining_f1_vars)))
        f2_tensor_std_shape = np.expand_dims(f2_tensor_new_shape, axis=f2_dim_expansion_axis)
        # now f1 has var order / dim order: [f1_vars_dims, [1]*num_f1_vars] =
        # [common_vars, rest_f1_vars_dims, [1]*num_f2_vars]
        result_vars = remaining_f2_vars + remaining_f1_vars + common_vars
        result_tensor = func(f1_tensor_std_shape, f2_tensor_std_shape)
        return result_tensor, result_vars

    # TODO: change back to log form
    def marginalize(self, vrs, keep=False):
        """
        Sum out variables from this factor.

        :param vrs: (list) a subset of variables in the factor's scope
        :param keep: Whether to keep or sum out vrs
        :return: The resulting factor.
        :rtype: Categorical
        """

        vars_to_keep = super().get_marginal_vars(vrs, keep)
        vars_to_sum_out = [v for v in self.var_names if v not in vars_to_keep]
        indices_to_sum_out = [self.var_names.index(v) for v in vars_to_sum_out]
        result_tensor = np.apply_over_axes(special.logsumexp, self.log_probs_tensor, axes=indices_to_sum_out)
        return Categorical(var_names=vars_to_keep, log_probs_tensor=np.squeeze(result_tensor))

    def reduce(self, vrs, values):
        """
        Observe variables to have certain values and return reduced table.

        :param vrs: (list) The variables.
        :param values: (tuple or list) Their values
        :return: The resulting factor.
        :rtype: Categorical
        """

        vars_unobserved = [v for v in self.var_names if v not in vrs]
        obs_dict = dict(zip(vrs, values))

        reduced_tensor_indexing = tuple([obs_dict[v] if v in obs_dict else slice(None) for v in self.var_names])
        result_table = self.log_probs_tensor[reduced_tensor_indexing]
        return Categorical(var_names=vars_unobserved, log_probs_tensor=result_table)

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
        :type factor: Categorical
        :return: The factor product.
        :rtype: Categorical
        """
        if not isinstance(factor, Categorical):
            raise ValueError(f'factor must be of SparseLogTable type but has type {type(factor)}')
        self._assert_consistent_cardinalities(factor)
        result_tensor, result_vars = self.tensor_operation(self.log_probs_tensor, factor.log_probs_tensor,
                                                           self.var_names, factor.var_names, operator.add)
        return Categorical(var_names=result_vars, log_probs_tensor=result_tensor)

    def cancel(self, factor):
        """
        Almost like divide, but with a special rule that ensures that division of zeros by zeros results in zeros. When
        categorical message factors with zero probabilities are used in Belief Update algorithms, multiplication by the
        zero probabilities cause zeros in the cluster potentials, when these messages need to be divided out again, this
        results in 0/0 operations. Since, in such cases, we know (from the information in the message) that the
        probability value should be zero, it makes sense to set the result of 0/0 operations to 0 in these cases.

        :param factor: The factor to divide by.
        :type factor: Categorical
        :return: The factor quotient.
        :rtype: Categorical
        """
        augmented_factor_tensor = factor.log_probs_tensor.copy()
        # when cancelling out a factor
        special_case_indices = np.where((augmented_factor_tensor == -np.inf) & (self.log_probs_tensor == -np.inf))

        indices_are_empty = [a.size == 0 for a in special_case_indices]
        if not any(indices_are_empty):
            augmented_factor_tensor[special_case_indices] = np.float(0.0)
        elif not all(indices_are_empty):
            print('augmented_factor_tensor: ')
            print(augmented_factor_tensor)
            print('self.log_probs_tensor: ')
            print(self.log_probs_tensor)
            raise ValueError('something strange happened.')
        result_tensor, result_vars = self.tensor_operation(self.log_probs_tensor, augmented_factor_tensor,
                                                           self.var_names, factor.var_names, operator.sub)
        return Categorical(var_names=result_vars, log_probs_tensor=result_tensor)

    def divide(self, factor, special_rules=None):
        """
        Divide this factor by another factor and return the result.

        :param factor: The factor to divide by.
        :type factor: Categorical
        :param special_rules: Any special rules to apply for specific values of the left and right variables.
            For example: {(left_var_val, right_var_val): result, ...}
        :type special_rules: dict
        :return: The factor quotient.
        :rtype: Categorical
        """
        if not isinstance(factor, Categorical):
            raise ValueError(f'factor must be of SparseLogTable type but has type {type(factor)}')
        self._assert_consistent_cardinalities(factor)
        result_tensor, result_vars = self.tensor_operation(self.log_probs_tensor, factor.log_probs_tensor,
                                                           self.var_names, factor.var_names, operator.sub)
        return Categorical(var_names=result_vars, log_probs_tensor=result_tensor)

    def argmax(self):
        """
        Get the Categorical assignment (vector value) that maximises the factor potential.

        :return: The argmax assignment.
        :rtype: int list
        """

        argmax_index = unravel_index(self.log_probs_tensor.argmax(), self.log_probs_tensor.shape)
        return argmax_index

    def normalize(self):
        """
        Normalize the factor.

        :return: The normalized factor.
        :rtype: Categorical
        """
        factor_copy = self.copy()
        logz = special.logsumexp(self.log_probs_tensor)
        factor_copy.log_probs_tensor -= logz
        return factor_copy

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
        Get the KL-divergence D_KL(P || Q) = D_KL(self || factor) between a normalized version of this factor and another factor.
        Reference https://infoscience.epfl.ch/record/174055/files/durrieuThiranKelly_kldiv_icassp2012_R1.pdf, page 1.

        :param factor: The other factor
        :type factor: Gaussian
        :param normalize_factor: Whether or not to normalize the other factor before computing the KL-divergence.
        :type normalize_factor: bool
        :return: The Kullback-Leibler divergence
        :rtype: float
        """
        normalized_self = self.normalize()
        log_p = normalized_self.log_probs_tensor
        factor_ = factor
        if normalize_factor:
            factor_ = factor.normalize()
        log_q = factor_.log_probs_tensor
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kl_array = np.where(log_p != -np.inf, np.exp(log_p) * (log_p - log_q), 0.0)
        kld = np.sum(kl_array)
        if kld < 0.0:
            if np.isclose(kld, 0.0, atol=1e-5):
                #  this is fine (numerical error)
                return 0.0
            print('\nnormalize_factor = ', normalize_factor)
            print('self = ')
            self.show()
            print('normalized_self = ')
            normalized_self.show()
            print('\nfactor = ')
            factor_.show()
            raise ValueError(f'Negative KLD: {kld}')
        return kld

    def distance_from_vacuous(self):
        """
        Get the Kullback-Leibler (KL) divergence between this factor and a uniform copy of it.

        :return: The KL divergence.
        :rtype: float
        """
        # make uniform copy
        uniform_log_prob = -np.log(np.product(self.log_probs_tensor.shape))
        uniform_log_tensor = np.ones(self.log_probs_tensor.shape) * uniform_log_prob
        uniform_factor = Categorical(var_names=self.var_names, log_probs_tensor=uniform_log_tensor)
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
        obs_dict = dict(zip(vrs, assignment))
        obs_tensor_indexing = tuple([obs_dict[v] if v in obs_dict else slice(None) for v in self.var_names])
        return self.log_probs_tensor[obs_tensor_indexing]

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
        for assignment in np.ndindex(self.log_probs_tensor.shape):
            prob = self.log_probs_tensor[assignment]
            if exp_log_probs:
                prob = np.exp(prob)
            print(assignment, ' ', prob)


class CategoricalTemplate(FactorTemplate):

    def __init__(self, log_probs, var_templates):
        """
        Create a Categorical factor template.

        :param probs_table: The log_probs_table that specifies the assignments and values for the template.
        :type probs_table: tuple:float dict
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
        self.log_probs_table = copy.deepcopy(log_probs)

    def make_factor(self, format_dict=None, var_names=None):
        """
        Make a factor with var_templates formatted by format_dict to create specific var names.

        :param format_dict: The dictionary to be used to format the var_templates strings.
        :type format_dict: str dict
        :return: The instantiated factor.
        :rtype: Categorical
        """
        if format_dict is not None:
            assert var_names is None
            var_names = [vt.format(**format_dict) for vt in self._var_templates]
        return Categorical(probs_table=copy.deepcopy(self.log_probs),
                           var_names=var_names, cardinalities=self.var_cards.values())
