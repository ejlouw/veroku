"""
A module containing general functions to help with factor operations and transformations.
"""


# TODO: simplify make_square_matrix and make_column vector (maybe renam to make ____ from list - there is already ...
# TODO: ... such a function for msquare matrix). These functions do not need al tie current functionality. ...
# TODO: ... Ensure factor classes still work.

import numpy as np
import matplotlib.pyplot as plt
import copy


def format_list_elements(lst, format_dict):
    formatted_list = [e.format(**format_dict) for e in lst]
    return formatted_list


def get_subset_evidence(all_evidence_dict, subset_vars):
    """
    Select evidence for certain variables only from a evidence_dict and return the variables for which there is evidence
     and the corresponding values.
    :param all_evidence_dict: (dict) The evidence dictionary.
    :param subset_vars: (string list) the subset of variables for which to select evidence from all_evidence_dict.
    :return: evidence variable names and values
    """
    factor_scope_evidence = {v: all_evidence_dict[v] for v in subset_vars if v in all_evidence_dict}
    evidence_vrs = list(factor_scope_evidence.keys())
    evidence_values = list(factor_scope_evidence.values())
    return evidence_vrs, evidence_values


def get_closest_psd_from_symmetric_matrix(mat):
    V, Q = np.linalg.eig(mat)
    Vp = np.maximum(V, 0.0)
    Dp = np.diag(Vp)
    closest_psd_mat = Q.dot(Dp).dot(Q.T)
    return closest_psd_mat


def is_pos_def(matrix):
    """
    Check if matrix is positive definite
    :param matrix: the matrix to check
    :return: (bool) whether the matrix is positive-definite or not
    """
    return np.all(np.linalg.eigvals(matrix) > 0)


def remove_duplicate_values(array_like, tol=0.0):
    """
    Removes duplicate values from list (when tol=0.0) or remove approximately duplicate
    values if tol!=0.0
    """
    unique_values = [array_like[0]]
    for element in array_like:
        element_is_duplicate = False
        for uval in unique_values:
            if abs(uval - element) <= tol:
                element_is_duplicate = True
        if not element_is_duplicate:
            unique_values.append(element)
    return unique_values


def make_list(list_like_object):
    """
    Convert a object to a list
    :param list_like_object: a list like object (i.e string or 1d numpy array)
    :return: the equivalent list
    """
    if not isinstance(list_like_object, list):
        lst = [list_like_object]
        return lst
    return list_like_object


def make_scalar(scalar_like_object):
    """
    Convert a list or numpy array containing a single element to a scalar.
    :param scalar_like_object: a list or numpy array containing a single element (either int or float).
    :return: the int or float contained in the scalar_like_object
    """
    if isinstance(scalar_like_object, np.ndarray):
        if len(scalar_like_object) != 1:
            raise ValueError('array must have length of one to be convertible to a scalar.')
        return scalar_like_object.min()
    if isinstance(scalar_like_object, list):
        if len(scalar_like_object) != 1:
            raise ValueError('list must have length of one to be convertible to a scalar.')
        return scalar_like_object[0]
    if isinstance(scalar_like_object, (int, float)):
        return scalar_like_object
    raise ValueError(f'{type(scalar_like_object)} is not supported')


def checked_colvector(col_vector):
    assert isinstance(col_vector, np.ndarray), 'Error: col_vector not numpy array'
    assert len(col_vector.shape) == 2, 'Error: col_vector not 2d matrix.'
    assert col_vector.shape[1] == 1, 'Error: col_vector not column vector'


def make_column_vector(vector_like_object):
    """
    This function converts a vector like object into a standard
    numpy column vector.
    :param vector_like_object: an int, float, list or (one dimensional) array
    :return: a numpy column vector (array)
    """
    array = copy.deepcopy(vector_like_object)
    if isinstance(array, np.ndarray):
        array = array.ravel()
        error_msg = f'Error: array of shape {vector_like_object.shape} cannot be convertes to column vector'
        if len(vector_like_object.shape) > 1:
            assert not all([s > 1 for s in vector_like_object.shape]), error_msg
    else:
        array = np.array(vector_like_object).ravel()
    return np.expand_dims(array, axis=1)


#TODO: simplify this function like the one above
# pylint: disable=inconsistent-return-statements
def list_to_square_matrix(matrix_like_object):
    """
    Make a square matrix from a list.
    :param matrix_like_object: a list containing a single int or float or n lists of length n
    :return: an nxn numpy array.
    """
    len_list = len(matrix_like_object)
    if len_list == 0:
        raise ValueError('cannot make matrix from empty list.')
    if len_list > 1:
        for element in matrix_like_object:
            if not isinstance(element, list):
                raise ValueError('cannot make square matrix from one dimensional list')
            if len(element) != len_list:
                raise ValueError('cannot make square matrix from different length lists')
            # if isinstance(element[0], list):
            #    raise ValueError('cannot make square matrix from 3d list')
        return np.array(matrix_like_object)
    # len_list == 1:
    if isinstance(matrix_like_object[0], (int, float)):
        # matrix_like_object = [float or int]
        return np.array([matrix_like_object])
    if len(matrix_like_object[0]) == 1:
        if isinstance(matrix_like_object[0][0], (int, float)):
            # matrix_like_object = [[float or int]]
            return np.array(matrix_like_object)
    raise ValueError(f'cannot convert matrix_like_object {matrix_like_object} to square matrix')
# pylint: enable=inconsistent-return-statements


# pylint: disable=inconsistent-return-statements
def make_square_matrix(matrix_like_object):
    """
    This function converts a matrix like object into a standard
    numpy square matrix.
    :param matrix_like_object: an int, float, list, list of lists or (square, two dimensional) array
    :return: a numpy square matrix (array)
    """
    if isinstance(matrix_like_object, (int, float)):
        return np.array([[matrix_like_object]])
    if isinstance(matrix_like_object, list):
        return list_to_square_matrix(matrix_like_object)

    if isinstance(matrix_like_object, np.ndarray):
        if len(matrix_like_object.shape) > 2:
            raise ValueError('cannot convert higher (than 2) dimensional tensor to 2d matrix')
        if len(matrix_like_object.shape) == 1:
            return np.expand_dims(matrix_like_object, axis=0)
        assert matrix_like_object.shape[0] == matrix_like_object.shape[1]
        return matrix_like_object.copy()
    raise ValueError(f'cannot convert matrix_like_object {matrix_like_object} to square matrix.')
# pylint: enable=inconsistent-return-statements


def indexed_square_matrix_operation(mat_a, mat_b, var_names_a, var_names_b, operator):
    # TODO: update usages
    """
    A function that performs element wise operations between square matrices where the corresponding indices in the
    different matrices do not necessarily correspond to the same variables, but can be mapped to each other through the
    variable name lists (var_names_a and var_names_b).

    :param mat_a: A nxn numpy array to be used with the operator
    :param mat_b:  A nxn numpy array to be used with the operator
    :param var_names_a: the variable names corresponding to the values in mat_a
    :param var_names_b: the variable names corresponding to the values in mat_b
    :param operator: the operator to be applied to mat_a and mat_b
    :return: the result of the operation

    Example:
        With:
            mat_a =
            [[11, 12, 13],
             [21, 22, 23],
             [31, 32, 33]]
            var_names_a = ['var1', 'var2']
            mat_b =
            [[11 , 13],
             [31,  33]]
            var_names_b = ['var1', 'var3']
            operator_str = '-'
        the function returns:
        [[0,  12, 0],
         [21, 22, 23],
         [0,  32, 0]]
    """

    assert len(mat_a.shape) == 2, 'Error: matrix needs to be 2 dimensional'
    assert len(mat_b.shape) == 2, 'Error: matrix needs to be 2 dimensional'
    assert mat_a.shape[0] == mat_a.shape[1], 'Error: matrix needs to be square'
    assert mat_b.shape[0] == mat_b.shape[1], 'Error: matrix needs to be square'
    if len(var_names_a) != mat_a.shape[0]:
        print('var_names_a = ', len(var_names_a))
        print('mat_a = \n', mat_a.shape)
        raise AssertionError('The number of variables in var_names_a does not match the dimensions of mat_a.')
    if len(var_names_b) != mat_b.shape[0]:
        print('var_names_b = ', len(var_names_b))
        print('mat_b = \n', mat_b.shape)
        raise AssertionError('The number of variables in var_names_b does not match the dimensions of mat_b.')
    # TODO: clean this up (also see todo in indexed_column_vector_operation)
    if set(var_names_b) <= set(var_names_a):
        new_vars = var_names_a
    elif set(var_names_a) <= set(var_names_b):
        new_vars = var_names_b
    else:
        new_vars = list(set(var_names_b + var_names_a))
        new_vars.sort()
    new_dim = len(new_vars)
    new_matrix = np.zeros([new_dim, new_dim])

    for i, var_i_name in enumerate(new_vars):
        for j, var_j_name in enumerate(new_vars):
            val_a = 0.0
            if (var_i_name in var_names_a) and (var_j_name in var_names_a):
                val_a = mat_a[var_names_a.index(var_i_name), var_names_a.index(var_j_name)]
            val_b = 0.0
            if (var_i_name in var_names_b) and (var_j_name in var_names_b):
                val_b = mat_b[var_names_b.index(var_i_name), var_names_b.index(var_j_name)]
            new_matrix[i, j] = operator(val_a, val_b)

    return new_matrix, new_vars


def indexed_column_vector_operation(colvec_a, colvec_b, var_names_a, var_names_b, operator):
    # TODO: modify to match indexed_square_matrix_operation and update usages
    """
    A function that performs element operations between column vectors, where the corresponding indices in the different
    vectors do not necessarily correspond to the same variables, but can be mapped to each other through the variable
    name lists (var_names_a and var_names_b).

    :param colvec_a: A nx1 numpy array to be used with the operator
    :param colvec_b:  A nx1 numpy array to be used with the operator
    :param var_names_a: the variable names corresponding to the values in colvec_a
    :param var_names_b: the variable names corresponding to the values in colvec_b
    :param operator: the operator to be applied to colvec_a and colvec_b
    :return: the result of the operation

    Example:
        With:
            colvec_a =
            [[11],
             [21],
             [31]]
            var_names_a = ['var1', 'var2']
            colvec_b =
            [[11],
             [31]]
            var_names_b = [var1', 'var3']
            operator_str = '-'
        the function returns:
        [[0],
         [21],
         [0]]
    """

    assert len(colvec_a.shape) == 2, 'Error: column vector needs to be 2 dimensional'
    assert len(colvec_b.shape) == 2, 'Error: column vector needs to be 2 dimensional'
    assert colvec_a.shape[1] == 1, 'Error: column vector array cannot have more than one column'
    assert colvec_b.shape[1] == 1, 'Error: column vector array cannot have more than one column'
    assert len(var_names_a) == colvec_a.shape[0], \
        'Error: number of variables in var_names_a does not match the dimension of colvec_a.'
    assert len(var_names_b) == colvec_b.shape[0], \
        'Error: number of variables in var_names_a does not match the dimension of colvec_b.'
    # TODO: clean this up (also see todo in indexed_square_matrix_operation)
    if set(var_names_b) <= set(var_names_a):
        new_vars = var_names_a
    elif set(var_names_a) <= set(var_names_b):
        new_vars = var_names_b
    else:
        new_vars = list(set(var_names_b + var_names_a))
        new_vars.sort()
    new_dim = len(new_vars)
    new_vector = np.zeros([new_dim, 1])

    for i, var_i_name in enumerate(new_vars):
        val_a = 0.0
        if var_i_name in var_names_a:
            val_a = colvec_a[var_names_a.index(var_i_name), 0]
        val_b = 0.0
        if var_i_name in var_names_b:
            val_b = colvec_b[var_names_b.index(var_i_name), 0]
        new_vector[i, 0] = operator(val_a, val_b)

    return new_vector, new_vars


def plot_2d(func, xlim, ylim, xlabel, ylabel, figsize=None):  # pragma: no cover
    """
    Plot a 2d function
    :param xlim: the x limits to plot the function over.
    :param ylim: the y limits to plot the function over.
    :param func: the function to plot
    :param xlim: the x limits over which to plot the function
    :param ylim: the x limits over which to plot the function
    :param xlabel: the x label for the plot
    :param ylabel: the y label for the plot
    :param figsize: the figure size
    """

    x_space = np.linspace(xlim[0], xlim[1], num=100)
    y_space = np.linspace(ylim[0], ylim[1], num=100)

    x_mesh, y_mesh = np.meshgrid(x_space, y_space)
    xx_mesh_array = np.array([x_mesh.ravel(), y_mesh.ravel()]).T
    z_mesh_array = np.array([func(X) for X in xx_mesh_array])
    z_mesh = z_mesh_array.reshape(x_mesh.shape)

    if figsize is not None:
        plt.figure(figsize=figsize)
    color_scale = plt.contourf(x_mesh, y_mesh, z_mesh, levels=30, cmap=plt.cm.viridis)
    plt.colorbar(color_scale, shrink=0.8, extend='both')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
