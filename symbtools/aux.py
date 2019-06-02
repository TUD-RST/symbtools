"""
This module contains some basic helper functions.
"""

import sympy as sp
from functools import wraps

t = sp.Symbol("t")


def lzip(*args):
    """
    this function emulates the python2 behavior of zip (saving parentheses in py3)
    """
    return list(zip(*args))


class Container(object):
    """General purpose container class to conveniently store data attributes
    """

    def __init__(self, **kwargs):
        assert len( set(dir(self)).intersection(list(kwargs.keys())) ) == 0
        self.__dict__.update(kwargs)


# data structure to store some data on module level without using `global` keyword
if not hasattr(sp, 'global_data'):
    # host this object in sp module to prevent dataloss when reloading the module
    # not very clean but facilitates interactive development
    sp.global_data = Container()

# noinspection PyUnresolvedReferences
global_data = sp.global_data

# the following is usefull for recursive functions to aviod code-duplication
# (repetition of function name)
# https://stackoverflow.com/a/35951133/333403


# this is the decorator
def recursive_function(func):

    @wraps(func)  # this decorator adapts name and docstring
    def tmpffunc(*args, **kwargs):
        return func(tmpffunc, *args, **kwargs)

    return tmpffunc


# noinspection PyPep8Naming
def matrix_atoms(M, *args, **kwargs):
    sets = [m.atoms(*args, **kwargs) for m in list(M)]
    S = set().union(*sets)

    return S


def atoms(expr, *args, **kwargs):
    if isinstance(expr, (sp.Matrix, list)):
        return matrix_atoms(expr, *args, **kwargs)
    else:
        return expr.atoms(*args, **kwargs)
