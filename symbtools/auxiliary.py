"""
This module contains some basic helper functions.
"""

import sympy as sp
from functools import wraps
import collections

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


def kwarg_wrapper(func, args=None):
    """
    Problem: Auto-generated functions (see e.g. expr_to_func) sometimes have many arguments.
    Then it is convenient to call them with keywordargs. If however the func has been wrapped by some postprocessing
    function (e.g. vectorization) this is not possible.

    Solution: This function creates a wrapper which takes the values from the dict and creates arglist according to
    the names in `args` (or func.args)

    :param func:    function which will be wrapped
    :param args:    list of argument names in correct order
    :return:
    """
    if args is None:
        args = func.args

    def inner(**kwargs):
        assert len(kwargs) == len(args)

        aa = [kwargs[a.name] for a in args]
        return func(*aa)

    inner.func = func
    inner.info = "This is a wrapper of `this.func`"

    return inner


def test_type(var, type_spec):
    """
    This function is like `isinstance()` but also handles complex types like `Sequence[str]` from the typing module.
    Currently only sequences are supported.

    :param var:
    :param type_spec:
    :return:
    """

    # prevent unnecessary compatibility break
    import typing as ty

    # noinspection PyUnresolvedReferences, PyProtectedMember
    if isinstance(type_spec, ty._GenericAlias):
        main_type = type_spec.__origin__

        if main_type is collections.abc.Sequence:
            sub_type = type_spec.__args__[0]

            if not isinstance(var, main_type):
                # not a sequence
                return False

            for elt in var:
                if not isinstance(elt, sub_type):
                    return False

            return True

        else:
            raise NotImplementedError()

    else:
        return isinstance(var, type_spec)
