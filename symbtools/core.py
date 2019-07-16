# -*- coding: utf-8 -*-


"""
useful functions on basis of sympy
"""

import inspect
from collections import Counter, Iterable, namedtuple
import random
import itertools as it
import collections as col
from functools import reduce
from .auxiliary import lzip, atoms, matrix_atoms, recursive_function, Container, global_data, t
from .time_deriv import time_deriv, is_derivative_symbol, matrix_time_deriv, get_sp_deriv_order, symb_to_time_func,\
    match_symbols_by_name, get_all_deriv_childs, get_all_deriv_parents

from .pickle_tools import pickle_full_dump, pickle_full_load

import warnings
# goal: trigger deprecation warnings in this module, result: triggers many warnings in
# other modules, too. -> comment out
# warnings.simplefilter('default')

import numpy as np
import sympy as sp


try:
    # usefull for debugging but not mandatory
    from ipydex import IPS
except ImportError:
    pass

# placeholder to inject a custom simplify function for the enullspace function
nullspace_simplify_func = None

# convenience
np.set_printoptions(8, linewidth=300)

piece_wise = sp.functions.elementary.piecewise.Piecewise # avoid name clashes with sage

zf = sp.numbers.Zero()

# SymNum_Container:
SNC = namedtuple("SNC", ("expr", "func"))

# The following definitions allow useful shorthands in interactive mode:
# (IPython, or IPython-Notebook):
# <object>.s  as alias for <object>.atoms(sp.Symbol)
# (determine from which symbols does an expression depend)
# <object>.co as alias for count_ops(object) (with matrix support)
# (determine how "big" an expression is without converting it to string (slow))

new_methods = []


@property
def satoms(self):
    """
    convenience property for interactive usage:
    returns self.atoms(sp.Symbol)
    """
    return self.atoms(sp.Symbol)


new_methods.append(('s', satoms))


@property
def sco(self):
    """
    convenience property for interactive usage:
    returns count_ops(self)
    """
    return count_ops(self)


new_methods.append(('co', sco))


def subz(self, args1, args2):
    """
    convenience property for interactive usage:
    returns self.subs(lzip(args1, args2))
    """
    return self.subs(lzip(args1, args2))


new_methods.append(('subz', subz))


def subz0(self, *args):
    """
    convenience property for interactive usage:
    returns self.subs(zip0(arg[0]) + zip0(args[1]) + ...)
    """

    # nested list comprehension http://stackoverflow.com/a/952952/333403
    # flatten the args
    all_args = [symb for sequence in args for symb in sequence]
    return self.subs(zip0(all_args))


new_methods.append(('subz0', subz0))


@property
def smplf(self):
    """
    convenience property for interactive usage:
    return sp.simplify(self)

    saves typing effort and handles scalars and matrices uniformly.
    (self.simplify() works in place for matrices, thus printing needs additional line)
    """
    return sp.simplify(self)


new_methods.append(('smplf', smplf))


@property
def srn01(self):
    """
    Convenience property for interactive usage:
    returns subs_random_numbers(self, **kwargs)
    """
    return subs_random_numbers(self)
new_methods.append(('srn01', srn01))


@property
def srn(self):
    """
    Convenience property for interactive usage:
    returns subs_random_numbers(self, **kwargs)
    """
    return subs_random_numbers(self, minmax=(1, 10))
new_methods.append(('srn', srn))


@property
def srnr(self):
    """
    Convenience property for interactive usage:
    returns subs_random_numbers(self, round_res_digits=2)
    """
    return subs_random_numbers(self, round_res_digits=2, minmax=(1, 10))
new_methods.append(('srnr', srnr))


@property
def ar(self):
    """
    Convenience property for interactive usage:
    returns apply_round(self)
    """
    return apply_round(self)
new_methods.append(('ar', ar))


@property
def srnp(self):
    """
    Convenience property for interactive usage:
    returns subs_random_numbers(self, **kwargs)
    """
    return subs_random_numbers(self, prime=True).evalf()
new_methods.append(('srnp', srnp))


target_classes = [sp.Expr, sp.ImmutableDenseMatrix, sp.Matrix]
for tc in target_classes:
    for name, meth in new_methods:
        setattr(tc, name, meth)


# create a place where userdefined attributes are stored (needed for difforder)
def init_attribute_store(reinit=False):
    if not hasattr(global_data, 'attribute_store') or reinit:
        global_data.attribute_store = {}


init_attribute_store()


def copy_custom_attributes(old_symbs, new_symbs):

    old_symbs = list(old_symbs)
    new_symbs = list(new_symbs)
    assert len(old_symbs) == len(new_symbs)

    map_old_to_new = dict(lzip(old_symbs, new_symbs))

    new_items = []

    # attribute_store is a dict like {(xddot, 'difforder'): 2}
    # i.e. keys are 2-tuples like (variable, attr_name)
    for key, value in list(global_data.attribute_store.items()):

        if key[0] in old_symbs:
            # there is an relevant attribute
            new_symb = map_old_to_new[key[0]]
            new_key = (new_symb, key[1])

            old_value = global_data.attribute_store.get(new_key)
            if  old_value is None or old_value == value :
                global_data.attribute_store[new_key] = value
            else:
                msg = "Name conflict: the attribute %s was already stored" % str(new_key)
                raise ValueError(msg)


def register_new_attribute_for_sp_symbol(attrname, getter=None, setter=None,
                                         getter_default=None, save_setter=True):
    """
    General way to register a new (fake-)attribute for sympy-Symbols

    :param attrname:
    :param getter:          function or None
    :param setter:          function or None
    :param getter_default:  default value for getter (avoid defining a function just to specify
                            the default value)
    :param save_setter:     forbid to change the value after it was set explicitly (default: True)


    :return:                None
    """

    assert attrname not in sp.Symbol.__dict__
    assert attrname not in sp.Expr.__dict__  # because this is manipulated above

    if getter is None:
        def getter(self):
            if getter_default == "__new_empty_list__":
                _getter_default = list()
            elif getter_default == "__self__":
                _getter_default = self
            else:
                _getter_default = getter_default
            return global_data.attribute_store.get((self, attrname), _getter_default)

    if setter is None:
        def setter(self, value):
            old_value = global_data.attribute_store.get((self, attrname))

            if save_setter and old_value is not None and not value == old_value:
                msg = "{} was already set to {} for symbol {}".format(attrname, old_value, self)
                raise ValueError(msg)

            global_data.attribute_store[(self, attrname)] = value
    theproperty = property(getter, setter)

    # theproperty.setter(setter)

    setattr(sp.Symbol, attrname, theproperty)

register_new_attribute_for_sp_symbol("difforder", getter_default=0)

register_new_attribute_for_sp_symbol("ddt_child")
register_new_attribute_for_sp_symbol("ddt_parent")
register_new_attribute_for_sp_symbol("ddt_func", getter_default="__self__")


def get_custom_attr_map(attr_name):
    res = []
    for key, value in global_data.attribute_store.items():
        if key[1] == attr_name:
            res.append((key[0], value))

    return res



# noinspection PyPep8Naming
class equation(object):

    def __init__(self, lhs, rhs = 0):
        self.lhs_ = sp.sympify(lhs)
        self.rhs_ = sp.sympify(rhs)

    def lhs(self):
        return self.lhs_

    def rhs(self):
        return self.rhs_

    def __repr__(self):
        return "%s == %s" % (self.lhs_, self.rhs_)

    def subs(self, *args, **kwargs):
        lhs_  = self.lhs_.subs(*args, **kwargs)
        rhs_  = self.rhs_.subs(*args, **kwargs)
        return type(self)(lhs_, rhs_)


def make_eqns(v1, v2 = None):
    if v2 == None:
        v2 = [0]*len(list(v1))
    return [equation(v1[i], v2[i]) for i in range(len(list(v1)))]

#def pw_curve(var, cn, left, right):
#    """
#    returns a piecewise polynomial y(t) that is cn times continous differentiable
#
#    left and right are sequences of conditions for the boundaries
#
#    left = (t1, y1,  derivs) # derivs contains cn derivatives
#
#    """
#    pass



# Der Inhalt des quickfrac moduls
"""
quick access to the most useful functionality of the fractions module
"""

import fractions as fr

Fr = fr.Fraction

def fractionfromfloat(x_, maxden = 1000):
  """
  fraction from float
  args:
   x
   maxdenominator (default = 1000)
  """

  x = float(x_)
  assert x == x_ # fails intentionally for numpy.complex
  return Fr.from_float(x).limit_denominator(maxden)

def sp_fff(x, maxden):
    """ sympy_fraction from float"""
    return sp.Rational(fractionfromfloat(x, maxden))



def condition_poly(var, *conditions):
    """
    This function is intended to be a generalization of trans_poly
    returns a polynomial y(t) that fullfills given conditions
    every condition is a tuple of the following form:

    (t1, y1,  *derivs) # derivs contains cn derivatives
    every derivative (to the highest specified [in each condition]) must be given
    """
    assert len(conditions) > 0

    # preparations
    cond_lengths = [len(c)-1 for c in conditions]  # -1: first entry is t
    condNbr = sum(cond_lengths)
    cn = max(cond_lengths)

    coeffs = [sp.Symbol('a%d' %i) for i in range(condNbr)]
    poly =  sum(map(lambda i, a: a*var**i, list(range(condNbr)), coeffs))

    Dpoly_list = [poly]+[sp.diff(poly, var, i) for i in range(1,cn+1)]

    new_conds = []
    for c in conditions:
        t = c[0]
        for i, d in enumerate(c[1:]):
            new_conds.append((t, d, i))
            # d : derivative at point t (including 0th)
            # i : derivative counter

    # evaluate the conditions

    conds = []

    for t, d, i in new_conds:
        conds += [equation(Dpoly_list[i].subs(var, t) , d)]

    sol = lin_solve_eqns(conds, coeffs)

    sol_poly = poly.subs(sol)

    return sol_poly


def trans_poly(var, cn, left, right):
    """
    Note: the usage of condition_poly is recommended over this function

    Returns a polynomial y(var) that is cn times continously differentiable

    left and right are sequences of conditions for the boundaries, e.g.,
        left = (t1, y1,  *derivs) # derivs contains cn derivatives

    """
    assert len(left) >= cn+2
    assert len(right) >= cn+2

    # allow to ignore superflous conditions
    left = left[:cn+2]
    right = right[:cn+2]

    t1, y1 = left[0:2]
    t2, y2 = right[0:2]

    assert t1 != t2

    for tmp in (y1, y2):
        assert not isinstance(tmp, (np.ndarray, np.matrix, sp.Symbol) )

    # store the derivs
    D1 = left[2:]
    D2 = right[2:]

    # preparations
    condNbr = 2 + 2*cn

    coeffs = [sp.Symbol('a%d' %i) for i in range(condNbr)]
    poly =  sum(map(lambda i, a: a*var**i, list(range(condNbr)), coeffs))

    Dpoly = [sp.diff(poly, var, i) for i in range(1,cn+1)]


    # create the conditions

    conds = []
    conds += [equation(poly.subs(var, t1) , y1)]
    conds += [equation(poly.subs(var, t2) , y2)]

    for i in range(cn):
        #

        conds += [equation(Dpoly[i].subs(var, t1) , D1[i])]
        conds += [equation(Dpoly[i].subs(var, t2) , D2[i])]


    sol = lin_solve_eqns(conds, coeffs)

    sol_poly = poly.subs(sol)

    return sol_poly


def create_piecewise(var, interface_positions, fncs):
    """
    Creates a sympy.Piecewise object, streamlined to trajectory planning.

    :var:                       variable (e.g. time t)
    :interface_positions:       sequence of n-1 values
    :fncs:                      sequence of n expressions (in depenence of var)


    example:

    create_piecewise(t, (0, 2), (0, t/2, 1))

    results in a ramp from 0 to 1 within 2 time units.
    """

    interface_positions = list(interface_positions)
    upper_borders = list(interface_positions)

    var = sp.sympify(var)
    inf = sp.oo

    assert len(upper_borders) == len(fncs) - 1

    pieces = [(fnc, var < ub) for ub, fnc in lzip(upper_borders[:-1], fncs[:-2])]

    # the last finite border sould be included, hence we use '<=' instead of '<'
    last_two_pieces = [(fncs[-2], var <= upper_borders[-1]), (fncs[-1], var < inf)]
    pieces.extend(last_two_pieces)

    return piece_wise(*pieces)


def create_piecewise_poly(var, *conditions):
    """
    For each successive pair of conditions create a `condition_poly`.
    Then create a piecewise function of all theses condition polys.
    """

    n_conditions = len(conditions)

    if  n_conditions == 0:
        raise ValueError("At least one condition is needed")
    polylist = []
    interface_points = []
    for i in range(n_conditions):
        polylist.append(condition_poly(var, *conditions[i:i+2]))
        interface_points.append(conditions[i][0])

    if n_conditions in (1, 2):
        # there is no intersection -> poly is globally defined
        return polylist[0]

    # remove the last poly (which has no successor)
    polylist.pop()
    # ignore boundary-points (they are no interfaces)
    interface_points = interface_points[1:-1]

    # SymNum_Container
    res = Container()
    # symbolic expression
    res.expr = create_piecewise(var, interface_points, polylist)

    # callable function as new attribute
    res.func = expr_to_func(var, res.expr)

    return res


def integrate_pw(fnc, var, transpoints):
    """
    due to a bug in sympy we must correct the offset in the integral
    to make the result continious
    """

    F=sp.integrate(fnc, var)

    fncs, conds = lzip(*F.args)

    transpoints = lzip(*transpoints)[0]

    oldfnc = fncs[0]
    new_fncs = [oldfnc]
    for f, tp  in lzip(fncs[1:], transpoints):
        fnew = f + oldfnc.subs(var, tp) - f.subs(var, tp)
        new_fncs.append(fnew)
        oldfnc = fnew

    pieces = lzip(new_fncs, conds)

    return piece_wise(*pieces)


# might be oboslete (intended use case did not carry on)
def deriv_2nd_order_chain_rule(funcs1, args1, funcs2, arg2):
    """
    :param funcs1: source functions f(a, b)
    :param args: arguments of f -> (a, b)
    :param funcs2: "arg functions a, b" (a(x), b(x))
    :param arg2: final arg x
    :return: the same as f.subs(...).diff(x, 2)

    background: the direct computation might take too long
    """

    if hasattr(funcs1, '__len__'):
        #funcs1 = list(funcs1)
        res = [deriv_2nd_order_chain_rule(f, args1, funcs2, arg2) for f in funcs1]

        if isinstance(funcs1, sp.MatrixBase):
            res = sp.Matrix(res)
            res = res.reshape(*funcs1.shape)

        return res

    assert isinstance(funcs1, sp.Expr)
    assert len(args1) == len(funcs2)
    assert isinstance(arg2, sp.Symbol)
    f = funcs1

    funcs2 = sp.Matrix(list(funcs2))

    H = sp.hessian(f, args1)
    H1 = H.subs(lzip(args1, funcs2))
    J = gradient(f, args1)

    v = funcs2.diff(arg2)
    Hterm = (v.T*H1*v)[0]
    J2 = (J*v).diff(arg2).subs(lzip(args1, funcs2))[0]

    return Hterm + J2


def lie_deriv(sf, *args, **kwargs):
    """
    lie_deriv of a scalar field along a vector field

    calling convention 1:
    lie_deriv(sf, vf, xx, order=1)

    calling convention 2:
    lie_deriv(sf, vf1, vf2, ..., vfn, xx)
    """

    # for backward-compatibilty:
    if 'n' in kwargs:
        assert not 'order' in kwargs
        kwargs['order'] = kwargs.pop('n')

    #determine the calling convenntion (cc):
    if len(args) == 3 and not kwargs.get('xx')\
            and not kwargs.get('order') and isinstance(args[2], int):

        cc = 1
        order = args[2]
        xx = args[1]
        vf_list = [args[0]]*order
        # example call:
        # lie_deriv(sf, vf, xx, 3)

    elif len(args) == 2 and not kwargs.get('xx') and 'order' in kwargs:
        cc = 1
        order = kwargs.get('order', 1)
        vf_list = [args[0]]*order
        xx = args[1]
        # example call:
        # lie_deriv(sf, vf, xx, order=3)

    elif len(args) == 1 and kwargs.get('xx'):
        cc = 1
        order = 1
        xx = kwargs.get('xx')
        vf_list = [args[0]]
        # example call:
        # lie_deriv(sf, vf, xx=xx)

    else:
        cc = 2
        assert not 'order' in kwargs
        if 'xx' in kwargs:
            xx = kwargs.get('xx')
            vf_list = args[:]  # make a copy
        else:
            assert len(args) >= 1
            xx = args[-1]
            vf_list = args[:-1]
        order = len(vf_list)

    if isinstance(xx, sp.Matrix):
        assert xx.shape[1] == 1
        xx = list(xx)

    for elt in xx:
        assert isinstance(elt, sp.Symbol)

    assert int(order) == order
    assert order >= 0
    assert order == len(vf_list)

    # check whether xx and the vectorfields all have the correct length
    if not [len(vf) for vf in vf_list] == [len(xx)]*order:
        msg = "At least one of the vector fields has a different length "\
        "compared to xx."
        raise ValueError(msg)


    if order == 0:
        return sf

    vf1 = vf_list[0]

    # now comes the actual calculation
    res = jac(sf, xx)*vf1
    assert res.shape == (1, 1)
    res = res[0]

    if order > 1:
        return lie_deriv(res, *vf_list[1:], xx=xx)
    else:
        return res


def lie_deriv_cartan(sf, vf, x, u=None, order=1, **kwargs):
    """
    lie_deriv of a scalar field along a cartan vector field
    (incorporating input derivatives)

    :param sf: scalar field ( e.g. h(x,u) )
    :param vf: vector field ( f(x, u) with xdot = f(x,u) )
    :param x:  state coordinates
    :param u:  input variables
    :param order: order of the lie derivative (integer)
    :return:

    see: M. Franke (PhD thesis), section 3.2.1
    """
    # TODO: find primary/english literature source

    if isinstance(x, sp.Matrix):
        assert x.shape[1] == 1
        x = list(x)

    assert int(order) == order and order >= 0
    if order == 0:
        return sf

    ordinary_lie_deriv = lie_deriv(sf, vf, x, n=1)

    if 0 and not u:
        return 1/0

    if u is None:
        u = []

    if is_symbol(u):
        # convenience
        u = [u]

    # assume u is a sequence of symbols (input variables)
    # or a sequence of sequences of symbols (input vars and its derivatives)
    assert hasattr(u, '__len__')

    if all( [ is_symbol(elt) for elt in u ] ):
        # sequence of symbols
        uu = sp.Matrix(u)
        uu_dot_list = [time_deriv(uu, uu, order=i)
                       for i in range(order + 1)]

    elif all([hasattr(elt, '__len__') for elt in u]):
        # sequence of sequences
        uu_dot_list = list(u)
        L = len(uu_dot_list[0])
        assert all([len(elt) == L for elt in uu_dot_list])

        N = len(uu_dot_list[1:]) # we already have derivatives up to order N
        # maybe we need more derivatives
        vv = sp.Matrix(uu_dot_list[-1])
        new_uu_dot_list = [time_deriv(vv, vv, order=i)
                           for i in range(1, order + 1)]
        uu_dot_list.extend(new_uu_dot_list)

    else:
        msg = "Expecting u like [u1, u2] or [(u1, u2), (udot1, udot2)]"
        raise ValueError(msg)

    # actually do the calculation
    u_res = 0
    for i, u_symbs_list in enumerate(uu_dot_list[:-1]):
        for j, us in enumerate(u_symbs_list):
            us_d = uu_dot_list[i + 1][j]  # get associated derivative
            u_res += sf.diff(us)*us_d

    res = ordinary_lie_deriv + u_res
    return lie_deriv_cartan(res, vf, x, uu_dot_list, order=order-1)


def lie_bracket(f, g, *args, **kwargs):
    """
    f, g should be vectors (or lists)
    args ... sequence of independent variables
    optional keyword arg `order` (of iterated lie-bracket (default: 1))

    call possibillities:

    lie_bracket(f, g, x1, x2, x3)
    lie_bracket(f, g, [x1, x2, x3])
    lie_bracket(f, g, sp.Matrix([x1, x2, x3]) )
    lie_bracket(f, g, [x1, x2, x3], n=3)   $[f, [f, [f, g]]]$
    """

    assert len(args) > 0

    if isinstance(args[0], sp.Matrix):
        assert args[0].shape[1] == 1
        args = list(args[0])

    if hasattr(args[0], '__len__'):
        args = args[0]

    if "n" in kwargs:
        raise DeprecationWarning("Parameter `n` is deprecated. Use `order`.")

    # remain compatible with the use of n but also handle cases where order is given
    n = kwargs.get("n", 1)

    order = kwargs.get("order", n)

    if order == 0:
        return g

    assert order > 0
    assert int(order) == order
    assert len(args) == len(list(f))

    # casting
    f = sp.Matrix(f)
    g = sp.Matrix(g)

    jf = f.jacobian(args)
    jg = g.jacobian(args)

    res = jg * f - jf * g

    if order > 1:
        res = lie_bracket(f, res, *args, order=order-1)

    return res


def lie_deriv_covf(w, f, args, **kwargs):
    """
    Lie derivative of covector field along vector field

    w, f should be 1 x n and n x 1 Matrices


    (includes the option to omit the transposition of Dw
    -> transpose_jac = False)
    """

    k, l = w.shape
    m, n = f.shape
    assert k == 1 and n == 1
    assert l == m

    if isinstance(args[0], sp.Matrix):
        assert args[0].shape[1] == 1
        args = list(args[0])

    if hasattr(args[0], '__len__'):
        args = args[0]

    assert len(args) == len(list(f))

    if "n" in kwargs:
        raise DeprecationWarning("Parameter `n` is deprecated. Use `order`.")

    # remain compatible with the use of n but also handle cases where order is given
    n = kwargs.get("n", 1)

    order = kwargs.get("order", n)

    if order == 0:
        return w

    # noinspection PyChainedComparisons
    assert order > 0 and int(order) == order

    # caution: in sympy jacobians of row and col vectors are equal
    # -> transpose is needless (but makes the formula consistent with books)
    jwT = w.T.jacobian(args)

    jf = f.jacobian(args)

    if not kwargs.get("transpose_jac", True):
        # strictly this is not a lie derivative
        # but nevertheless sometimes needed
        res = w*jf + f.T * jwT
    else:

        # This is the default case :
        res = w*jf + f.T * jwT.T

    if order > 1:
        res = lie_deriv_covf(res, f, args, order=order-1)

    return res


def involutivity_test(dist, xx, **kwargs):
    """
    Test whether a distribution is closed w.r.t the lie_bracket.
    This is done by checking whether the generic rank changes

    :param dist:    Matrix whose columns span the distribution
    :param xx:      coordinates
    :param kwargs:
    :return:        True or False and the first failing combination for cols
    """

    assert isinstance(dist, sp.Matrix)
    nr, nc = dist.shape
    assert len(xx) == nr
    combs = it.combinations(list(range(nc)), 2)

    rank0 = generic_rank(dist)

    res = True
    fail = []

    # if the distribution already spans the whole space
    if rank0 == nr:
        return res, fail

    for c in combs:
        f1, f2 = col_split(col_select(dist, *c))
        lb = lie_bracket(f1, f2, xx)

        tmp = col_stack(dist, lb)

        if generic_rank(tmp) != rank0:
            res = False
            fail = c
            break

    return res, fail


def system_pronlongation(f, gg, xx, prl_list, **kwargs):
    """
    Extend the system input by integrator-chains

    :param f:        drift vector field
    :param gg:       matrix of input vfs
    :param xx:       state-coordinates
    :param prl_list: list of tuples (input_number, prolongation_order)
    :return:         fnew, Gnew, xxnew
    """

    # convenience
    if isinstance(xx, (list, tuple)):
        xx = sp.Matrix(xx)

    nr, nc = gg.shape
    assert len(f) == nr == len(xx)
    # resort the prolongation_list
    idcs, orders = lzip(*prl_list)
    for idx, order in prl_list:
        assert 0 <= idx < nc
        assert 0 <= order == int(order)

    state_symbol = kwargs.get('state_symbol', 'z')
    Nz = sum(orders)
    max_idx = str(Nz + 1)
    # create new state_variables:

    # prevent name collisions
    collision_state_names = [str(x) for x in xx if str(x).startswith(state_symbol)]
    if len(collision_state_names) == 0:
        first_idx_str = '1'
    else:
        collision_state_names.sort()
        highest_collision_state_name = collision_state_names[-1]
        idx_str = highest_collision_state_name.replace(state_symbol, '')
        first_idx_str = str(int(idx_str) + 1)

    max_idx = str(Nz + int(first_idx_str))

    zz = symb_vector(state_symbol + first_idx_str + ':' + max_idx)

    fnew = row_stack( f, sp.zeros(Nz, 1) )
    ggnew = row_stack( gg, sp.zeros(Nz, nc) )
    xxnew = row_stack(xx, zz)

    k = 0
    for idx, order in prl_list:
        if order == 0:
            continue
        for j in range(order):
            k += 1
            if j == 0:
                # we convert an input to a state component
                # add the column (*z_k) to the drift vf
                fnew += ggnew[:, idx]*zz[k-1]

                # we can now already finish the column of ggnew
                # subtraction reasons -1 because k has already been incremented
                # uv uses 1-indexing, i.e.: uv(3, 1) = [1, 0, 0]
                matching_unit_vector = uv(n=nr + Nz, i=nr + k + order - 1)
                ggnew[:, idx] = matching_unit_vector
            else:
                assert k >= 2
                # we just add an integrator:
                # zz[k-2]_dot = zz[k-1] (i.e. zdot1 = z2, etc.)
                fnew[nr + k - 2, 0] = zz[k-1]

    return fnew, ggnew, xxnew


def multi_taylor(expr, args, x0 = None, order=1):
    """
    compute a multivariate taylor polynomial of a scalar function

    default: linearization about 0 (all args)
    """

    if x0 == None:
        x0 = [0 for a in args]
    x0 = list(x0) # to handle matrices
    assert len(args) == len(x0)

    x0list = lzip(args, x0)

    res = expr.subs(x0list)

    arg_idx_list = list(range( len(args)))

    for o in range(1,order+1):

        diff_list = it.product( *([arg_idx_list]*o) )

        for idx_tup in diff_list:

            arg_tup = [args[k] for k in idx_tup]

            prod = sp.Mul( *[args[k]-x0[k] for k in idx_tup] )

            tmp = expr.diff(*arg_tup)/sp.factorial(o)

            res+= tmp.subs(x0list) * prod
    return res


def multi_taylor_matrix(M, args, x0=None, order=1):
    """
    applies multi_taylor to each element
    """

    def func(m):
        return multi_taylor(m, args, x0, order)

    return M.applyfunc(func)


def numer_denom(expr):
    num, denom = expr.as_numer_denom() # resolves problems with multifractions
    return num/denom


def expand(arg):
    """
    sp.expand currently has no matrix support
    """
    if isinstance(arg, sp.Matrix):
        return arg.applyfunc(sp.expand)
    else:
        return sp.expand(arg)



def simplify(arg):
    """
    sp.simplify currently has no matrix support
    """
    if isinstance(arg, sp.Matrix):
        return arg.applyfunc(sp.simplify)
    else:
        return sp.simplify(arg)


def trigsimp(arg, **kwargs):
    """
    sp.trigsimp currently has no matrix support
    """
    if isinstance(arg, (sp.Matrix, sp.ImmutableMatrix)):
        return arg.applyfunc(lambda x: sp.trigsimp(x, **kwargs))
    else:
        return sp.trigsimp(arg, **kwargs)


def ratsimp(arg):
    """
    sp.ratsimp currently has no matrix support
    """
    if isinstance(arg, sp.MatrixBase):
        return arg.applyfunc(sp.ratsimp)
    else:
        return sp.ratsimp(arg)

def uv(n, i):
    """
    unit vectors (columns)
    """
    uv = sp.Matrix([0]*n)
    uv[i-1] = sp.sympify(1)
    return uv


def is_symbol(expr):
    """
    :param expr: any object
    :return: True or False

    avoids the additional test whether an object has the attribute is_Symbol
    """
    return hasattr(expr, 'is_Symbol') and expr.is_Symbol


def is_number(expr, eps=1e-25, allow_complex=False):
    """
    Test whether or not expr is a real (or complex, if explictly stated) number.

    :param expr:                any object
    :param allow_complex:       False (default) or True
    :return: True or False

    Background:
    avoids the additional test whether an arbitrary is a sympy object (has .is_Symbol)
    """

    if isinstance(expr, (str, sp.MatrixBase, list, tuple, np.ndarray, np.matrix, dict)):
        # filter out most likely erroneous calls
        msg = "Unexpected type for is_number: %s." % type(expr)
        raise TypeError(msg)

    try:
        cond = expr.is_Number or expr.is_NumberSymbol
        if cond:
            return True
    except AttributeError:
        pass

    if allow_complex:
        try:
            c = np.complex(expr)
        except TypeError:
            return False
        if c == expr:
            return True
        else:
            msg = "Unexpected behavior: np.complex not equal to original expression"
            raise NotImplementedError(msg)
    try:
        f = float(expr)
    except TypeError:
        return False

    if np.isnan(f) or abs(f) == float('inf'):
        return False

    if f == expr:
        return True

    # Situation: expr can be converted to float but float(expr) differs from expr
    # there might be a problem with different precisions

    L = len(str(expr))
    N = L + 20
    f2 = expr.evalf(n=N)

    if isinstance(f2, sp.Float) and abs((f2 - expr).evalf(N)) < eps:
        return True
    else:
        msg = "Could not decide, whether the following is a number: " + str(expr)
        raise ValueError(msg)


def is_scalar(expr):
    if isinstance(expr, (sp.MatrixBase, np.ndarray)):
        return False
    if isinstance(expr, sp.Basic):
        return True

    return is_number(expr, allow_complex=True)


def symb_vector(*args, **kwargs):
    return sp.Matrix(sp.symbols(*args, **kwargs))


# Todo Unittest (sp.Symbol vs. sp.cls)
def symbMatrix(n, m, s='a', symmetric = 0, cls = sp.Symbol, **kwargs):
    """
    create a matrix with Symbols as each element
    cls might also be sp.Dummy
    """
    A = sp.Matrix(n,m, lambda i,j:cls( s+'%i%i'%(i+1, j+1), **kwargs) )
    if symmetric == 1:
        subs_list = symmetryDict(A)
        A = A.subs(subs_list)
    return A


def symbs_to_func(expr, symbs=None, arg=None):
    """
    in expr replace x by x(arg)
    where x is any element of symbs

    if symbs == None, we assume symbs == expr:
        (useful for conversion of matrices with single-symbol-components)

    arg=None is not allowed
    """

    if symbs == None:
        if  hasattr(expr, 'is_Symbol') and expr.is_Symbol:
            symbs = [expr] # a list with a single Symbol-object
        else:
            assert len(expr)>0
            symbs = expr

    assert all([isinstance(s, sp.Symbol) for s in symbs])
    funcs = [sp.Function(s.name)(arg) for s in symbs]

    # conveniece: type casting
    if isinstance(expr, (tuple, list)):
        expr = sp.Matrix(expr)

    return expr.subs(lzip(symbs, funcs))


# TODO: Unittest
def funcs_to_symbs(expr, funcs=None, symbs=None, arg=None, kwargs = None):
    """
    in expr replace x(arg) by x
    where x is any element of symbs
    (not fully implemented)

    # conveniece: type casting
    if isinstance(expr, (tuple, list)):
        expr = sp.Matrix(expr)
    """
    if not kwargs:
        kwargs = {}

    funcs = list(atoms(expr, sp.function.AppliedUndef))
    symbs = [sp.Symbol(str(type(f)), **kwargs) for f in funcs]

    return expr.subs(lzip(funcs, symbs))


def getOccupation(M):
    """
    maps (m_ij != 0) to every element
    """
    M = sp.sympify(M)
    n, m = M.shape
    tmp = sp.Matrix(n, m, lambda i,j: 1 if not M[i,j]==0 else 0)
    return tmp


def symmetryDict(M):
    """
    erstellt ein dict, was aus einer beliebigen Matrix M
    mittels M.subs(..) eine symmetrische Matrix macht
    """
    n, m = M.shape
    res = {}
    for i in range(1,n):
        for j in range(i):
            # lower triangle
            res[M[i,j]] = M[j,i]

    return res


# TODO: unit test
# TODO: make it intuitivly work in IPyton NB
# (currently up_count=1 is neccessary)
def make_global(*args, **kwargs):
    """
    injects the symbolic variables of a collection to the global namespace
    useful for interactive sessions

    :upcount: is the number of frames to go back;
    upcount = 0 means up to the upper_most frame
    """

    assert len(args) > 0

    upcount = kwargs.pop('upcount', 1)

    if len(kwargs) > 0:
        msg = "The following kwargs are unknown: %s" % ", ".join(list(kwargs.keys()))
        raise ValueError(msg)

    if isinstance(args[-1], int):
        msg = "The signature of this function changed. Now, upcount must be passed separately."
        #raise NotImplementedError(msg)
        warnings.warn(msg)
        upcount = args[-1]
        args = args[:-1]

    varList = []
    for i, a in enumerate(args):
        if isinstance(a, (list, tuple, set, sp.MatrixBase)):
            varList.extend(list(a))
        else:
            msg = "Unexpected type for argument %i: %s" %(i, type(a))
            raise TypeError(msg)

    import inspect

    # get the topmost frame
    frame = inspect.currentframe()
    i = upcount
    while True:
        if frame.f_back == None:
            break
        frame = frame.f_back
        i -= 1
        if i == 0:
            break

    # this is strongly inspired by sympy.var
    try:
        for v in varList:

            if getattr(v, 'is_Function', False):
                v = v.func
            if hasattr(v, 'name'):
                frame.f_globals[v.name] = v
            elif hasattr(v, '__name__'):
                frame.f_globals[v.__name__] = v
            elif is_number(v):
                # conveniece: allow sequences like [0, x1, x2]
                continue

            else:
                raise ValueError( 'Object %s has no name' % str(v) )
    finally:
        # we should explicitly break cyclic dependencies as stated in inspect
        # doc
        del frame

makeGlobal = make_global


def prev(expr, **kwargs):
    """
    sympy preview abbreviation
    """
    KWargs = {'output':'pdf'}
    KWargs.update(kwargs)
    sp.preview(expr, **KWargs)


def mdiff(M, var):
    """
    returns the elementwise derivative of a matrix M w.r.t. var
    """
    return M.applyfunc(lambda elt: sp.diff(elt, var))


# TODO: seems to conflict with zip0
def tup0(xx):
    """
    helper function for substituting.
    takes (x1, x2, x3, ...)
    returns [(x1, 0), (x2, 0), ...]
    """

    return lzip(xx, [0]*len(xx))


def jac(expr, *args):
    if not hasattr(expr, '__len__'):
        expr = [expr]
    return sp.Matrix(expr).jacobian(args)


def kalman_matrix(A, B):
    """
    Kalmans controlability matrix
    """
    A = sp.Matrix(A)
    B = sp.Matrix(B)

    assert A.shape[0] == A.shape[1]
    assert A.shape[0] == B.shape[0]
    assert 1 == B.shape[1]

    n = A.shape[0]

    Q = sp.Matrix(B)
    for i in range(1, n):
        Q = Q.row_join(A**i * B)
        Q = Q.applyfunc(sp.expand)

    return Q

# for backward compatibility:
cont_mat = kalman_matrix

def nl_cont_matrix(vf_f, vf_g, xx, n_extra_cols=0):
    """
    'Controllability' (or 'Reachability') matrix for the nonlinear system based on iterated
     lie bracketts.

    :param vf_f:            drift vector filed ((n x 1) sympy-matrix)
    :param vf_g:            input vector field ((n x 1) sympy-matrix)
    :param xx:              state vector ((n x 1) sympy-matrix)
    :param n_extra_cols:    number of extra columns (higher lie brackets)

    :return:        (n x n) sympy-matrix
    """

    n = len(vf_f)
    assert len(vf_g) == n
    assert len(xx) == n

    Q = sp.Matrix(vf_g)
    ad = sp.Matrix(vf_g).applyfunc(sp.expand)
    for i in range(1, n + n_extra_cols):
        sign = -1
        ad = lie_bracket(sign*vf_f, ad, xx)
        ad = ad.applyfunc(sp.expand)
        Q = Q.row_join(ad)

    return Q

# ---- some functions which are targeted to LTI systems ---


def sympy_to_tf(expr, laplace_var=None):
    """
    Convert a sympy expression to transferfunction object of control package.

    Note that to make this work obviously the package `control` (which ist not a hard dependency of symbtools)
    has to be installed .

    :param expr:
    :param laplace_var:
    :return:    tf object
    """
    expr = sp.sympify(expr)

    symbols = list(expr.atoms(sp.Symbol))
    assert len(symbols) in (0, 1)
    if laplace_var is None:
        # handle also the case when a mere number was passed
        symbols.append(sp.Symbol("s"))
        laplace_var = symbols[-1]

        # if no var was passed, we want prevent misunderstandings where some parameter k is malinterpreted
        assert laplace_var.name == "s"
    num, denom = expr.as_numer_denom()

    cn = to_np(coeffs(num, laplace_var))
    cd = to_np(coeffs(denom, laplace_var))

    try:
        import control
    except ImportError:
        msg = "The package control seems not to be installed but is needed for this function."
        raise ImportError(msg)

    res = control.tf(cn, cd)

    return res


def siso_place(A, b, ev):
    """
    :param A:       system matrix
    :param b:       input vector
    :param ev:      sequence of eigenvalues
    :return:        feedback f such that A + b*f.T has the desired eigenvalues
    """
    n, n2 = A.shape
    assert n == n2
    assert b.shape == (n, 1)
    assert len(ev) == n

    assert all([is_number(v, allow_complex=True) for v in ev])
    assert all([is_number(v) for v in list(A.atoms().union(b.atoms()))])

    coeffs = sp.Matrix(np.poly(ev))[::-1]  # reverse. now: low -> high

    QT = kalman_matrix(A, b).T
    en = sp.zeros(n, 1)
    en[-1, 0] = 1
    qn = QT.solve(en).T

    factor = sp.eye(n)
    res = QT*0#qn*0
    for c in coeffs:
        res += c*factor
        factor = A*factor

    return -(qn*res).T


def get_rows(A):
    """
    returns a list of n x 1 vectors
    """
    A = sp.Matrix(A)
    n, m = A.shape

    return [A[:,i] for i in range(m)]


# ---------------- July 2010 -----------
def elementwise_mul(M1, M2):
    """
    performs element-wise multiplication of matrices
    """
    assert M1.shape == M2.shape
    return sp.Matrix(np.array(M1) * np.array(M2))


#TODO: obsolete?
def poly_occ_matrix(expr, arg1, arg2, n = 2):
    """
    expects expr to be a bivariate polynomial in arg1, arg2 up to order n

    returns a matrix which symbolizes which coefficients are vanishing
    and which are not
    """
    p = sp.Poly(expr, arg1, arg2)
    M = sp.matrices.zeros(n+1)
    star = sp.Symbol('#')
    space = sp.Symbol('.')
    for i in range(n+1):
        for j in range(n+1):
            if p.coeff(i,j) != 0:
                M[i, j] = star
            else:
                M[i, j] = space
    return M


def get_coeff_row(eq, variables):
    """
    takes one linear equation and returns the corresponding row of
    the system matrix
    """
    if not isinstance(eq, equation):
        # assume its the lhs     and rhs = 0
        eq = equation(eq,0)

    if isinstance(variables, sp.Matrix):
        variables = list(variables)

    get_coeff = lambda var: sp.diff(eq.lhs(), var)
    coeffs =  list(map(get_coeff, variables))
    rest = eq.lhs() - sum([coeffs[i]*variables[i] for i in range( len(variables) )])
    coeff_row = list(map(get_coeff, variables)) + [eq.rhs() - rest]
    return coeff_row


def lin_solve_all(eqns):
    """
    takes a list of equations and tries to solve wrt. to all
    occurring symbols
    """
    eqns = sp.Matrix(eqns)

    Vars = list(atoms(eqns, sp.Symbol))

    return lin_solve_eqns(eqns, Vars)


def lin_solve_eqns(eqns, vars):
    """
    takes a list of equation objects
    creates a system matrix of and calls sp.solve
    """
    n = len(eqns)

    vars = list(vars) # if its a vector
    m = len(vars)

    rows = [get_coeff_row(eq, vars) for eq in eqns]

    sysmatrix = sp.Matrix(rows)

    sol = sp.solve_linear_system(sysmatrix, *vars)

    return sol


def lin_solve_eqns_jac(eqns, vars):
    """
    takes a list of equation objects
    creates a system matrix of and calls sp.solve

    # new version !!
    # should replace lin_solve_eqns

    # assumes that eqns is a list of expressions where rhs = 0
    """
    eqm = sp.Matrix(eqns)

    Jac = eqm.jacobian(vars)
    rhs = -eqm.subs(zip0(vars))

    sysmatrix = Jac.row_join(rhs)

    sol = sp.solve_linear_system(sysmatrix, *vars)

    return sol


def solve_linear_system(eqns, vars):
    """
    Tries to solve symbolic equations of the form
    A*x + b = 0
    """

    eqns = sp.Matrix([e for e in eqns if not e == 0])
    vars = sp.Matrix(vars)

    if len(eqns) == 0:
        return []  # no restrictions for vars

    assert eqns.shape[1] == 1

    A = eqns.jacobian(vars)

    if atoms(A).intersection(vars):
        raise ValueError("The equations seem to be non-linear.")

    b = eqns.subs(zip0(vars))

    ns1 = sp.numbered_symbols('aa')
    ns2 = sp.numbered_symbols('bb')

    replm1, (A_cse, ) = sp.cse(A, ns1)
    replm2, (b_cse, ) = sp.cse(b, ns2)

    if eqns.shape[0] < len(vars):
        # assuming full rank there are infinitely many solution
        # we have to find a regular submatrix of A_cse
        # Heuristics: try the "simplest" combination of columns (w.r.t. _extended_count_ops)

        M, N, idcs1, idcs2 = _simplest_regular_submatrix(A_cse)

        y, z = [], []
        for i, v in enumerate(vars):
            if i in idcs1:
                y.append(v)
            else:
                assert i in idcs2
                z.append(v)

        y, z = sp.Matrix(y), sp.Matrix(z)

        rest = N*z + b_cse

        k = M.shape[0] + 1
        rr = sp.Matrix(sp.symbols('r1:%i' % k))

        subs_rest = lzip(rr, rest)

        EQNs_yz = M*y + rr

    sol = sp.solve(EQNs_yz, y)

    assert isinstance(sol, dict)

    replm1.reverse()
    replm2.reverse()

    y_sol = y.subs(sol).subs(subs_rest + replm1 + replm2)

    return lzip(y, y_sol)


def _simplest_regular_submatrix(A):
    """Background: underdetermined system of linear equations
    A*x + b = 0

    We want to find a representation

    M*y + N*z + b = 0

    where M is a combination of columns and a regular Matrix.
    We look for M as simple as possible (w.r.t. _extended_count_ops)

    returns M, N, indices_M, indices_N
    """

    m, n = A.shape
    Anum = subs_random_numbers(A, seed=28)
    if m == n:
        # quick check for generic rank
        assert Anum.rank() == n
        return A

    co = A.applyfunc(_extended_count_ops)
    col_sums = np.sum(to_np(co), axis=0)

    sum_tuples = lzip(col_sums, list(range(n)))
    # [(sum0, 0), (sum1, 1), ...]

    # combinations of tuples
    combs = list( it.combinations(sum_tuples, m) )

    # [ ( (sum0, 0), (sum1, 1) ),   ( (sum0, 0), (sum2, 2) ), ...]

    # now we sort this list of m-tuples of 2-tuples

    def comb_sum(comb):
        # unpack the column sums
        col_sums = lzip(*comb)[0]
        return sum(col_sums)

    combs.sort(key=comb_sum)

    # now take the first column combination which leads to a regular matrix
    for comb in combs:
        idcs = lzip(*comb)[1]
        M = col_select(A, *idcs)

        M_num = subs_random_numbers(M, seed=1107)
        if M_num.rank() == m:
            other_idcs = [i for i in range(n) if not i in idcs]
            N = col_select(A, *other_idcs)
            return M, N, idcs, other_idcs

    raise ValueError("No regular submatrix has been found")




#def lin_solve_eqns(eqns, vars):
#    """
#    takes a list of equation objects
#    creates a system matrix of and calls sp.solve
#    """
#    n = len(eqns)
#
#    vars = list(vars) # if its a vector
#    m = len(vars)
#
#
#
#    #sysmatrix = sp.Matrix(0,0).zeros((n,m+1)) # last col is col of rhs
#
#
#
#    def get_coeff_row(eq):
#        """
#        takes one equation object and returns the corresponding row of
#        the system matrix
#        """
#        get_coeff = lambda var: sp.diff(eq.lhs(), var)
#        coeffs =  map(get_coeff, vars)
#        rest = eq.lhs() - sum([coeffs[i]*vars[i] for i in range( len(vars) )])
#        coeff_row = map(get_coeff, vars) + [eq.rhs() - rest]
#        return coeff_row
#
#    rows = map(get_coeff_row, eqns)
#
#    sysmatrix = sp.Matrix(rows)
#
#
#    sol = sp.solve_linear_system(sysmatrix, *vars)
#
#    return sol


# !! nicht von allgemeinem Interesse

def extract_independent_eqns(M):
    """
    handles only homogeneous eqns

    M Matrix

    returns two lists: indices_of_rows, indices_of_cols

    """

    #S = M # save the symbolical matrix for later use
    M = (np.array(M)!= 0)*1 # Matrix of ones and zeros
    n, m = M.shape

    list_of_occtuples = []

    for i in range(n):
        tmplist = []
        for j in range(m):
            if M[i,j] == 1:
                tmplist.append(j)

        list_of_occtuples.append(tuple(tmplist))

    print(list_of_occtuples)

    #list_of_occtuples_save = list_of_occtuples[:]

    s0 = set(list_of_occtuples[0])

    list_of_rows=[0]

    while True:
        i=-1
#        print "s0="
#        print s0
#        print
        end = False
        for ot in list_of_occtuples:
            i+=1
            if i in list_of_rows:
                continue

#            print i
            if s0.intersection(ot) != set():
                s0 = s0.union(ot)
                #print " >",i
                list_of_rows.append(i)
                break # restart for loop


#        if end == True:
        if i == len(list_of_occtuples)-1:
            #print "durch"
            break

    s0 = list(s0)
    return sorted(list_of_rows), sorted(s0)


def cancel_rows_cols(M, rows, cols):
    """
    cancel rows and cols form a matrix

    rows ... rows to be canceled
    cols ... cols to be canceled
    """

    idxs_col = list(range(M.shape[1]))
    idxs_row = list(range(M.shape[0]))

    rows.sort(reverse=True)
    cols.sort(reverse=True)

    for c in cols:
        idxs_col.pop(c)

    for r in rows:
        idxs_row.pop(r)

#    all_coeffs = sp.Matrix(np.array(all_coeffs)[idxs_col])

    tmp = np.array(M)[idxs_row, :]
    M = sp.Matrix(tmp[:, idxs_col])

    return M


def norm2(v):
    """
    assumes that v is a n x 1 matrix (column vector)
    returns (v.T*v)[0,0] i.e. the squared 2-norm
    """
    assert isinstance(v, sp.Matrix)
    return (v.T*v)[0,0]


# TODO: Doctest/unittest
def concat_cols(*args):
    """
    takes some col vectors and aggregetes them to a matrix
    """

    col_list = []

    for a in args:
        if isinstance(a, list):
            # convenience: interpret a list as a column Matrix:
            a = sp.Matrix(a)
        if not a.is_Matrix:
            # convenience: allow stacking scalars
            a = sp.Matrix([a])
        if a.shape[1] == 1:
            col_list.append( list(a) )
            continue
        for i in range(a.shape[1]):
            col_list.append( list(a[:,i]) )
    m = sp.Matrix(col_list).T

    return m

# other name:
col_stack = concat_cols


def col_split(A, *indices):
    """
    returns a list of columns corresponding to the passed indices
    """
    if not indices:
        indices = list(range(A.shape[1]))
    res = [ A[:, i] for i in indices ]
    return res


def row_split(A, *indices):
    """
    returns a list of rows corresponding to the passed indices
    """
    if not indices:
        indices = list(range(A.shape[0]))
    res = [ A[i, :] for i in indices ]
    return res


# TODO: Doctest
def concat_rows(*args):
    """
    takes some row (hyper-)vectors and aggregetes them to a matrix
    """

    row_list = []

    for a in args:
        if isinstance(a, list):
            # convenience: interpret a list as a row-Matrix:
            a = sp.Matrix(a).T
        if not a.is_Matrix:
            a = sp.Matrix([a])
        if a.shape[0] == 1:
            row_list.append( list(a) )
            continue
        for i in range(a.shape[0]):
            row_list.append( list(a[i, :]) )
    m = sp.Matrix(row_list)

    return m

# other name:
row_stack = concat_rows

# geschrieben für Polynommatritzen

def col_minor(A, *cols, **kwargs):
    """
    returns the minor (determinant) of the columns in cols
    """
    n, m = A.shape

    method = kwargs.get('method', "berkowitz")

    assert m >= n  # -> fat matrix (more total cols)
    assert len(cols) == n

    M = sp.zeros(n, n)
    for i, idx in enumerate(cols):
        M[:, i] = A[:, idx]

    return M.det(method = method).expand()


def general_minor(A, rows, cols, **kwargs):
    """
    selects some rows and some cols of A and returns the det of the resulting
    Matrix
    """

    method = kwargs.get('method', "berkowitz")

    Q = row_col_select(A, rows, cols)

    return Q.det(method = method).expand()


def all_k_minors(M, k, **kwargs):
    """
    returns all minors of order k of M

    Note that if k == M.shape[0]

    this computes all "column-minors"
    """
    m, n = M.shape

    assert k<= m
    assert k<= n

    row_idcs = list(it.combinations(list(range(m)), k))
    col_idcs = list(it.combinations(list(range(n)), k))

    rc_idx_tuples = list(it.product(row_idcs, col_idcs))

    method = kwargs.get('method', "berkowitz")

    res = []
    for rr, cc in rc_idx_tuples:
        res.append(general_minor(M, rr, cc, method = method))

    return res

def row_col_select(A, rows, cols):
    """
    selects some rows and some cols of A and returns the resulting Matrix
    """

    Q1 = sp.zeros(A.shape[0], len(cols))
    Q2 = sp.zeros(len(rows), len(cols))

    for i, c in enumerate(cols):
        Q1[:, i] = A[:, c]


    for i, r in enumerate(rows):
        Q2[i, :] = Q1[r, :]

    return Q2




def col_select(A, *cols):
    """
    selects some columns from a matrix
    """
    Q = sp.zeros(A.shape[0], len(cols))

    for i, c in enumerate(cols):
        Q[:, i] = A[:, c]

    return Q



#def is_left_coprime2(Ap, eps=1e-1):
#    """
#    Alternativer Zugang
#    TODO: merge
#    """

#
#    nonzero_numbers = []
#    root_list = []
#    for m in minors:
#        if m.is_number and m != 0:
#            nonzero_numbers.append(m)
#            # we could return true already here!!
#        else:
#            root_list.append(roots(m))
#
#    if len(nonzero_numbers) > 0:
#        return True
#
#    # test is there a common zero in all root-lists?






def is_left_coprime(Ap, Bp=None, eps = 1e-10):
    """
    Test ob Ap,Bp Linksteilerfrei sind
    keine Parameter zulässig

    """

# folgendes könnte die Berechnung vereinfachen
#    r1, r2 = Ap.shape
#
#    assert r1 <= r2
#
#    minors = all_k_minors(Ap, r1)
#
#    minors = list(set(minors)) # make entries unique


    #print "Achtung, BUG: Wenn ein minor konstant (auch 0) ist kommt ein Fehler"
    r1, r2 = Ap.shape
    if Bp == None:
        # interpret the last columns of Ap as Bp
        Bp = Ap[:, r1:]
        Ap = Ap[:, :r1]
        r1, r2 = Ap.shape


    assert r1 == r2
    r = r1
    r1, m =  Bp.shape
    assert r1 == r
    assert m <= r

    M = (Ap*1).row_join(Bp)

    symbs = list(matrix_atoms(M, sp.Symbol))
    assert len(symbs) == 1
    symb = symbs[0]

    combinations = it.combinations(list(range(r+m)), r)

    minors = [col_minor(M, *cols) for cols in combinations]

    nonzero_const_minors = [m for m in minors if (m !=0) and not m.has(symb)]

    if len(nonzero_const_minors) > 0:
        return True

    #zero_minors = [m for m in minors if m == 0]

    # polymionors (rows belong together)
    all_roots = [roots(m) for m in minors if m.has(symb)]

    # obviously all minors where zeros
    if len(all_roots) == 0:
        return False

    # in general the arrays in that list have differnt lenght
    # -> append a suitable number of roots at inf

    max_len = max([len(a) for a in all_roots])
    root_list = [np.r_[a, [np.inf]*(max_len-len(a))] for a in all_roots]

    all_roots = np.array(root_list)

    # now testing if some finite root is common to all minors
    for i in range(all_roots.shape[0]):
        test_roots = all_roots[i, :]

        other_roots = np.delete(all_roots, i, axis = 0)

        for tr in test_roots:
            if tr == np.inf:
                continue

            min_dist = np.min(np.abs(other_roots-tr), axis = 1)
            if np.all(min_dist < eps):
                # the test root was found in all other minors

                print("critical root:", tr)
                return False

    return True






def series(expr, var, order):
    """
    taylor expansion at zero (without O(.) )
    """
    if isinstance(expr, sp.Matrix):
        return type(expr)([series(x, var, order) for x in expr])

    # expr is scalar
    expr = expr.series(var, 0, order).removeO()
    p = sp.Poly(expr, var, domain='EX')
    s = 0

    #!! limit the order (due to a sympy bug this is not already done)
    for i in range(order+1):
        s+= p.nth(i) * t**i

    return s


def hoderiv(f, x, N=2):
    """
    computes a H igher O rder derivative of the vectorfield f

    Result is a tensor of type (N,0)

    or a  n x L x ... x L (N times) hyper Matrix

    (represented a (N+1)-dimensional numpy array

    """

    import itertools as it

    assert f.shape[1] == 1

    n = f.shape[0]
    L = len(list(x))

    res = np.zeros([n]+[L]*N)

    res= res * sp.Symbol('dummy')

    idx_list = [0]*(N)
    i = 0
    k = 0


    list_of_idcs = list(it.product(*[list(range(L))]*N))

    # example: [(0, 0), (0, 1), (1, 0), (1, 1)]

    for fi in f:
        #print fi, i
        for idcs in list_of_idcs:
            pos = tuple([i]+list(idcs))

            tmp = fi
            #print pos
            for j in idcs:
                #print x[j]
                tmp = tmp.diff(x[j])

            #print "---\n"
#            res.itemset(pos, k)
            res.itemset(pos, tmp)
            k+=1
            #print "  .. ", i, k

        i+=1

    return res


def get_expr_var(expr, var=None):
    """
    auxillary function
    if var == None returns the unique symbol which is contained in expr:
    if no symbol is found, returns None
    """
    expr = sp.sympify(expr)
    if not var == None:
        assert isinstance(var, sp.Symbol)
        return var
    else: # var == None
        symbs = list(expr.atoms(sp.Symbol))
        if len(symbs) == 0:
            return None
        elif len(symbs) == 1:
            return symbs[0]
        else:
            errmsg = "%s contains more than one variable: %s " % (expr, symbs)
            raise ValueError(errmsg)


def poly_degree(expr, var=None):
    """
    returns degree of monovariant polynomial
    """
    var = get_expr_var(expr, var)
    if not var:
        return sp.sympify(0)

    P = sp.Poly(expr, var, domain = "EX")
    return P.degree()


def poly_coeffs(expr, var=None):
    """
    returns all (monovariant)-poly-coeffs (including 0s) as a list
    first element is highest coeff.
    """
    var = get_expr_var(expr, var)
    if var == None:
        return [expr]

    P = sp.Poly(expr, var, domain="EX")

    pdict = P.as_dict()

    d = P.degree()

    return [pdict.get((i,), 0) for i in reversed(range(d+1))]


def coeffs(expr, var = None):
    # TODO: besser über as_dict
    # TODO: überflüssig wegen poly_coeffs?
    """if var == None, assumes that there is only one variable in expr"""
    expr = sp.sympify(expr)
    if var == None:
        vars = [a for a in list(expr.atoms()) if a.is_Symbol]
        if len(vars) == 0:
            return [expr] # its a constant
        assert len(vars) == 1
        var=vars[0]
        dom = 'RR'
    else:
        dom = 'EX'
    return sp.Poly(expr, var, domain =dom).all_coeffs()


# TODO: harmonize with poly_coeffs
def poly_expr_coeffs(expr, variables, maxorder=2):
    """
    :param expr: expression assumed to be a polynomial in `variables`
    :param variables: the independend variables
    :param maxorder maximum order to look for
    :return: a dictionary like Poly.as_dict (except that it is a default-dict)

    Background: For expressions where the coefficients are too big, the
    conversion to a sp.Poly object takes too long. We calculate the coeffs
    by differentiating and substitution
    """

    dd = col.defaultdict
    if not hasattr(variables, '__len__'):
        assert variables.is_Symbol
        variables = [variables]
    N = len(variables)
    assert int(maxorder) == maxorder

    # defaultdict needs a callable as constructor arg. This factory is called
    # each time when the default value is needed (no matching key present)
    # int() -> 0
    result = col.defaultdict(int)

    v0 = zip0(variables)

    orders = list(range(1, maxorder + 1))

    diff_list, order_tuples = get_diffterms(variables, orders, order_list=True)
    # -> lists like [(x1,x1), (x1, x2), ...], [(2, 0, 0), (1, 1, 0), ...]

    # special case: order 0
    key = (0,)*N
    value = expr.subs(v0)
    result[key] = value

    for diff_tup, order_tup in lzip(diff_list, order_tuples):
        gamma = sp.Mul(*[sp.factorial(o) for o in order_tup])
        tmp = expr.diff(*diff_tup).subs(v0) / gamma
        result[order_tup] = tmp

    return result


def monomial_from_signature(sig, variables):
    """
    :param sig: tuple of integers, example (2, 1)
    :param variables: sequence of symbols example (x, y)
    :return: multivariant monomial example (x**2*y)
    """
    assert len(sig) == len(variables)

    res = 1
    for i, v in lzip(sig, variables):
        res *= v**i

    return res


def rat_if_close(x, tol=1e-10):
    s = sp.sympify(x)

    maxden = int(tol**-1 / 10.0)
    f  = fractionfromfloat(x, maxden)
    r = sp.Rational(f.numerator, f.denominator)
    if abs(r-x) < tol:
        return r
    else:
        return x


# TODO: this function should be renamed: approx_rationalize_close_numbers
def rationalize_expression(expr, tol=1e-10):
    """
    substitutes real numbers occuring in expr which are closer than tol to a
    rational with a sufficiently small denominator with these rationals

    usefull special case 1.2346294e-15 -> 0

    """
    a = list(expr.atoms(sp.Number))
    b = [rat_if_close(aa, tol) for aa in a]

    return expr.subs(lzip(a,b))


def rationalize_all_numbers(expr):
    """
    converts all numbers in expr to sp.Rational-objects. This does not change Integers
    :param expr:
    :return:
    """
    numbers_atoms = list(expr.atoms(sp.Number))
    rationalized_number_tpls = [(n, sp.Rational(n)) for n in numbers_atoms]
    return expr.subs(rationalized_number_tpls)


def matrix_with_rationals(A):
    A = sp.Matrix(A)

    def rat(x):
        y = fractionfromfloat(x)
        return sp.Rational(y.numerator, y.denominator)

    A2 = A.applyfunc(rat)

    # error

    diff = A-A2
    a_diff = np.abs(to_np(diff))

    A3 = np.array(A2) # (dtype=object)

    res = np.where(a_diff < 1e-10, A3, to_np(A))

    return sp.Matrix(res)

arr_float = np.frompyfunc(np.float, 1,1)


def to_np(arr, dtype=np.float):
    """ converts a sympy matrix in a nice numpy array
    """
    if isinstance(arr, sp.Matrix):
        symbs = list(matrix_atoms(arr, sp.Symbol))
        assert len(symbs) == 0, "no symbols allowed"

    # because np.int can not understand sp.Integer
    # we temporarily convert to float
    # TODO: make this work with complex numbers..
    arr1 = arr_float( np.array(arr) )
    return np.array( arr1, dtype )


def roots(expr):
    import scipy as sc
    return sc.roots(coeffs(expr))


def real_roots(expr):
    import scipy as sc
    r = sc.roots(coeffs(expr))
    return np.real( r[np.imag(r)==0] )


def zeros_to_coeffs(*z_list, **kwargs):
    """
    calculates the coeffs corresponding to a poly with provided zeros
    """

    s = sp.Symbol("s")
    p = sp.Mul(*[s-s0 for s0 in z_list])

    real_coeffs = kwargs.get("real_coeffs", True)
    c = np.array(coeffs(p, s), dtype=np.float)

    if real_coeffs:
        c = np.real(c)
    return c


def fac(i):
    # see also sp.factorial
    if i == 0:
        return 1
    return i * fac(i-1)


def div(vf, x):
    """divergence of a vector field"""
    vf = list(vf)
    x = list(x)
    assert len(vf) == len(x)

    return sum([c.diff(xi) for c,xi in lzip(vf, x)])


def chop(expr, tol = 1e-10):
    """suppress small numerical values"""

    expr = sp.expand(sp.sympify(expr))
    if expr.is_Symbol: return expr

    assert expr.is_Add

    return sp.Add(*[term for term in expr.as_Add() if sp.abs(term.as_coeff_terms()[0]) >= tol])


# TODO: obsolete?
def trigsimp2(expr):
    """
    sin**2 + cos**2 = 1 in big expressions
    """

    expr = expr.expand()

    trigterms_sin = list(expr.atoms(sp.sin))
    trigterms_cos = list(expr.atoms(sp.cos))


    trigterm_args = []

    # gucken, ob cos auch vorkommt
    for tts in trigterms_sin:
        arg = tts.args[0]
        if trigterms_cos.has(sp.cos(arg)):
            trigterm_args.append(arg)



    for s in trigterm_args:
        poly = trig_term_poly(expr, s)

        dd = poly.as_dict()


        uncommon_coeff = (dd[(2,0)] - dd[(0,2)]).expand()



        if uncommon_coeff == 0:
            expr += dd[(2,0)] - dd[(2,0)]*sp.sin(s)**2 - dd[(2,0)]*sp.cos(s)**2


        print(dd[(2,0)])

    return expr.expand()


def introduce_abreviations(M, prefix='A', time_dep_symbs=[]):
    """returns a matrix with the same shape, but with complicated expressions substituted
    by new symbols. Additionally returns two lists: one for the reverse substitution and
    one which contains only those of the new symbols which are time dependent.

    :param M:                   symbolic matrix
    :param prefix:              prefix-string for the new symbols
    :param time_dep_symbs:      sequence of time dependend symbols
    :return: M_new, subs_tuples, new_time_dep_symbs
    """

    M_new = (M*1).as_mutable()
    Nc, Nr = M.shape

    all_symbols = []
    diff_symbols = []
    subs_tuples = []

    gen = sp.numbered_symbols(prefix)

    for i, j in it.product(list(range(Nc)), list(range(Nr))):
        elt0 = M_new[i, j]
        if not is_number(elt0):
            symb = next(gen)
            all_symbols.append(symb)
            M_new[i, j] = symb
            subs_tuples.append((symb, elt0))
            if depends_on_t(elt0, 't', time_dep_symbs):
                diff_symbols.append(symb)

    return M_new, subs_tuples, sp.Matrix(diff_symbols)


# TODO: rename to rev_tuple_list
def rev_tuple(tuples):
    """
    :param tup: a sequence of 2-tuples
    :return: a list of 2-tuples where each tuple has the reversed order

    this is useful for reverting variable substitution, e.g. in the context of
    coordinate transformation etc.
    """
    return [(t[1], t[0]) for t in tuples]


def gradient(scalar_field, xx):
    # returns a row vector (coverctorfiel)!
    return sp.Matrix([scalar_field]).jacobian(xx)


def trig_term_poly(expr, s):
    """
    s ... the argument of sin, cos
    """

    X, Y = sp.symbols('tmpX_, tmp_Y')

    poly = sp.Poly( expr.subs([(sp.sin(s), X), (sp.cos(s),Y)]), X,Y, domain='EX')

    return poly


def re(M):
    """extend sympy.re to matrix classes (elementwise)
    """
    if isinstance(M, sp.MatrixBase):
        return M.applyfunc(sp.re)
    else:
        return sp.re(M)


def im(M):
    """extend sympy.im to matrix classes (elementwise)
    """
    if isinstance(M, sp.MatrixBase):
        return M.applyfunc(sp.im)
    else:
        return sp.im(M)




def matrix_count_ops(M, visual=False):
    def co(expr):
        return count_ops(expr, visual=visual)
    return M.applyfunc(co)


def count_ops(expr, *args, **kwargs):
    """
    Matrix aware wrapper for sp.count_ops

    In difference to the sympy version this function only returns 0
    if the operand is equal to 0, otherwise it returns at least 1 (for atoms)
    """

    if isinstance(expr, sp.MatrixBase):
        return matrix_count_ops(expr, *args, **kwargs)
    else:
        res = sp.count_ops(expr, *args, **kwargs)
        if expr == 0:
            return res
        else:
            return res + 1


def get_diffterms(xx, order, order_list=False):
    """
    returns a list such as

    [(x1, x1), (x1, x2), (x1, x3), (x2, x2), (x2, x3), (x3, x3)]

    :param xx: example: xx = (x1, x2, x3)
    :param order: example: order =2

    :param order_list: flag whether or not to return an additional index list
      like [(2, 0, 0), (1, 1, 0), ...]
    :return:
    """

    if order == 0:
        return []

    if isinstance(order, (list, tuple)):
        if not order_list:
            return sum([get_diffterms(xx, o) for o in order], [])
        else:
            terms, terms_indices = get_diffterms(xx, order[0], order_list=order_list)
            if len(order) > 1:
                t2, ti2 = get_diffterms(xx, order[1:], order_list=order_list)
                terms += t2
                terms_indices += ti2
            return terms, terms_indices

    assert isinstance(order, int)

    terms = list(it.combinations_with_replacement(xx, order))

    if order_list:
        terms_indices = []
        for tup in terms:
            element = [tup.count(x) for x in xx]
            terms_indices.append( tuple(element) )

        return terms, terms_indices
    return terms


def multi_series(expr, xx, order, poly=False):
    """
    Reihenentwicklung (um 0) eines Ausdrucks in mehreren Variablen
    """

    xx0 = lzip(xx, [0]*len(xx)) # Entwicklungsstelle
    res = 0
    for i in range(order+1):
        if i == 0:
            res += expr.subs(xx0)
            continue
        terms = get_diffterms(xx, i)
        for tup in terms:
            cnt = Counter(tup) # returns a dict
            fac_list = [sp.factorial(n) for n in list(cnt.values())]
            #fac = 1/sp.Mul(*fac_list)
            res += expr.diff(*tup).subs(xx0)*sp.Mul(*tup) / sp.Mul(*fac_list)

    if poly:
        res = sp.Poly(res, *xx, domain="EX")
    return res


def matrix_series(m, xx, order, poly=False):

    assert isinstance(m, sp.Matrix)
    def appfnc(expr):
        return multi_series(expr, xx, order, poly)

    return m.applyfunc(appfnc)


def linear_input_trafo(B, row_idcs):
    """
    serves to decouple inputs from each other
    """
    B = sp.Matrix(B)

    n, m = B.shape

    assert len(row_idcs) == m
    assert len(set(row_idcs)) == m

    Bnew = sp.zeros(m,m)
    for i, idx in enumerate(row_idcs):
        Bnew[i, :] = B[idx, :]

    P = symbMatrix(m,m)

    leqs = Bnew*P-sp.eye(m)
    res = sp.solve(leqs)

    return to_np(P.subs(res))


def poly_scalar_field(xx, symbgen, orders, poly=False):
    """
    returns a multivariate poly with specified oders
    and symbolic coeffs

    :param xx:          independent variables
    :param symbgen:     generator for coeffs
    :param orders:      int or list of occurring orders
    :param poly:        flag whether or not to return a sp.Poly object (instead of sp.Expr)

    :return:            poly, <column vector of the coefficients>
    """

    if isinstance(orders, int):
        orders = [orders]
    elif isinstance(orders, (list, tuple, sp.Matrix)):
        orders = list(orders)

    res = 0
    coeff_list = []
    for i in orders:
        if i == 0:
            c = next(symbgen)
            res += c
            coeff_list.append(c)
            continue

        terms = get_diffterms(xx, i)

        for tup in terms:
            c = next(symbgen)
            res += c*sp.Mul(*tup)
            coeff_list.append(c)

    if poly:
        res = sp.Poly(res, *xx, domain='EX')
    return res, sp.Matrix(coeff_list)


def solve_scalar_ode_1sto(sf, func_symb, flow_parameter, **kwargs):

    assert is_symbol(func_symb)
    iv = kwargs.get('initial_value')
    if iv is None:
        iv = sp.Symbol(str(func_symb)+'_0')

    sf = sp.sympify(sf)
    func = sp.Function(str(func_symb))(flow_parameter)

    eq = func.diff(flow_parameter) - sf.subs(func_symb, func)
    old_atoms = eq.atoms(sp.Symbol)
    res = sp.dsolve(eq, func)
    if isinstance(res, list):
        # multiple solutions might occur (complex case)
        msg = 'Warning: got multiple solutions while solving %s.' \
              'continuing with the first...' % eq
        warnings.warn(msg)
        res = res[0]

    new_atoms = res.atoms(sp.Symbol) - old_atoms

    new_atoms = list(new_atoms)
    assert len(new_atoms) in (1, 2)
    CC = new_atoms

    # handling initial conditions (ic)
    res0 = res.subs(flow_parameter, 0)
    eq_ic = iv - res0.rhs
    sol = sp.solve(eq_ic, CC, dict=True)  # gives a list of dicts
    assert len(sol) >= 1
    if len(sol) > 2:
        msg = 'Warning: multiple solutions for initial values; taking the first.'
        warnings.warn(msg)
    sol = list(sol[0].items())

    # selecting the rhs of the first solution and look for other C-vars
    free_symbols = list(sol[0][1].atoms().intersection(CC))
    if free_symbols:
        msg = 'Warning: there are still some symbols free while calculating ' \
              'initial values; substituting them with 0.'
        warnings.warn(msg)

    res = res.subs(sol).subs(zip0(free_symbols))

    return_iv = kwargs.get('return_iv')
    if return_iv:
        return res.rhs, iv
    else:
        return res.rhs


def calc_flow_from_vectorfield(vf, func_symbs, flow_parameter=None, **kwargs):
    """
    Calculate the flow along a vectorfield by solving ordinary differential equations.
    Note that it is not alway possible to solve odes symbolicaly.
    :param vf:              vectorfiled
    :param func_symbs:      state_vector (functions of time)
    :param flow_parameter:  variable for the time in the solution (optional; default: t)

    :param kwargs:          optional:
                                sol_subs = [(x1, x1_extern_solution), ...]
                                iv_list = [<list of initial values>]

    :return:
    """
    if flow_parameter is None:
        flow_parameter = sp.Symbol('t')

    assert is_symbol(flow_parameter)
    assert len(vf) == len(func_symbs)
    assert all([is_symbol(fs) for fs in func_symbs])
    assert vf.shape[1] == 1

    func_symbs = sp.Matrix(func_symbs)

    # ## build dependency graph
    J = vf.jacobian(func_symbs)

    # find autonomous odes -> jacobian has no entry apart from diagonal
    lines = J.tolist()

    aut_indices = []

    for i, line in enumerate(lines):
        line.pop(i)
        if not any(line):
            # indices of autonomous odes
            aut_indices.append(i)

    sol_subs = kwargs.get('sol_subs', [])
    iv_list = kwargs.get('iv_list', [])
    sol_subs_len = len(sol_subs)

    # just solve the autonomous odes
    for i in aut_indices:
        rhs = vf[i]
        fs = func_symbs[i]
        if sol_subs and fs in lzip(*sol_subs)[0]:
            continue
        sol, iv = solve_scalar_ode_1sto(rhs, fs, flow_parameter, return_iv=True)
        sol_subs.append((fs, sol))
        iv_list.append((fs, iv))

    new_vf = vf.subs(sol_subs)

    if len(sol_subs) == len(func_symbs):
        return func_symbs.subs(sol_subs), flow_parameter, func_symbs.subs(iv_list)

    # If there has not been any progress
    if len(sol_subs) == sol_subs_len:
        raise ValueError("This vectorfield cannot be symbolically integrated with this algorithm")

    return calc_flow_from_vectorfield(new_vf, func_symbs, flow_parameter, sol_subs=sol_subs, iv_list=iv_list)


def reformulate_integral_args(expr):
    """
    This function replaces any indefinite integral like Integral(F(t), t)
     by a definite integral from zero to arg, like Integral(F(t_), (t_, 0, t))
    :param expr: sympy expr (or matrix)
    :return: expr (or matrix) with integrals replaced
    """

    integrals = list(atoms(expr, sp.Integral))
    subs_list = []

    for i in integrals:
        kernel, arg_tup = i.args
        if len(arg_tup) == 3:
            # this is a determined integral -> ignore
            continue
        if len(arg_tup) == 2:
            # unexpected value -> better raise an error
            raise ValueError('semi evaluated integral')
        assert len(arg_tup) == 1
        x = arg_tup[0]
        assert x.is_Symbol
        x_ = sp.Dummy(x.name+'_', **x.assumptions0)
        new_kernel = kernel.subs(x, x_)
        new_int = sp.Integral(new_kernel, (x_, 0, x))
        subs_list.append((i, new_int))

    return expr.subs(subs_list)


def np_trunc_small_values(arr, lim = 1e-10):
    assert isinstance(arr, (np.ndarray, np.matrix))

    bool_abs = np.abs(arr) < lim

    res = arr*1
    res[bool_abs] = 0
    return res


def trunc_small_values(expr, lim = 1e-10, n=1):
    expr = ensure_mutable( sp.sympify(expr) )

    a_list = list(atoms(expr, sp.Number))
    subs = []
    for a in a_list:
        if sp.Abs(a) < lim:
            subs.append((sp.Abs(a), 0))
            # substituting Abs(a) circumvents Problems in terms like sin(...)
            if a < 0:
                subs.append((a, 0))

    res = expr.subs(subs)

    if n <= 1:
        return res
    else:
        return trunc_small_values(res, lim, n-1)


def apply_round(expr, digits=3):
    """
    Apply `round` from python std lib to all Float and Complex instances
    :param expr:
    :param digits:
    :return:
    """

    # taken from here: https://stackoverflow.com/questions/43804701/round-floats-within-an-expression
    if isinstance(expr, sp.MatrixBase):
        def fnc(arg):
            return apply_round(arg, digits=digits)

        return expr.applyfunc(fnc)

    assert is_scalar(expr)

    res = expr

    for a in sp.preorder_traversal(expr):
        # note that complex numbers are Float + Mul(Float, ImaginaryUnit), so they are handled here

        if isinstance(a, sp.Float):
            res = res.subs(a, round(a, digits))

    return res


def clean_numbers(expr, eps=1e-10):
    """
    trys to clean all numbers from numeric noise
    """

    if isinstance(expr, (list, tuple)):
        return [clean_numbers(elt, eps) for elt in expr]

    expr = trunc_small_values(expr)

    maxden = int(1/eps)
    floats = list(atoms(expr, sp.Float))
    rats = []
    dummy_symbs = []
    symb_gen = sp.numbered_symbols('cde', cls = sp.Dummy)
    for f in floats:
        rat = sp_fff(f, maxden)
        rats.append(rat)
        dummy_symbs.append(next(symb_gen))

    res1 = expr.subs(lzip(floats, dummy_symbs))
    res2 = res1.subs(lzip(dummy_symbs, rats))

    return res2


def random_equaltest(exp1, exp2,  info = False, integer = False, seed = None, tol = 1e-14, min=-1, max=1):
    """
    serves to check numerically (with random numbers) whether exp1, epx2 are equal
    # TODO: unit test
    """

    if isinstance(exp1, sp.Matrix):
        assert isinstance(exp2, sp.Matrix)
        assert exp1.shape == exp2.shape, "Different shape"
        m,n = exp1.shape

        def func(exp1, exp2):
            return random_equaltest(exp1, exp2, info, integer, seed,
                                                            tol, min, max)
        res = [func(e1, e2) for e1,e2 in lzip(list(exp1), list(exp2))]

        if info == True:
            res = [tup[1] for tup in res]
        return sp.Matrix(res).reshape(m,n)

    exp1 = sp.sympify(exp1)
    exp2 = sp.sympify(exp2)

    gen = sp.numbered_symbols('ZZZ', cls=sp.Dummy)

    derivs = exp1.atoms(sp.Derivative).union(exp2.atoms(sp.Derivative))

    # we only look for undefined functions, but not for cos(..) etc
    # the latter would be matched by sp.Function
    func_class = sp.function.AppliedUndef

    funcs = exp1.atoms(func_class).union(exp2.atoms(func_class))

    # replace all functions and derivs with symbols
    SL = [(X, next(gen)) for X in list(derivs)+list(funcs)]


    exp1 = exp1.subs(SL)
    exp2 = exp2.subs(SL)



    a1 = exp1.atoms(sp.Symbol, sp.Dummy)
    a2 = exp2.atoms(sp.Symbol, sp.Dummy)

    r = random
    if not seed is None:
        r.seed(seed)

    def get_rand():
        if not integer:
            return (r.random()*(max-min)+min)
        else:
            return r.randint(min, max)

    tuples = [(s, get_rand()) for s in a1.union(a2)]

    if not integer:
        diff = exp1.subs(tuples).evalf() - exp2.subs(tuples).evalf()
    else:
        diff = exp1.subs(tuples) - exp2.subs(tuples)


    if info == False:
        return abs(diff) <= tol
    else:
        return abs(diff) <= tol, diff


def rnd_number_subs_tuples(expr, seed=None, rational=False, prime=False, minmax=None, **kwargs):
    """

    :param expr: expression
    :return: [(a1, r1), (a2, r2), ...]

    where a1, a2, ... are the Symbols occurring in expr
    and r1, r2, ... are random numbers

    keyword args:
    seed:
    rational:       generate sp.Ratioinal instead of sp.Float objects (default: False)
    minmax:         2-tuple: (min_value, max_value)
    prime:          (default: False)
    prec:           evalf-precision (default 100)
    exclude:        symbols to exclude (iterable or single sp.Symbol)
    """

    derivs = list(expr.atoms(sp.Derivative))

    derivs.sort(key=get_sp_deriv_order, reverse=True)  # highest derivatives come first

    # now functions
    # we only look for undefined functions, but not for cos(..) etc
    # the latter would be matched by sp.Function
    func_class = sp.function.AppliedUndef
    funcs = list(expr.atoms(func_class))

    # the challenge is to filter out all functions and derivative objects
    # separatly and later do backsubstitution

    gen = sp.numbered_symbols('ZZZ', cls=sp.Dummy)
    # replace all functions and derivs with symbols
    SL = []
    dummy_symbol_list = []
    for X in list(derivs) + list(funcs):
        dummy = next(gen)
        SL.append( (X, dummy) )
        dummy_symbol_list.append(dummy)

    regular_symbol_list = list(expr.atoms(sp.Symbol))  # original Symbols

    for atom in list(derivs) + list(funcs) + regular_symbol_list:
        if not atom.is_commutative:
            msg = "At least one atom is not commutative." \
                  "Substituting with numbers might be misleading."
            warnings.warn(msg)
            break

    # the order does matter (highest derivative first)
    # inherit the order from the symb number which inherited it from deriv-order
    dummy_symbol_list.sort(key=str)
    atoms_list = dummy_symbol_list + regular_symbol_list

    # for back substitution
    reverse_dict = dict( rev_tuple(SL) +
                         lzip(regular_symbol_list, regular_symbol_list) )

    if not seed is None:
        random.seed(seed)

    if minmax is None:
        min_val, max_val = 0, 1
    else:
        min_val, max_val = minmax

    assert max_val > min_val
    assert float(max_val) == max_val
    assert float(min_val) == min_val

    delta = max_val - min_val

    prec = kwargs.pop('prec', 100)

    def rnd():
        val = random.random()
        return sp.Float(val, prec)

    if prime:
        assert minmax is None
        N = len(atoms_list)
        list_of_primes = prime_list(2*N)  # more numbers than needed
        random.shuffle(list_of_primes)
        tuples = [(reverse_dict[s], list_of_primes.pop()) for s in atoms_list]

    elif rational:
        tuples = [(reverse_dict[s], clean_numbers( rnd()*delta + min_val )) for s in atoms_list]
    else:
        tuples = [(reverse_dict[s], rnd()*delta + min_val) for s in atoms_list]

    exclude = kwargs.get("exclude", [])
    if isinstance(exclude, (sp.Symbol, sp.Function, sp.Derivative)):
        exclude = [exclude]

    remove_idcs = []
    for idx, tup in enumerate(tuples):
        if tup[0] in exclude:
            remove_idcs.append(idx)

    remove_idcs.reverse()
    for idx in remove_idcs:
        tuples.pop(idx)

    return tuples


# TODO: unit test
def rnd_trig_tuples(symbols, seed = None):
    """
    assigns to each element of symbols a value m*sp.pi where m is such
    that sp.sin(m*pi) evaluates to some "algebraic number" like (sqrt(2)/2)
    """
    denoms = [2, 3, 4, 5, 6, 8, 12]
    if seed:
        random.seed(seed)

    L = len(denoms)
    tuples = []
    for s in symbols:
        i = random.randint(0, L-1)
        den = denoms[i]
        num = random.randint(0, den*2)*sp.pi

        tuples.append((s, num/den))

    return tuples


def subs_random_numbers(expr, *args, **kwargs):
    """
    replaces all symbols in the given expr (scalar or matrx) by random numbers
    and returns the substituted result

    usefull for e.g. for checking the rank at a "generic" point

    :param expr:  sympy expression or matrix

    :param \**kwargs:

    :Keyword Arguments:
        * *seed* (``float``) --
          seed for random number generator
        * *round_res_digits* (``int``) --
          number of digitis to round (None -> do not round)
    """

    round_res_digits = kwargs.pop("round_res_digits", None)
    tuples = rnd_number_subs_tuples(expr, *args, **kwargs)
    res = expr.subs(tuples)

    if round_res_digits is not None:
        assert int(round_res_digits) == round_res_digits
        res = apply_round(res, digits=int(round_res_digits))

    return res


def generic_rank(M, **kwargs):
    """
    Evaluate the rank of the matrix M by substituting a random number for each symbol,
    function, etc. and then applying the Berkowitz algorithm (the Berk. alg. returns the coeffs
    of the characteristic polynomial, see sympy docs).

    :param M:       Matrix of interest
    :param eps:
    :param kwargs:  prime(=True by default), seed, ...
    :return:        the rank (w.r.t. the numeric tolerance eps)

    see also: rnd_number_subs_tuples

    Background: Let n1, n2 = M.shape and n1 >= n2. Then the generic rank r <= n2.
    Let d:= n2 - r >= 0 be the generic defect of the matrix. It equals the number of vanishing
    singular values, which, in turn, is equal to the number of vanishing coefficients of the
    characteristic polynomial of M.T*M (n2 x n2 matrix)
    """

    assert isinstance(M, sp.MatrixBase)

    types1 = [type(a) for a in M.atoms(sp.Number)]
    if sp.Float in types1:
        msg = "There are Float-Objects contained in the matrix. They are converted to rationals." \
              "To make sure that no harm is done, the data should be converted before passing" \
              "to this function. you can use e.g. rationalize_all_numbers()."
        warnings.warn(msg, UserWarning)

        M = rationalize_all_numbers(M)

    n1, n2 = M.shape
    if n2 > n1:
        M = M.T
        n1, n2 = n2, n1

    # using random prime numbers?
    prime = kwargs.get('prime', False)
    kwargs.update(prime=prime)

    #eps = kwargs.pop('eps', 1e-160)
    seed = kwargs.pop('seed', random.randint(0, 1e5))

    # define the precisions
    prec1, prec2, prec3 = plist = 100, 200, 300
    # rnst1, rnst2, rnst3 = rnst_list = [rnd_number_subs_tuples(M, seed=seed, prec=p) for p in plist]
    # M1, M2, M3 = [M.subs(r).evalf(prec=p) for (r, p) in lzip(rnst_list, plist)]

    rnst = rnd_number_subs_tuples(M, seed=seed, rational=True)
    M1, M2, M3 = [M.subs(rnst).evalf(prec=p) for p in plist]

    # calculate the coeffs of the charpoly
    # berkowitz-method sorts from highes to lowest -> reverse the tuples
    coeffs1 = (M1.T*M1).berkowitz()[-1][::-1]
    coeffs2 = (M2.T*M2).berkowitz()[-1][::-1]
    coeffs3 = (M3.T*M3).berkowitz()[-1][::-1]

    # a coefficient is considered as vanishing if it is
    # a) exactly zero, or
    # b) if its absolute value decreases several orders of magnitudes when
    # the precision increases.

    # count the exact zeros
    # !! obsolete?
    #zero_count = [cl.count(0) for cl in (coeffs1, coeffs2, coeffs3)]

    # find out the first (w.r.t to the index) non-vanishing coeff

    nz_coeffs1 = np.array([c for c in coeffs1 if c != 0], dtype=sp.Float)
    nz_coeffs2 = np.array([c for c in coeffs2 if c != 0], dtype=sp.Float)
    nz_coeffs3 = np.array([c for c in coeffs3 if c != 0], dtype=sp.Float)

    res21_list = []
    res32_list = []

    err_msg = "unexpected behavior of berkowitz coeffs during rank calculation"
    threshold = 1e-2
    for i, (c1, c2, c3) in enumerate(lzip(coeffs1, coeffs2, coeffs3)):
        if 0 in (c1, c2, c3):
            # assume that this coeff indeed vanishes
            continue

        q21 = abs(c2/c1)
        q32 = abs(c3/c2)

        res21 = q21 < threshold
        res32 = q32 < threshold

        if res21 != res32:
            # one precision-step indicates a vanishing coeff
            # while the other does not
            raise ValueError(err_msg)

        if res21 == False:
            # the coeff has not changed sufficiently due to the precision step
            # this means the coeff is not considered to vanish,
            # i.e. it is the first non-vanishing coeff
            break

    # the index of the first False-event gives the number of the "vanishing low coeffs"
    # = number of zero singular values = defect (rank drop)
    # Note: vanishing coeffs after the first non-vanishing coeff can be ignored here

    defect = i
    rank = n2 - defect

    return rank


# TODO: due to generic_rank() this function is obsolete and thus deprecated
def rnd_number_rank(M, **kwargs):
    """
    evaluates the rank of the matrix m by substituting a random number for each symbol, etc.

    :param M:       Matrix of interest
    :param eps:
    :param kwargs:  prime(=True by default), seed, ...
    :return:        the rank (w.r.t. the numeric tolerance eps)

    see also: rnd_number_subs_tuples
    """
    msg = "`rnd_number_rank` is deprecated. Use `generic_rank` instead."
    warnings.warn(msg, DeprecationWarning)

    assert isinstance(M, sp.MatrixBase)

    class iszero_fnc_factory(object):
        """
        Instances of this class are callable.
        They are intended to be used as comparision to zero (w.r.t the threshold eps)
        """
        def __init__(self, eps):
            self.eps = eps

        def __call__(self, x):
            x = sp.sympify(x)
            if x.is_Number:
                res = sp.Abs(x) < self.eps
            else:
                res = x.is_zero

            assert res in (True, False)
            return res

    # using random prime numbers by default
    prime = kwargs.get('prime', True)
    kwargs.update(prime=prime)


    # Problem: some nonzero symbolic expressions in M might lead to very small values
    # (e.g. sin(3)**50 ≈ 3.015e-43 )
    # other expressions which are "symbolically 0" lead to bigger numerical expressions:
    # sp.sin(3.32)**2 + sp.cos(3.32)**2 -1 -> 1.11...e-16

    # Solution: convert the expressions to floats with medium (-> M1) and high (-> M2) precision.
    # identify a threshold for eps (used for iszerofunc in M1.rank()) where the rank increases.
    # Then look at M2.rank for the same eps
    # see also the unit tests for illustration


    # nod means number of digits
    nod1 = kwargs.pop('nod1', 100)
    nod2 = nod1 + 200

    rnst = rnd_number_subs_tuples(M, **kwargs)
    M1 = M.subs(rnst).evalf(prec=nod1)
    M2 = M.subs(rnst).evalf(prec=nod2)

    if 0:

        rank_list1 = []
        eps_list = [10**-k for k in range(10, 200, 10)]

        for eps in eps_list:
            rank_list1.append(M1.rank(iszerofunc=iszero_fnc_factory(eps)))

        if rank_list1[-1] == rank_list1[0]:
            return rank_list1[-1]
        last_change_index = np.where(np.diff(rank_list1))[-1][-1] + 1

        eps_krit = eps_list[last_change_index]

        r = M2.rank(iszerofunc=iszero_fnc_factory(eps_krit))

    else:
        r = M2.rank(iszerofunc=iszero_fnc_factory(1e-190))

    return r


def zip0(*xx, **kwargs):
    """
    returns a list of tuples like: [(x1, arg), (x2, arg), ...]
    this is very handy for substiution of equilibrium points at zero

    For convenience it is also possible to pass a sequence of sequences of
    symbols:
    zip0([a1, a2, a3], [x1, x2])

    valid kwargs: arg (if a different value than zero is desired)
    """
    arg = kwargs.get('arg', 0)

    res = []
    for x in xx:
        if hasattr(x, '__len__'):
            res.extend( zip0(*x, **kwargs) )
        else:
            assert isinstance(x, (sp.Symbol, sp.Function, sp.Derivative))
            res.append((x, arg))

    return res


def aux_make_tup_if_necc(arg):
    """
    checks whether arg is iterable.
    if not return (arg,)
    """

    # allow iterators and sympy-Matrices
    # TODO: test for iterators
    if not ( isinstance(arg, Iterable) or hasattr(arg, '__len__') ):
        return (arg, )

    return arg

#TODO:
"""
https://github.com/sympy/sympy/wiki/Release-Notes-for-0.7.6

Previously lambdify would convert Matrix to numpy.matrix by default.
This behavior is being deprecated, and will be completely phased out with
the release of 0.7.7. To use the new behavior now set the modules
kwarg to [{'ImmutableMatrix': numpy.array}, 'numpy'].
If lambdify will be used frequently it is recommended to wrap it with a
partial as so:
lambdify =
functools.partial(lambdify, modules=[{'ImmutableMatrix': numpy.array}, 'numpy']).
For more information see #7853 and the lambdify doc string.
"""


def expr_to_func(args, expr, modules='numpy', **kwargs):
    """
    wrapper for sympy.lambdify to handle constant expressions
    (shall return a numpyfied function as well)

    this function bypasses the following problem:

    f1 = sp.lambdify(t, 5*t, modules = "numpy")
    f2 = sp.lambdify(t, 0*t, modules = "numpy")

    f1(np.arange(5)).shape # -> array
    f2(np.arange(5)).shape # -> int


    Some special kwargs:
    np_wrapper      (default False):
                    the return-value of the resulting function is passed through
                    to_np(..) before returning

    eltw_vectorize: allows to handle vectors of piecewise expression (default=True)

    keep_shape:     (default False)
                    Flag to ensure that the result has the same shape as the input

    """

    # TODO: sympy-Matrizen mit Stückweise definierten Polynomen
    # numpy fähig (d.h. vektoriell) auswerten

    keep_shape = kwargs.get("keep_shape", False)

    expr = sp.sympify(expr)
    expr = ensure_mutable(expr)
    expr_tup = aux_make_tup_if_necc(expr)
    arg_tup = aux_make_tup_if_necc(args)

    new_expr = []
    arg_set = set(arg_tup)

    # be prepared for the case that the args might not occur in the expression
    # constant function (special case of a polynomial)
    for e in expr_tup:
        assert isinstance(e, sp.Expr)
        # args (Symbols) which are not in that expression
        diff_set = arg_set.difference(e.atoms(sp.Symbol))

        # add and subtract the respective argument such that it occurs
        # without changing the result
        for d in diff_set:
            assert isinstance(d, sp.Symbol)
            e = sp.Add(e, d, -d, evaluate = False)

        new_expr.append(e)

    # if not hasattr(expr, '__len__'):
    #     assert len(new_expr) == 1
    #     new_expr = new_expr[0]

    # warn if expr contains symbols which are not in args
    unexpected_symbols = []

    for xpr in expr_tup:
        unexpected_symbols.extend(xpr.atoms(sp.Symbol).difference(arg_tup))
    unexpected_symbols = list(set(unexpected_symbols))  # dismiss duplicates
    unexpected_symbols.sort(key=lambda s: str(s))
    # noinspection PySimplifyBooleanCheck
    if unexpected_symbols != []:
        msg = "the following symbols were in expr, but not in args:\n{}\n"
        warnings.warn(msg.format(unexpected_symbols))

    # TODO: Test how this works with np_wrapper and vectorized arguments
    if hasattr(expr, 'shape'):
        new_expr = sp.Matrix(new_expr).reshape(*expr.shape)
        sympy_shape = expr.shape
    else:
        sympy_shape = None

    # extract kwargs specific for lambdify
    printer = kwargs.get('printer', None)
    use_imps = kwargs.get('use_imps', True)
    func = sp.lambdify(args, new_expr, modules, printer, use_imps)


    if kwargs.get('np_vectorize', False):
        func1 = np.vectorize(func)
    else:
        func1 = func

    if kwargs.get('eltw_vectorize', True):
        # elementwise vectorization to handle piecewise expressions
        # each function returns a 1d-array
        assert len(new_expr) >=1 # elementwise only makes sense for sequences
        funcs = []
        for e in new_expr:
            func_i = sp.lambdify(args, e, modules, printer, use_imps)
            func_iv = np.vectorize(func_i, otypes=[np.float])
            funcs.append(func_iv)

        def func2(*allargs):
            # each result should be a 1d- array
            results = [to_np(f(*allargs)) for f in funcs]

            # transpose, such that the input axis (e.g. time) is the first one
            res = to_np(results).T.squeeze()
            if not hasattr(res, '__len__'):
                res = float(res)  # scalar results: array(5.0) -> 5.0
            return res

    elif kwargs.get('np_wrapper', False):
        def func2(*allargs):
            return to_np(func1(*allargs))
    elif kwargs.get('list_wrapper', False):
        def func2(*allargs):
            return list(func1(*allargs))
    else:
        func2 = func1

    if keep_shape:
        def reshape_func(*allargs):
            return func2(*allargs).reshape(sympy_shape)

        func3 = reshape_func
    else:
        func3 = func2

    return func3


def ensure_mutable(arg):
    """
    ensures that we handle a mutable matrix (iff arg is a matrix)
    """
    # TODO: e.g. sp.sympify converts a MutableMatrix to ImmutableMatrix
    # maybe this changes in future sympy releases
    # which might make this function obsolete (?)
    if isinstance(arg, sp.matrices.MatrixBase):
        return as_mutable_matrix(arg)
    else:
        return arg


def as_mutable_matrix(matrix):
    """
    sympy sometimes converts matrices to immutable objects
    this can be reverted by a call to    .as_mutable()
    this function provides access to that call as a function
    (just for cleaner syntax)
    """
    return matrix.as_mutable()


def is_col_reduced(A, symb, return_internals = False):
    """
    tests whether polynomial Matrix A is column-reduced

    optionally returns internal variables:
        the list of col-wise max degrees
        the matrix with the col.-wise-highest coeffs (Gamma)

    Note: concept of column-reduced matrix is important e.g. for
    solving a Polynomial System w.r.t. highest order "derivative"

    Note: every matrix can be made col-reduced by unimodular transformation
    """
    Gamma = as_mutable_matrix(A*0)
    n, m = A.shape

    assert n == m

    A = trunc_small_values(A)

    # degrees:
    A_deg = to_np(matrix_degrees(A, symb), dtype = np.float)
    max_degrees = list(A_deg.max(axis=0)) # columnwise maximum

    # TODO: unit-Test
    # handle zero columns:
    infty = float(sp.oo)
    max_degrees = [int(md) if not md == -infty else md for md in max_degrees]

    # maximum coeffs:
    for j in range(m):
        deg = max_degrees[j]
        for i in range(n):
            Gamma[i,j] = get_order_coeff_from_expr(A[i,j], symb, deg)

    result = Gamma.rank() == m
    if return_internals:
        # some functions might need this information
        internals = Container(Gamma = Gamma, max_degrees = max_degrees)
        return result, internals
    else:
        return result


def is_row_reduced(A, symb, *args, **kwargs):
    """
    transposed Version of is_col_reduced(...)
    """
    res = is_col_reduced(A.T, symb, *args, **kwargs)
    if isinstance(res, tuple):
        C = res[0]
        C.Gamma = C.Gamma.T
    return res


def get_col_reduced_right(A, symb, T = None, return_internals = False):
    """
    Takes a polynomial matrix A(s) and returns a unimod Transformation T(s)
    such that   A(s)*T(s) (i.e. right multiplication) is col_reduced.

    Approach is taken from appendix of the PHD-Thesis of S. O. Lindert (2009)

    :args:
        A:  Matrix
        s:  symbol
        T:  unimod-Matrix from preceeding steps

    -> recursive approach

    :returns:
        Ar: reduced Matrix
        T:  unimodular transformation Matrix
    """

    n, m = A.shape
    assert n == m

    if T == None:
        T = sp.eye(n)
    else:
        assert T.shape == (n, m)
        d = T.berkowitz_det().expand()
        assert d != 0 and not d.has(symb)

    # noinspection PyPep8Naming
    A_work = trunc_small_values(sp.expand(A*T))

    cr_flag, C = is_col_reduced(A_work, symb, return_internals = True)

    # C.Gamma is the matrix with col-wise highest coeff
    if cr_flag:
        # this is the only exit point
        res = A_work.expand(), T
        if return_internals:
            res += (C,)
        return res
    else:
        pass
        # C.Gamma is nonregular

    g = C.Gamma.nullspace()[0]
    # noinspection PyPep8Naming
    non_zero_cols_IDX = to_np(g).flatten() != 0
    # get the max_degrees wrt. to each non-zero component of g
    non_zero_cols_degrees = to_np(C.max_degrees)[non_zero_cols_IDX]

    N = max(non_zero_cols_degrees)
    # construct the diagonal matrix
    diag_list = []
    for i in range(m):
        cd = col_degree(A_work[:, i],symb)
        diag_list.append( symb**int(N-cd) )

    # gamma_p:
    gp = sp.diag(*diag_list)*g

    T1 = unimod_completion(gp, symb)

    TT = trunc_small_values( sp.expand(T*T1) )

    # recall this method with new T

    return get_col_reduced_right(A, symb, TT, return_internals)


def get_order_coeff_from_expr(expr, symb, order):
    """
    example:
        3*s**2 -4*s + 5, s, 3 -> 0
        3*s**2 -4*s + 5, s, 2 -> 3
        3*s**2 -4*s + 5, s, 1 -> -4
        3*s**2 -4*s + 5, s, 9 -> 0
    """
    p = sp.Poly(expr, symb, domain = "EX")
    default = 0
    return p.as_dict().get( (order,), default )


def element_deg_factory(symb):
    """
    returns a function for getting the polynomial degree of an expr. w.r.t.
    a certain symbol
    """
    def element_deg(expr):
        return poly_degree(expr, symb)

    return element_deg


def matrix_degrees(A, symb):

    element_deg = element_deg_factory(symb)

    return A.applyfunc(element_deg)


def col_degree(col, symb):
    return max(matrix_degrees(col, symb))


def unimod_completion(col, symb):
    """
    takes a column and completes it such that the result is unimodular
    """

    # there must at least one nonzero constant in col:

    n, m = col.shape
    assert m == 1
    element_deg = element_deg_factory(symb)

    idx = None
    for i, c in enumerate(list(col)):
        if c != 0 and element_deg(c) == 0:

        # we want the index of the first non-zero const. of col
            idx = i
            break

    assert not idx == None, "there should have been a nonzero const."


    T = sp.eye(n)

    T[:, idx] = col

    return T


def subs_same_symbs(expr, new_symbs):
    """
    subs_same_symbs(x+y, [x, y])
    returns x+y, where the symbols are taken from the list
    (symbs in exp might be different objects with the same name)

    this functions helps if expr comes from a string

    """

    old_symbs = list(atoms(expr, sp.Symbol))

    string_dict = dict([(s.name, s) for s in new_symbs])

    subs_list = [ (s, string_dict[s.name]) for s in old_symbs]

    return expr.subs(subs_list) # replpace new symbs by old ones


def symm_matrix_to_vect(M):
    """ converts
     a b
     b c
            to      [a, b, c]
    """

    n, m = M.shape
    assert m == n
    assert M == M.T

    res = sp.zeros(int(n+.5*n*(n-1)), 1)
    k = 0
    for i in range(n):
        for j in range(i, n):
            if i == j:
                val = M[i,j]
            else:
                val = 2*M[i,j]
            res[k] = val
            k += 1

    return res


# todo: unit test
def rpinv(M):
    """compute symbolic right pseudo inverse"""
    # http://stackoverflow.com/questions/15426624/computing-pseudo-inverse-of-a-matrix-using-sympy
    M = sp.Matrix(M)
    n1, n2 = M.shape
    #if n2 > n1:
    # more columns -> assume full row rank
    # right inverse
    rpinv = M.T * (M * M.T).inv()
    res = M*rpinv

    return rpinv


def lpinv(M):
    """compute symbolic left pseudo inverse"""
    # http://stackoverflow.com/questions/15426624/computing-pseudo-inverse-of-a-matrix-using-sympy
    M = sp.Matrix(M)
    n1, n2 = M.shape

    # assume full column rank
    # left inverse
    lpinv = (M.T * M).inv() * M.T
    res = lpinv*M

    # print res
    return lpinv


# todo: unit test, (split up into rpinv und lpinv?)
def pinv(M):
    """compute symbolic pseudo inverse"""
    # http://stackoverflow.com/questions/15426624/computing-pseudo-inverse-of-a-matrix-using-sympy
    M = sp.Matrix(M)
    n1, n2 = M.shape
    if n2 > n1:
        # more columns -> assume full row rank
        # right inverse
        pinv = M.T * (M * M.T).inv()
        res = M*pinv

    else:
        #assume full column rank
        # left inverse
        pinv = (M.T * M).inv() * M.T
        res = pinv*M

    #print res
    return pinv


# noinspection PyPep8Naming
def nullspaceMatrix(M, *args, **kwargs):
    """
    wrapper for the sympy-nullspace method
    returns a Matrix where each column is a basis vector of the nullspace
    additionally it uses the enhanced nullspace function to calculate
    ideally simple (i.e. fraction free) expressions in the entries
    """

    n = enullspace(M, *args, **kwargs)
    return col_stack(*n)


# todo: (idea) extend penalty to rational and complex numbers
def _extended_count_ops(expr):
    """
    extended count ops, which penalizes symbols over numbers
    """
    expr = sp.sympify(expr)
    res = sp.count_ops(expr)
    if res > 0:
        return res
    if expr.is_Symbol:
        return res+.5
    return res


# todo: unit test-example P = sp.Matrix([-x2, -x1, 1]).T (-> )
def enullspace(M, *args, **kwargs):
    """
    enhanced nullspace: calculate basis vectors with ideally simple
    (i.e. fraction free) expressions in the entries

    :kwargs:
        simplify = True (apply simplification before examining denominators)

    """

    if kwargs.get('sort_rows', False) and not 0 in M.shape:
        # it might help sympy to calculate the symbolic nullspace
        # if the rows are sorted w.r.t. their complexity
        # simplest rows first
        rows = M.tolist()
        rows.sort( key=lambda row: sum( count_ops(sp.Matrix(row)) ) )
        M = sp.Matrix(rows)

    # ensure that the key is in
    kwargs['sort_rows'] = None
    # and pop it out (avoid sp.Matrix,nullspace complaints)
    kwargs.pop('sort_rows')

    # two different targets for kwargsuments -> create a copy
    spns_kwargs = dict(kwargs)

    if kwargs.get('simplify') is False:
        # the user does not want to apply simplify at all
        # this has consequences on 2 levels:
        # sp.nullspace and this function
        # -> sp.nullspace expects a callable
        # create a separat kwargs structure with a
        # dummy function which does nothing

        empty_simplify_func = lambda arg: arg

        spns_kwargs['simplify'] = empty_simplify_func

    vectors = M.nullspace(*args, **spns_kwargs)

    if kwargs.get('simplify', True):
        custom_simplify = nullspace_simplify_func
        if custom_simplify is None:
            custom_simplify = sp.simplify
        else:
            assert custom_simplify(sp.cos(1)**2 + sp.sin(1)**2) == 1

        vectors = [custom_simplify(v) for v in vectors]

    new_vectors = []
    for v in vectors:
        # get the "most complicated" denomitator of the coordinates
        denoms = [ c.as_numer_denom()[1] for c in v]
        #denom_tuples = [(d.count_ops(), d) for d in denoms]
        #denom_tuples.sort()
        denoms.sort( key=_extended_count_ops )

        d = denoms[-1]
        # convert to mutable matrix
        res = sp.Matrix(v*d)
        new_vectors.append( res )

    return new_vectors


def vect_to_symm_matrix(v):
    """ converts
     [a, b, c]

     to    a b
           b c
    """

    v = sp.Matrix(list(v))
    L, m = v.shape
    assert m == 1
    n = -.5 + sp.sqrt(.25+2*L)

    if not int(n) == n:
        raise ValueError("invalid length")
    n = int(n)

    M = sp.zeros(n,n)
    k = 0
    for i in range(n):
        for j in range(i, n):
            if i == j:
                M[i,j] = v[k]
            else:
                M[i,j] = v[k]/2
                M[j,i] = v[k]/2
            k+=1

    return M


# ! eigentlich eher numtools

def dd(*args):
    """
    dd(a,b,c, ...) = np.dot(a, np.dot(b, np.dot(c, ...)))
    """
    return reduce(np.dot, args)


def sorted_eigenvalues(M, **kwargs):
    """
    returns a list of eigenvalues ordered by their real part in decreasing order

    :param M:      square matrix (entries are assumed to be numbers)
    :return:       ordered list of eigenvalues
    """

    assert isinstance(M, sp.MatrixBase)
    assert M.is_square

    for elt in list(M):
        assert is_number(elt)

    # get triples (eigenvalue, multiplicity, L)
    # where L is a list of corresponding eigenvectors
    eig_triples = M.eigenvects()

    def key_func(trip):
        # take the real part of ther first element
        return sp.re(trip[0])

    # decreasing order of the real parts
    eig_triples.sort(key=key_func, reverse=True)

    if kwargs.get('get_triples', False):
        return eig_triples

    res = []
    for value, multiplicity, vects in eig_triples:
        for k in range(multiplicity):
            res.append(value)

    return res


def sorted_eigenvector_matrix(M, numpy=False, increase=False, eps=1e-14, **kwargs):
    """
    returns a matrix whose columns are the normalized eigenvectors of M

    :M:         square matrix (entries are assumed to be numbers)
    :numpy:     use numyp instead of sympy
    :increase:  sort in increasing order (default False)
    :return:    V (Matrix whose cols are the eigenvectors corresponding to the sorted ev.)
    """

    assert increase in (True, False)

    if numpy:
        # use numpy
        N = to_np(M)
        ev, V = np.linalg.eig(N)
        idcs = np.argsort(ev)
        if not increase:
            idcs = idcs[::-1] # reverse

        ev = ev[idcs]
        V = V[:, idcs]

        diff = abs(np.diff(ev))
        if min(diff) < 1e-5:
            msg = "There might be an eigenvalue with algebraic multiplicity > 1. " \
                  "Handling of such cases is currently not implemented for the numpy approach."
            raise NotImplementedError(msg)

        # remove numeric noise
        realpart  = np.real(V)*1.0
        imagpart  = np.imag(V)*1.0

        realpart[np.abs(realpart) < eps] = 0
        imagpart[np.abs(imagpart) < eps] = 0
        V2 = realpart + 1j*imagpart
        return sp.Matrix(V2)

    # np_flag was false

    # get the triples in decresing order
    sorted_triples = sorted_eigenvalues(M, get_triples=True, **kwargs)
    if increase:
        sorted_triples = sorted_triples[::-1]  # reverse

    cols = []
    for value, multiplicity, vects in sorted_triples:
        for v in vects:
            cols.append(v/sp.sqrt((v.T*v)[0]))

    V = col_stack(*cols)
    return V


## !! Laplace specific
def do_laplace_deriv(laplace_expr, s, t, tds=None):
    """
    Convenience function to aplly powers of the Laplace s operator to expressions in time domain.

    laplace_expr : symbolic expression containing symbols s and t
    s : Symbol used for the Laplace operator  (usually something like sp.Symbol("s"))
    t : Symbol used for the time variable  (usually something like sp.Symbol("t"))
    tds: (optional) sequence of time dependend symbols (passed to time_deriv)

    Examples:
    do_laplace_deriv( s*(t**3 + 7*t**2 - 2*t + 4), s, t ) ->  3*t**2  +14*t - 2
    do_laplace_deriv( 4*s*x1 + (-s**2 + 1)*a*x2, s, t, [x1, x2]) ->  4*xdot1 + a*x2 - a*xddot2
    """

    laplace_expr = sp.sympify(laplace_expr)
    if isinstance(laplace_expr, sp.Matrix):
        return laplace_expr.applyfunc(lambda x: do_laplace_deriv(x, s,t))

    exp = laplace_expr.expand()

    #assert isinstance(exp, sp.Add)

    P = sp.Poly(exp, s, domain = "EX")
    items = list(P.as_dict().items())

    if not tds:
        tds = []

    res = 0
    for key, coeff in items:
        exponent = key[0] # exponent wrt s

        res += time_deriv(coeff, tds, order=exponent)

    return res


def sca_integrate(f, x):
    """
    special case aware integrate

    :param f: expression to be integrated
    :param x: variable
    :return: F with the property: F.diff(x) == f

    Background: sympy.integrate sometimes gives results which are technically correct
    but unnecessary complicated example:
     integrate(tan(x), x) -> -log(sin(x)**2 - 1)/2
     expand_log(integrate(tan(x), x).trigsimp(), force=True)
     -log(cos(x)) - I*pi/2 #  i.e. imaginary integration constant (which can be ignored)

    This function handles such special cases to provide "cleaner results".
    """
    assert is_symbol(x)

    w1, w2, w3, w4 = ww = sp.symbols('w1, w2, w3, w4', cls=sp.Wild)

    # first special case
    f = sp.trigsimp(f)
    thematch = f.match(w1*sp.tan(w2*x + w3) + w4)  # -> dict or None
    if thematch and not thematch.get(w1) == 0 and not thematch.get(w2, 0).has(x):

        w4_int = sca_integrate(w4.subs(thematch), x)
        result = (-w1/(w2)*sp.log(sp.cos(w2*x + w3))).subs(thematch) + w4_int

        # eliminate the remaining Wildcard Symbols
        result = result.subs(zip0(ww))
        return result

    # no special case matches
    else:
        return sp.integrate(f, x)


def smart_integrate(expr, var, **kwargs):
    """
    Integrate expressions with the awareness of derivative symbols
    """

    expr2 = replace_deriv_symbols_with_funcs(expr)

    res1 = 0
    if isinstance(expr2, sp.Add):
        # handle each summand separately
        for arg in expr2.args:
            res1 += sp.integrate(arg, var, **kwargs)

    else:
        res1 = sp.integrate(expr2, var, **kwargs)

    # keep integrals which have not beed simplified
    # (dont apply backsubstitution)

    int_expressions = list(res1.atoms(sp.Integral))
    INT_symbs = sp.numbered_symbols("INT", cls=sp.Dummy)

    int_subs = lzip(int_expressions, INT_symbs)
    int_back_subs = rev_tuple(int_subs)

    res1b = res1.subs(int_subs)

    rplmts = get_custom_attr_map("ddt_func")
    rplmts.sort(key=lambda x: get_sp_deriv_order(x[1]), reverse=True)
    rplmts2 = rev_tuple(rplmts)
    res2 = res1b.subs(rplmts2)

    res3 = res2.subs(int_back_subs)

    return res3


def replace_deriv_symbols_with_funcs(expr, return_rplmts=False):
    """
    Iterate through all symbols of expression and substitute them with their respective func
    :param expr:            expression
    :param return_rplmts:   flag whether to return the replacements
    :return:        replaced expressions
    """

    symbs = list(expr.atoms(sp.Symbol))
    symbs.sort(key=lambda x: x.difforder, reverse=True)
    items = [(s, s.ddt_func) for s in symbs]
    res = expr.subs(items)

    if return_rplmts:
        return res, items
    else:
        return res



def simplify_derivs(expr):
    """
    iterates over Derivative-atoms and performs "doit"
    (workarround for a sympy functionality)
    """
    A = list(atoms(expr, sp.Derivative))

    def key_fnc(a):
        """
        get the length of the args to determine the order of the derivative
        """
        return len(a.args)

    # highest first
    A.sort(key = key_fnc, reverse=True)

    SUBS = [(a, a.doit()) for a in A]
#    print SUBS
    return expr.subs(SUBS)


def depends_on_t(expr, t, dependent_symbols=[]):
    """
    Returns whether or not an expression depends (implicitly) on the independed variable t

    :param expr: the expression to be analysed
    :param t: symbol of independet variable
    :param dependendt_symbols: sequence of implicit time dependent symbols
                            default: []
                            if it is set to None this function returns always False

    :return: True or False
    """

    # This is usefull for "naively" perform right_shifting, i.e. ignoring any time_dependence
    if dependent_symbols is None:
        return False

    satoms = atoms(expr, sp.Symbol)

    if t in satoms:
        return True

    res = False
    for a in satoms:
        if a in dependent_symbols:
            return True
        if is_derivative_symbol(a):
            return True

    return False


# noinspection PyPep8Naming
@recursive_function
def dynamic_time_deriv(thisfunc, expr, vf_Fxu, xx, uu, order=1):
    """
    Calculate the time derivative along the solutions of the ode
     xdot = F(x, u). This adds input-derivatives as needed

    :param thisfunc:    implicit argument, automatically passed by decorator `recursive_function`
    :param expr:        expression
    :param vf_Fxu:      input dependent vector field
    :param xx:          state vector
    :param uu:          input symbol or vector
    :param order:       derivative order (default: 1)
    :return:
    """

    if isinstance(expr, sp.MatrixBase):
        def tmpfunc(entry):
            return thisfunc(entry, vf_Fxu, xx, uu, order=order)
        return expr.applyfunc(tmpfunc)

    if order == 0:
        return expr

    if order > 1:
        expr = thisfunc(expr, vf_Fxu, xx, uu, order=order-1)

    if isinstance(uu, sp.Symbol):
        uu = sp.Matrix([uu])
    # be sure that we have a matrix
    uu = sp.Matrix(uu)

    # find out order of highest input derivative in expr
    uu_diffs = get_all_deriv_childs(uu)
    input_derivatives = row_stack(uu, uu_diffs)
    next_input_derivatives = time_deriv(input_derivatives, uu)

    result = gradient(expr, xx) * vf_Fxu
    assert result.shape == (1, 1)
    result = result[0, 0]

    for u, udot in zip(input_derivatives, next_input_derivatives):
        result += expr.diff(u)*udot
    return result


def get_symbols_by_name(expr, *names):
    """
    convenience function to extract symbols from expressions by their name
    :param expr: expression or matrix
    :param *names: names of the desired symbols
    :return: a list of)symbols matching the names
    (if len == 1, only return the symbol)
    """

    symbols = atoms(expr, sp.Symbol)
    items = [(s.name, s) for s in symbols]
    d = dict(items)

    res_list = []
    for n in names:
        res = d.get(n)
        if not res:
            raise ValueError("no Symbol with name %s in expr: %s" % (n, expr))
        res_list.append(res)

    if len(res_list) == 1:
        return res_list[0]
    return res_list


def update_cse(cse_subs_tup_list, new_subs):
    """

    :param cse_subs_tup_list: list of tuples: [(x1, a+b), (x2, x1*b**2)]
    :param new_subs: list of tuples: [(a, b + 5), (b, 3)]
    :return: list of tuples [(x1, 11), (x2, 99)]

    useful to substitute values in a collection returned by sympy.cse
    (common subexpressions)
    """
    res = []
    for e1, e2 in cse_subs_tup_list:
        new_tup = (e1, e2.subs(res + new_subs))
        res.append(new_tup)
    return res


# TODO: (multipl. by integers (especially -1) should be preserved by default)
def symbolify_matrix(M):
    """
    associates a symbol to every matrix entry
    respects equal values, keeps atoms unchanged


    Example: f1, ... f4 are expressions, a1 is a symbol

    Input:

    f1+f2       a1         0    7
     a1     f3*sin(f4)   f1+f2  a1

    Output:

    [(x0, f1+f2), (x1, f1+f2)]

    x0 a1 0  7
    a1 x1 x0 a1
    """
    assert isinstance(M, sp.Matrix)
    L = list(M)

    gen = sp.numbered_symbols(cls = sp.Dummy)
    replaced = []
    new_symbol = []
    result = []

    for e in L:
        if e.is_Atom:
            # leave the entry unchanged
            ns = e
        elif replaced.has(e):
            # get the old symbol
            ns = new_symbol[replaced.index(e)]
        else:
            replaced.append(e)
            # create a new symbol
            ns = next(gen)
            new_symbol.append(ns)

        result.append(ns)

    res = sp.Matrix(result).reshape(*M.shape)
    replacements = lzip(new_symbol, replaced)
    return replacements, res


# TODO: this should live in a separate class
# noinspection PyPep8Naming
class SimulationModel(object):
    """
    This class encapsulates all data pertaining a nonlinear state-space model
    (parameter values, state-dimension, number of inputs)
    """

    # TODO:
    # Decide whether a procedural interface (function instead of class)
    # would be better.
    def __init__(self, f, G, xx, model_parameters=None,):
        """
        'Constructor' method

        :param f: drift vector field
        :param G: matrix whose columns are the input vector fields
        :param xx: the state
        :param model_parameters: dict (or tuple-list) for the numerical values
        of the parameters
        """
        self.f = sp.Matrix(f)
        self.G = sp.Matrix(G)
        if model_parameters is None:
            self.mod_param_dict = {}
        else:
            self.mod_param_dict = dict(model_parameters)

        assert G.shape[0] == f.shape[0]
        self.state_dim = f.shape[0]
        self.input_dim = G.shape[1]
        self.xx = xx

        # instance_variables for rhs_funcs
        self.f_func = None
        self.G_func = None
        self.u_func = None
        self.use_sp2c = None
        self.compiler_called = None

    @staticmethod
    def exceptionwrapper(fnc):
        """
        prevent the integration algorithm to get stuck if
        a exception occurs in rhs
        """

        def newfnc(*args, **kwargs):
            # print(args)
            try:
                res = fnc(*args, **kwargs)
                return res
            except Exception as e:
                import traceback
                traceback.print_exc()
                import sys
                sys.exit(1)

        return newfnc

    def _get_input_func(self, kwargs):

        if kwargs.get("free_input_args", False):
            # this option leads to rhs(xx, uu, time)
            return None

        input_function = kwargs.get('input_function')
        controller_function = kwargs.get('controller_function')

        if input_function is None and controller_function is None:
            zero_m = np.array([0]*self.input_dim)

            def u_func(xx, t):
                return zero_m
        elif not input_function is None:
            assert hasattr(input_function, '__call__')
            assert controller_function is None

            def u_func(xx, t):
                return input_function(t)
        else:
            assert hasattr(controller_function, '__call__')
            assert input_function is None

            u_func = controller_function

        tmp = u_func([0]*self.state_dim, 0)
        tmp = np.atleast_1d(tmp)
        if not len(tmp) == self.input_dim:
            msg = "Invalid result dimension of controller/input_function."
            raise TypeError(msg)

        return u_func

    def create_simfunction(self, **kwargs):
        """
        Creates the right-hand-side function of xdot = f(x) + G(x)u

        signature is adapted to scipy odeint: rhs(state, time)
        exception: see `free_input_args` below


        :kwargs:

        :param controller_function: callable u(x, t)
        this can be a controller function,
        a desired trajectory (x being ignored -> open loop)
        or a zero-function to simulate the autonomous system xdot = f(x).
        As default a zero-function is used

        :param input_function: callable u(t)
        shortcut to pass only open-loop control

        :param free_input_args: boolean;
        True -> rhs has signature: rhs(state, input, time)

        :param use_sp2c: boolean flag whether to use sympy to c bridge (default: False)

        Note: input_function and controller_function mutually exclude each other
        """

        self.u_func = self._get_input_func(kwargs)
        use_sp2c = bool(kwargs.get("use_sp2c", False))

        if callable(self.f_func) and callable(self.G_func) and use_sp2c == self.use_sp2c:
            # this is just an update (of input function)
            return self._produce_sim_function()

        self.use_sp2c = use_sp2c
        n = self.state_dim

        f = self.f.subs(self.mod_param_dict)
        G = self.G.subs(self.mod_param_dict)

        # find unexpected symbols:
        ue_symbols_f = atoms(f, sp.Symbol).difference(set(self.xx))
        ue_symbols_G = atoms(G, sp.Symbol).difference(set(self.xx))

        errmsg = "The following unexpected symbols where found in {}: {}"
        if ue_symbols_f:
            raise ValueError(errmsg.format("`f`", ue_symbols_f))
        elif ue_symbols_G:
            raise ValueError(errmsg.format("`G`", ue_symbols_G))

        assert atoms(f, sp.Symbol).issubset( set(self.xx) )
        assert atoms(G, sp.Symbol).issubset( set(self.xx) )

        if use_sp2c:
            import sympy_to_c as sp2c
            self.f_func = sp2c.convert_to_c(self.xx, f, cfilepath="vf_f.c", use_exisiting_so=False)
            self.G_func = sp2c.convert_to_c(self.xx, G, cfilepath="matrix_G.c", use_exisiting_so=False)
            self.compiler_called = True

        else:
            self.f_func = expr_to_func(self.xx, f, np_wrapper=True)
            self.G_func = expr_to_func(self.xx, G, np_wrapper=True, eltw_vectorize=False)
            self.compiler_called = False

        rhs = self._produce_sim_function()

        # handle exceptions which occur inside
        # rhs = self.exceptionwrapper(rhs)
        return rhs

    def _produce_sim_function(self):
        # load the (possibly compiled) functions
        # this is faster than resolve `self.` in every step
        f_func = self.f_func
        G_func = self.G_func
        u_func = self.u_func

        if u_func is not None:
            # this is the usual case

            # do not use self.f_func here because resolution of `self.xyz` is slow
            def rhs(xx, time):
                xx = np.ravel(xx)
                uu = np.ravel(u_func(xx, time))
                ff = np.ravel(f_func(*xx))
                GG = G_func(*xx)

                xx_dot = ff + np.dot(GG, uu)

                return xx_dot
        else:
            # user wants to pass input (uu) by themselves (kwarg: "free_input_args")

            # noinspection PyUnusedLocal
            def rhs(xx, uu, time):
                xx = np.ravel(xx)

                ff = np.ravel(f_func(*xx))
                GG = G_func(*xx)

                xx_dot = ff + np.dot(GG, uu)

                return xx_dot

        return rhs

    def num_trajectory_compatibility_test(self, tt, xx, uu, rtol=0.01, **kwargs):
        """ This functions accepts 3 arrays (time, state, input) and tests, whether they are
        compatible with the systems dynamics of self

        :param tt: time array
        :param xx: state array
        :param uu: input array
        :param rtol: relative tolerance w.r.t abs_max of xdot_num

        further kwargs:
        'full_output': also return the array of residual values
        """

        assert tt.ndim == 1
        assert xx.ndim == 2
        if uu.ndim == 1:
            uu = uu.reshape(-1, 1)
        assert xx.shape[0] == uu.shape[0] == len(tt)
        assert xx.shape[1] == self.state_dim
        assert uu.shape[1] == self.input_dim

        dt = tt[1] - tt[0]
        xdot_num = np.diff(xx, axis=0)/dt

        threshold = np.max(np.abs(xdot_num))*rtol * self.state_dim

        f = self.f.subs(self.mod_param_dict)
        G = self.G.subs(self.mod_param_dict)

        f_func = expr_to_func(self.xx, f, np_wrapper=True)
        G_func = expr_to_func(self.xx, G, np_wrapper=True, eltw_vectorize=False)

        N = len(tt) - 1

        res = np.zeros(N)
        eqnerr_arr = np.zeros((N, 2))

        for i in range(N):
            x = xx[i,:]
            xd = xdot_num[i,:]
            u = uu[i, :]

            ff = np.ravel(f_func(*x))
            GG = G_func(*x)

            # calculate the equation error (should be near zero)
            eqnerr = xd - ff - np.dot(GG, u)
            assert eqnerr.ndim == 1 and len(eqnerr) == self.state_dim

            res[i] = np.linalg.norm(eqnerr)

        # two conditions must be fulfilled:
        # 1.: less than 10% of the residum values are "medium big"
        medium_residua_bool = res > threshold*0.1  # boolean array
        nbr_medium_residua = sum(medium_residua_bool)
        cond1 = nbr_medium_residua < N*0.1

        # 2.: less than 1% of the residum values are "big"
        big_residua_bool = res > threshold  # boolean array
        nbr_big_residua = sum(big_residua_bool)
        cond2 = nbr_big_residua < N*0.01

        cond_res = cond1 and cond2

        if kwargs.get('full_output'):
            return cond_res, res
        else:
            return cond_res


# eigene Trigsimp-Versuche
# mit der aktuellen sympy-Version (2013-03-29)
# eigentlich überflüssig
# TODO: in Schnipsel-Archiv überführen
def sort_trig_terms(expr):

    expr = expr.expand()
    assert type(expr) == sp.Add

    trig_terms = list(expr.atoms(sp.sin, sp.cos))

    res = {}

    for a in expr.args:
        coeff = a.subs(lzip(trig_terms, [1]*len(trig_terms)))


        sign = 1

        # Vorzeichen in auf die Funktion verschieben
        if str(coeff)[0] == "-":
            sign = -1
            coeff*=-1
        else:
            sign = 1

        L = res.get(coeff)
        if L == None:
            L = []
            res[coeff] = L
        L.append( a.subs(coeff, 1) ) # minus is already considered

    return res


def simp_trig_dict(sdict):
    """
    takes a sorted dict, simplifies each value and adds all up
    """

    items = list(sdict.items())

    res = 0
    for k, v in items:
        res += k*sp.trigsimp(sum(v))

    return res


def my_trig_simp(expr):
    """
    ersetzt größere argumente in trigonom funktionen durch einfache Symbole
    """

    trig_terms = list(expr.atoms(sp.sin, sp.cos))


    args = []

    for tt in trig_terms:
        args.append(tt.args[0])

    args = list(set(args))
    args.sort(key = sp.count_ops)

    symbs = sp.symbols( 'x1:%i' %( len(args)+1) )

    subslist = lzip(args, symbs)
    subslist.reverse()

    return expr.subs(subslist), subslist


def gen_primes():
    """ Generate an infinite sequence of prime numbers.
    """
# Source:
# http://stackoverflow.com/questions/1628949/to-find-first-n-prime-numbers-in-python

    # Maps composites to primes witnessing their compositeness.
    # This is memory efficient, as the sieve is not "run forward"
    # indefinitely, but only as long as required by the current
    # number being tested.
    #
    D = {}

    # The running integer that's checked for primeness
    q = 2

    while True:
        if q not in D:
            # q is a new prime.
            # Yield it and mark its first multiple that isn't
            # already marked in previous iterations
            #
            yield q
            D[q * q] = [q]
        else:
            # q is composite. D[q] is the list of primes that
            # divide it. Since we've reached q, we no longer
            # need it in the map, but we'll mark the next
            # multiples of its witnesses to prepare for larger
            # numbers
            #
            for p in D[q]:
                D.setdefault(p + q, []).append(p)
            del D[q]

        q += 1


def prime_list(n):
    a = gen_primes()
    res = [next(a) for i in range(n)]
    return res


# after deletion of depricated functions here we store some
# some information about them and their successors
# git blame might also help

# def make_pw(var, transpoints, fncs, ignore_warning=False):
#     if not ignore_warning:
#         msg = "This function is deprecated. Use create_piecewise(...) "\
#         "with slightly different syntax."
#         raise DeprecationWarning(msg)
#
#
# def crow_split(*args):
#     raise DeprecationWarning('use row_split(..) instead')
#
#
# def matrix_random_equaltest(M1, M2,  info=False, **kwargs):
#     raise DeprecationWarning("use random_equaltest instead")
#
#
# def matrix_random_numbers(M):
#     raise DeprecationWarning("use subs_random_numbers")
#
#
# def perform_time_derivative(*args, **kwargs):
#
#     msg = "This function name is deprecated. Use time_deriv instead. "
#     #raise DeprecationWarning, msg
#     warnings.warn(msg)
#     1/0
#
#     return time_deriv(*args, **kwargs)
