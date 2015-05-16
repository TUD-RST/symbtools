# -*- coding: utf-8 -*-


import sympy as sp
from sympy import Matrix, sin, cos
from model_tools import generate_model, Rz
from symb_tools import symbs_to_func, zip0
import symb_tools as st

import pickle

from ipHelp import IPS, Tracer, ip_syshook, sys

#ip_syshook(1)


"""
Collection of Code for noncommutative calculations ("s = d/dt")
"""

# time
t = sp.Symbol('t', commutative = False)

# Laplace-Variable (d/dt)
s = sp.Symbol('s', commutative = False)


def apply_deriv(term, power, s, t):
    res = 0
    assert int(power) == power
    if power == 0:
        return term


    # !!stimmt das??
    res = 0
    for i in range(power):
        tmp = term.diff(t) + term*s
        res += apply_deriv(tmp, power-1, t, s).expand()

    tmp = term.diff(t) + term*s
    res = apply_deriv(tmp, power-1, s, t).expand()

    return res


def right_shift(mul, s, t, max_pow = 4):
    """
    Vorgehen:
        index des ersten Auftauchens einer s-Potenz finden
        -> mul = L *s**p * X* R0
        -> s**p auf X anwenden -> es entstehen p + 1 Rest-Terme
        -> Auf diese wird die Funktion rekursiv angewendet
        -> Rückgabewert ist Mul oder Add von Muls wo die s-Terme ganz rechts stehen
    """

    assert isinstance(s, sp.Symbol)

    # nothing to do
    if s not in mul:
        return mul

    if not mul.expand().count_ops() == mul.count_ops():
        raise ValueError, 'mul expression must be expanded:'+ str(mul)

    if not isinstance(mul, sp.Mul):
        if mul == s:
            return mul
        elif isinstance(mul, sp.Pow) and mul.args[0] == s:
            return mul
        else:
            raise ValueError, 'Expected Mul, Symbol or Pow (like s**2), not ' +  str(mul)
    assert isinstance(mul, sp.Mul)
    assert s.is_commutative == False

    s_terms = [s**(i+1) for i in range(max_pow)]
    args = mul.args

    if not t in mul:
        # aggregate all s-Terms at the right margin
        # handle: s*a*s**2*b*s*c -> a*b*c*s**4
        s_terms = []
        rest = []
        for a in args:
            if s in a:
                s_terms.append(a)
            else: rest.append(a)
        new_args = rest + s_terms
        return sp.Mul(*new_args)

    idx = min([args.index(st) for st in s_terms if st in args])

    if not idx < len(args)-1:
        # s already is at the right
        return mul

    L = sp.Mul(*args[:idx]) # left term
    C = args[idx] # current term
    N = args[idx+1] # next term
    R0 = sp.Mul(*args[idx+2:]) # all right terms
    if not t in N:
        assert t in R0
        return L*N*right_shift(C*R0, s, t)

    exponent = C.as_base_exp()[1]
    N_new = apply_deriv(N, exponent, s, t)
    assert isinstance(N_new, sp.Add)

    res = 0
    for a in N_new.args:
        if not a.count_ops() == a.expand().count_ops():
            IPS()
        tmp = L*right_shift(a*R0, s, t, max_pow)
        res+= tmp.expand()

    return res


def right_shift_all(sum_exp, s, t, max_pow=4):
    """
    applies the right_shift to all arguments of a sum
    if sum only consists of one arg this is also accepted
    """

    if isinstance(sum_exp, sp.Matrix):
        def fnc(a):
            return right_shift_all(a, s, t, max_pow)
        return sum_exp.applyfunc(fnc)

    assert isinstance(sum_exp, sp.Basic)

    if isinstance(sum_exp, sp.Add):
        args = sum_exp.args
    elif isinstance(sum_exp, (sp.Mul, sp.Atom)):
        args = (sum_exp,)
    else:
        raise ValueError, "unexpected type: %s" %type(sum_exp)

    res = 0
    for a in args:
        assert isinstance(a, (sp.Mul, sp.Atom))
        res += right_shift(a, s, t, max_pow)


    return res

def make_all_symbols_commutative(expr, appendix='_c'):
    '''
    :param expr:
    :return: expr (with all symbols commutative) and
              a subs_tuple_list [(s1_c, s1_nc), ... ]
    '''

    symbs = st.atoms(expr, sp.Symbol)

    nc_symbols = [s for s in symbs if s.is_commutative == False]

    new_symbols = [sp.Symbol(s.name+appendix, commutative=True)
                   for s in symbs]

    tup_list = zip(new_symbols, nc_symbols)
    return expr.subs(zip(nc_symbols, new_symbols)), tup_list



def nc_mul(L, R):
    """
    This function performs matrix multiplication while respecting the multiplication
    order of noncommutative symbols

    :param L:
    :param R:
    :return:
    """

    if isinstance(L, sp.Expr) and isinstance(R, sp.Expr):
        return L*R
    elif isinstance(L, sp.Expr):
        assert isinstance(R, sp.Matrix)
        res = R.applyfunc(lambda x: L*x)
    elif isinstance(R, sp.Expr):
        assert isinstance(L, sp.Matrix)
        res = L.applyfunc(lambda x: x*R)
    elif isinstance(L, sp.Matrix) and isinstance(R, sp.Matrix):
        nrL, ncL = L.shape
        nrR, ncR = R.shape

        assert ncL == nrR

        res = sp.zeros(nrL, ncR)

        for i in xrange(nrL):  # iterate over the rows of L
            for j in xrange(ncR):  # iterate over the columns of R

                res_elt = 0
                # dot product of row and column
                for k in xrange(ncL):
                    res_elt += L[i, k] * R[k, j]

                res[i, j] = res_elt
    else:
        msg = "at least one invalid type: %s, %s" %(type(L), type(R))
        raise TypeError(msg)

    return res

def _method_mul(self, other):
    return nc_mul(other, self)

# TODO: Tests!!!



if __name__ == "__main__":
    p = sp.Function("p")(t)
    f = cos(p)

    w = s**2*f

    r = right_shift(w, s, t)

