# -*- coding: utf-8 -*-

from functools import partial
import sympy as sp
import symb_tools as st


from ipHelp import IPS, Tracer, ip_syshook, sys

#ip_syshook(1)


"""
Collection of Code for noncommutative calculations ("s = d/dt")
"""


gC = st.Container()  # global Container
t = gC.t = sp.Symbol('t', commutative=False)  # time
s = gC.s = sp.Symbol('s', commutative=False)  # Laplace-Variable (d/dt)


def apply_deriv(term, power, s, t, func_symbols=[]):
    res = 0
    assert int(power) == power
    if power == 0:
        return term

    if isinstance(term, sp.Add):
        res = sum(apply_deriv(a, power, s, t, func_symbols) for a in term.args)

        return res

    tmp = st.perform_time_derivative(term, func_symbols) + term*s
    res = apply_deriv(tmp, power-1, s, t, func_symbols).expand()

    return res


def right_shift(mul, s=None, t=None, func_symbols=[]):
    """
    mul:            the expression to be worked on
    s:              Laplace variable (optional)
    t:              time variable (optional)
    func_symbols:   sequence of time dependend symbols
                    (see `perform_time_derivative`)
    Vorgehen:
        index des ersten Auftauchens einer s-Potenz finden
        -> mul = L *s**p * X* R0
        -> s**p auf X anwenden -> es entstehen p + 1 Rest-Terme
        -> Auf diese wird die Funktion rekursiv angewendet
        -> Rückgabewert ist Mul oder Add von Muls wo die s-Terme ganz rechts stehen
    """

    if s is None:
        s = gC.s
    if t is None:
        t = gC.t

    assert isinstance(s, sp.Symbol)

    # nothing to do
    if not mul.has(s):
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
    assert not s.is_commutative

    # find out which s-terms occur:
    linear_term = list(mul.atoms().intersection([s]))
    powers = [p for p in mul.atoms(sp.Pow) if p.args[0] == s ]
    s_terms = linear_term + powers
    assert len(s_terms) > 0

    args = mul.args

    depends_on_time = partial(st.depends_on_t, t=t, dependent_symbols=func_symbols)


    idx = min([args.index(sterm) for sterm in s_terms if sterm in args])

    if not idx < len(args) - 1:
        # s already is at the right
        return mul

    L = sp.Mul(*args[:idx])  # left term
    C = args[idx]  # current term
    N = args[idx+1]  # next term
    R0 = sp.Mul(*args[idx+2:])  # all right terms

    if not depends_on_time(N*R0):
        # aggregate all s-Terms at the right margin
        # handle: f(t)*s*a*s**2*b*s*c -> f(t)*a*b*c*s**4
        s_terms = []
        rest = []
        for a in args:
            if a.has(s):
                s_terms.append(a)
            else:
                rest.append(a)
        new_args = rest + s_terms
        return sp.Mul(*new_args)

    if not depends_on_time(N):
        assert depends_on_time(R0)
        return L*N*right_shift(C*R0, s, t, func_symbols)

    exponent = C.as_base_exp()[1]
    N_new = apply_deriv(N, exponent, s, t, func_symbols)
    assert isinstance(N_new, sp.Add)

    res = 0
    for a in N_new.args:
        assert a.count_ops() == a.expand().count_ops()
        tmp = L*right_shift(a*R0, s, t, func_symbols)
        res += tmp.expand()

    return res


def right_shift_all(expr, s=None, t=None, func_symbols=[]):
    """
    applies the right_shift to all arguments of a sum (`expr`)
    if expr only consists of one arg this is also accepted

    :func_symbols: sequence of implicitly dependent symbols
    """

    expr = expr.expand()

    if isinstance(expr, sp.Matrix):
        def fnc(a):
            return right_shift_all(a, s, t, func_symbols)
        return expr.applyfunc(fnc)

    assert isinstance(expr, sp.Basic)

    if isinstance(expr, sp.Add):
        args = expr.args
    elif isinstance(expr, (sp.Mul, sp.Atom)):
        args = (expr,)
    elif isinstance(expr, sp.Pow):
        base, expo = expr.args
        assert int(expo) == expo
        assert expo < 0
        args = (expr,)

    else:
        raise ValueError, "unexpected type: %s" % type(expr)

    res = 0
    for a in args:
        assert isinstance(a, (sp.Mul, sp.Atom, sp.Pow))
        res += right_shift(a, s, t, func_symbols)

    return res


def make_all_symbols_commutative(expr, appendix='_c'):
    """
    :param expr:
    :return: expr (with all symbols commutative) and
              a subs_tuple_list [(s1_c, s1_nc), ... ]
    """

    symbs = st.atoms(expr, sp.Symbol)
    nc_symbols = [s for s in symbs if not s.is_commutative]

    new_symbols = [sp.Symbol(s.name+appendix, commutative=True)
                   for s in nc_symbols]

    tup_list = zip(new_symbols, nc_symbols)
    return expr.subs(zip(nc_symbols, new_symbols)), tup_list


def nc_coeffs(poly, var, max_deg=10, order='increasing'):
    """Returns a list of the coeffs w.r.t. var (expecting a monovariate polynomial)

    sympy.Poly can not handle non-commutative variables.
    This function mitigates that problem.

    :param poly:
    :param var:
    :param max_deg:     maximum degree to look for
    :param order:       increasing / decreasing

    :return: list of coeffs

    Caveat: This function expects all factors of `var` to be in one place.
    Terms like a*x*b*x**2 will not be handled correctly.
    """

    # TODO: elegant way to find out the degree
    # workarround: pass the maximum expected degree as kwarg

    D0 = sp.Dummy('D0')
    poly = poly.expand() + D0  # ensure class add

    assert isinstance(poly, sp.Add)
    res = []
    # special case: 0-th power of var
    coeff = 0
    for a in poly.args:
        if not a.has(var):
            coeff += a
    res.append(coeff.subs(D0, 0))

    # special case: first power of var
    coeff = poly.diff(var).subs(var, 0)
    res.append(coeff)

    # powers > 1:
    for i in xrange(1, max_deg):
        coeff = 0
        for a in poly.args:
            if a.has(var**(i + 1)):
                term = a.subs(var, 1)
                coeff += term
        res.append(coeff)

    if order == "decreasing":
        res.reverse()

    return res



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


