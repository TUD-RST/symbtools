# -*- coding: utf-8 -*-

from functools import partial
import sympy as sp
import symbtools as st
from symbtools import lzip


"""
Collection of Code for noncommutative calculations ("s = d/dt")
"""


gC = st.Container()  # global Container
t = gC.t = sp.Symbol('t', commutative=False)  # time
s = gC.s = sp.Symbol('s', commutative=False)  # "Laplace-Variable" (d/dt)


def apply_deriv(term, power, s, t, func_symbols=[]):
    res = 0
    assert int(power) == power
    if power == 0:
        return term

    if isinstance(term, sp.Add):
        res = sum(apply_deriv(a, power, s, t, func_symbols) for a in term.args)

        return res

    tmp = st.time_deriv(term, func_symbols) + term*s
    res = apply_deriv(tmp, power-1, s, t, func_symbols).expand()

    return res


def right_shift(mul, s=None, t=None, func_symbols=[]):
    """
    mul:            the expression to be worked on
    s:              Laplace variable (optional)
    t:              time variable (optional)
    func_symbols:   sequence of time dependend symbols
                    (see `time_deriv`)
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
    if func_symbols is not None:
        assert not s in func_symbols

    # nothing to do
    if not mul.has(s):
        return mul

    if not mul.expand().count_ops() == mul.count_ops():
        msg = 'mul expression must be expanded:' + str(mul)
        raise ValueError(msg)

    if not isinstance(mul, sp.Mul):
        if mul == s:
            return mul
        elif isinstance(mul, sp.Pow) and mul.args[0] == s:
            return mul
        else:
            msg = 'Expected Mul, Symbol or Pow (like s**2), not ' + str(mul)
            raise ValueError(msg)
    assert isinstance(mul, sp.Mul)
    assert not s.is_commutative

    # find out which s-terms occur:
    linear_term = list(mul.atoms().intersection([s]))
    powers = [p for p in mul.atoms(sp.Pow) if p.args[0] == s ]
    s_terms = linear_term + powers

    # ensure that there are no furhter s-terms:
    atoms = list( mul.atoms(sp.Function, sp.Derivative) )
    if any([atom.has(s) for atom in atoms]):
        msg = "Unsupported or unexpected occurence of differential operator within function: %s"
        msg = msg % mul
        raise ValueError(msg)
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

    if isinstance(expr, sp.MatrixBase):
        def fnc(a):
            return right_shift_all(a, s, t, func_symbols)
        return expr.applyfunc(fnc)

    if s is None:
        s = gC.s
    if not expr.has(s):
        return expr

    assert isinstance(expr, sp.Basic)

    if isinstance(expr, sp.Add):
        args = expr.args
    elif isinstance(expr, (sp.Mul, sp.Atom)):
        args = (expr,)
    elif isinstance(expr, sp.Pow):
        # either s**a or some fraction
        base, expo = expr.args
        if not int(expo) == expo:
            msg = "unexpected exponent in expr %s" % (expr)
            raise ValueError(msg)
        if base == s or expo < 0:
            args = (expr,)
        else:
            msg = "unexpected expr (%s) of type: %s" % (expr, type(expr))
            ValueError(msg)

    else:
        msg = "unexpected type: %s" % type(expr)
        raise ValueError(msg)

    res = 0
    for a in args:
        if not isinstance(a, (sp.Mul, sp.Atom, sp.Pow, sp.Function)):
            msg = "unexpected arg: %s" %a
            raise ValueError(msg)
        if isinstance(a, sp.Function):
            if a.has(s):
                msg = "unexpected or unsupported occurence of differential operator"
                "inside function: %s" %a
                raise ValueError
        res += right_shift(a, s, t, func_symbols)

    return res


def make_all_symbols_commutative(expr, appendix='_c'):
    """
    :param expr:
    :return: expr (with all symbols commutative) and
              a subs_tuple_list [(s1_c, s1_nc), ... ]
    """

    if isinstance(expr, (list, tuple, set)):
        expr = sp.Matrix(list(expr))

    symbs = st.atoms(expr, sp.Symbol)
    nc_symbols = [s for s in symbs if not s.is_commutative]

    new_symbols = [sp.Symbol(s.name+appendix, commutative=True)
                   for s in nc_symbols]

    # preserve difforder attributes
    st.copy_custom_attributes(nc_symbols, new_symbols)
    tup_list = lzip(new_symbols, nc_symbols)
    return expr.subs(lzip(nc_symbols, new_symbols)), tup_list


def make_all_symbols_noncommutative(expr, appendix='_n'):
    """
    :param expr:
    :return: expr (with all symbols noncommutative) and
              a subs_tuple_list [(s1_n, s1_c), ... ]
    """

    if isinstance(expr, (list, tuple, set)):
        expr = sp.Matrix(list(expr))

    symbs = st.atoms(expr, sp.Symbol)
    c_symbols = [s for s in symbs if s.is_commutative]

    new_symbols = [sp.Symbol(s.name+appendix, commutative=False)
                   for s in c_symbols]

    # preserve difforder attributes
    st.copy_custom_attributes(c_symbols, new_symbols)
    tup_list = lzip(new_symbols, c_symbols)
    return expr.subs(lzip(c_symbols, new_symbols)), tup_list


def commutative_simplification(expr, exclude, max_deg=20):
    """
    :param expr:    expression or matrix
    :return:        simplified expression

    This function assumes that all symbols commute with each other except of exclude,
    which is required to occur only polynomial (with maximum degree <= max_deg)
    and placed at the right. Than simplifications like x*y*s - y*x*s = 0 are performed.
    """

    if isinstance(expr, sp.MatrixBase):
        def fnc(element):
            return commutative_simplification(element, exclude)
        return expr.applyfunc(fnc)

    assert isinstance(expr, sp.Basic)
    if not isinstance(exclude, sp.Symbol):
        msg = "Argument `exclude` must be a single symbol. Got %s" % type(exclude)
        raise ValueError(msg)

    expr = expr.expand()

    from IPython import embed as IPS

    # check all symbols are noncommutative
    all_symbols = list(expr.atoms(sp.Symbol))
    com_symbs = [symb for symb in all_symbols if symb.is_commutative]
    if len(com_symbs) > 0:
        msg = "All symbols in `expr` are expected to be non_commutative. "\
              "The following are not: %s" % str(com_symbs)
        raise ValueError(msg)

    functions = expr.atoms(sp.function.AppliedUndef)
    if not len(functions) == 0:
        msg = "Undefined functions currently not supported."
        raise NotImplementedError(msg)
    # Funktionen ausschließen

    include = all_symbols*1
    if exclude in all_symbols:
       include.remove(exclude)

    # check `exclude` is at the right border
    shift_res = right_shift_all(expr, s=exclude, func_symbols=include)
    if not shift_res == expr:
        msg = "Excluded var must be placed at the right. This is not the case."
        raise ValueError(msg)

    deg = nc_degree(expr, exclude, max_deg)

    coeffs = nc_coeffs(expr, exclude, deg)

    simplified_coeffs = []



    for c in coeffs:
        c_commutative, replacements = make_all_symbols_commutative(c)
        # simplify and afterwards substitute the original symbols back
        simplified_coeffs.append(c_commutative.simplify().subs(replacements))

    res = 0
    for i, c in enumerate(simplified_coeffs):
        res += c*s**i

    return res





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
    # TODO: use nc_degree (after performance-testing)
    # workarround: pass the maximum expected degree as kwarg

    if poly == 0:
        return [sp.sympify(0)]*(max_deg + 1)

    D0 = sp.Dummy('D0')
    poly = sp.sympify(poly).expand() + D0  # ensure class add

    if not isinstance(poly, sp.Add):
        msg = "Expected sp.Add. Got %s" % type(poly)
        raise TypeError(msg)

    res = []
    # special case: 0-th power of var
    coeff = 0
    for a in poly.args:
        if not a.has(var):
            coeff += a
    res.append(coeff.subs(D0, 0))

    # special case: 1st power of var
    if max_deg >= 1:
        coeff = poly.diff(var).subs(var, 0)
        res.append(coeff)

    # powers > 1:
    for i in range(2, max_deg+1):
        coeff = 0
        for a in poly.args:
            if a.has(var**(i)):
                term = a.subs(var, 1)
                coeff += term
        res.append(coeff)

    if order == "decreasing":
        res.reverse()

    return res


# TODO: Test how much max_degree affects the performance for large expressions
def nc_degree(expr, var, max_deg=20):

    if not expr.has(var):
        return 0

    res = [-1]  # we dont know about the 0th order term (and it does not matter)
    for i in range(0, max_deg):
        if expr.has(var**(i + 1)):
            res.append(1)
        else:
            res.append(0)

    assert 1 in res
    # for expr = x - a*x**3 res looks like [-1, 1, 0, 1, 0, 0, 0, ...]
    # we are interested in the highest index of a 1-entry (first index of reversed list)
    res.reverse()
    deg = len(res) - 1 - res.index(1)

    return deg


def nc_mul(L, R):
    """
    Sympy >=1.0 has fixed multiplication of noncommutative symbols
    with matrices, so nc_mul(L, R) is not necessary anymore.

    This function performs matrix multiplication while respecting the multiplication
    order of noncommutative symbols

    :param L:
    :param R:
    :return:
    """

    if st.is_scalar(L) and st.is_scalar(R):
        return L*R
    elif st.is_scalar(L):
        assert isinstance(R, sp.MatrixBase)
        res = R.applyfunc(lambda x: L*x)
    elif st.is_scalar(R):
        assert isinstance(L, sp.MatrixBase)
        res = L.applyfunc(lambda x: x*R)
    elif isinstance(L, sp.MatrixBase) and isinstance(R, sp.MatrixBase):
        nrL, ncL = L.shape
        nrR, ncR = R.shape

        assert ncL == nrR

        res = sp.zeros(nrL, ncR)

        for i in range(nrL):  # iterate over the rows of L
            for j in range(ncR):  # iterate over the columns of R

                res_elt = 0
                # dot product of row and column
                for k in range(ncL):
                    res_elt += L[i, k] * R[k, j]

                res[i, j] = res_elt
    else:
        msg = "at least one invalid type: %s, %s" %(type(L), type(R))
        raise TypeError(msg)

    return res


def unimod_inv(M, s=None, t=None, time_dep_symbs=[], simplify_nsm=True, max_deg=None):
    """ Assumes that M(s) is an unimodular polynomial matrix and calculates its inverse
    which is again unimodular

    :param M:               Matrix to be inverted
    :param s:               Derivative Symbol
    :param time_dep_symbs:  sequence of time dependent symbols
    :param max_deg:       maximum polynomial degree w.r.t. s of the ansatz

    :return: Minv
    """

    assert isinstance(M, sp.MatrixBase)
    assert M.is_square

    n = M.shape[0]

    degree_m = nc_degree(M, s)

    if max_deg is None:
        # upper bound according to
        # Levine 2011, On necessary and sufficient conditions for differential flatness, p. 73

        max_deg = (n - 1)*degree_m

    assert int(max_deg) == max_deg
    assert max_deg >= 0

    C = M*0
    free_params = []

    for i in range(max_deg+1):
        prefix = 'c{0}_'.format(i)
        c_part = st.symbMatrix(n, n, prefix, commutative=False)
        C += nc_mul(c_part, s**i)
        free_params.extend(list(c_part))

    P = nc_mul(C, M) - sp.eye(n)

    P2 = right_shift_all(P, s, t, time_dep_symbs).reshape(n*n, 1)

    deg_P = nc_degree(P2, s)

    part_eqns = []
    for i in range(deg_P + 1):
        # omit the highest order (because it behaves like in the commutative case)
        res = P2.diff(s, i).subs(s, 0)#/sp.factorial(i)
        part_eqns.append(res)

    eqns = st.row_stack(*part_eqns)  # equations for all degrees of s

    # now non-commutativity is inferring
    eqns2, st_c_nc = make_all_symbols_commutative(eqns)
    free_params_c, st_c_nc_free_params = make_all_symbols_commutative(free_params)

    # find out which of the equations are (in)homogeneous
    eqns2_0 = eqns2.subs(st.zip0(free_params_c))
    assert eqns2_0.atoms() in ({0, -1}, {-1}, set())
    inhom_idcs = st.np.where(st.to_np(eqns2_0) != 0)[0]
    hom_idcs = st.np.where(st.to_np(eqns2_0) == 0)[0]

    eqns_hom = sp.Matrix(st.np.array(eqns2)[hom_idcs])
    eqns_inh = sp.Matrix(st.np.array(eqns2)[inhom_idcs])

    assert len(eqns_inh) == n

    # find a solution for the homogeneous equations
    # if this is not possible, M was not unimodular
    Jh = eqns_hom.jacobian(free_params_c).expand()

    nsm = st.nullspaceMatrix(Jh, simplify=simplify_nsm, sort_rows=True)

    na = nsm.shape[1]
    if na < n:
        msg = 'Could not determine sufficiently large nullspace. '\
        'Either M is not unimodular or the expressions are to complicated.'
        # TODO: decide which of the two cases occurs, via substitution of
        # random numbers and singular value decomposition
        # (or application of st.generic_rank)
        raise ValueError(msg)

    # parameterize the inhomogenous equations with the solution of the homogeneous equations
    # new free parameters:
    aa = st.symb_vector('_a1:{0}'.format(na+1))
    nsm_a = nsm*aa

    eqns_inh2 = eqns_inh.subs(lzip(free_params_c, nsm_a))

    # now solve the remaining equations

    # solve the linear system
    Jinh = eqns_inh2.jacobian(aa)
    rhs_inh = -eqns_inh2.subs(st.zip0(aa))
    assert rhs_inh == sp.ones(n, 1)
    
    sol_vect = Jinh.solve(rhs_inh)
    sol = lzip(aa, sol_vect)

    # get the values for all free_params (now they are not free anymore)
    free_params_sol_c = nsm_a.subs(sol)

    # replace the commutative symbols with the original non_commutative symbols (of M)
    free_params_sol = free_params_sol_c.subs(st_c_nc)

    Minv = C.subs(lzip(free_params, free_params_sol))

    return Minv


def _method_mul(self, other):
    return nc_mul(other, self)


