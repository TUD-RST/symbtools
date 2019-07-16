"""
This module provides functions w.r.t. time_derivation of sympy-symbols.

This has several advantages compared to using time-dependent functions, e.g. the much shorter string representation.
"""

import sympy as sp
from .auxiliary import lzip, atoms, recursive_function, t

from ipydex import IPS


def time_deriv(expr, func_symbols, prov_deriv_symbols=None, t_symbol=None,
               order=1, **kwargs):
    """
    Example: expr = f(a, b). We know that a, b are time-functions: a(t), b(t)
    we want : expr.diff(t) with te appropriate substitutions made
    :param expr:                the expression to be differentiated
    :param func_symbols:        the symbols which are functions (e.g. of the time)
    :param prov_deriv_symbols:  a sequence of symbols which will be used for the
                                derivatives of the symbols
    :param t_symbol:            symbol for time (optional)
    :param order:               derivative order

    :return: derived expression


    Note: this process might be tricky because symbols with the same name
    but different sets of assumptions (real=True etc.) are handled as
    different symbols by sympy. Here we dont want this. If the name of
    func_symbols occurs in expr this is sufficient for being regarded as equal.

    for new created symbols the assumptions are copied from the parent symbol
    """

    if prov_deriv_symbols is None:
        prov_deriv_symbols = []

    if not t_symbol:
        # try to extract t_symbol from expression
        tmp = match_symbols_by_name(expr.atoms(sp.Symbol), 't', strict=False)
        if len(tmp) > 0:
            assert len(tmp) == 1
            t = tmp[0]
        else:
            t = sp.Symbol("t")
    else:
        t = t_symbol

    if isinstance(expr, (sp.MatrixSymbol, sp.MatAdd, sp.MatMul)):
        return matrix_time_deriv(expr, func_symbols, t_symbol,
                                 prov_deriv_symbols, order=order)

    func_symbols = list(func_symbols)  # convert to list

    # expr might contain derivative symbols -> add them to func_symbols
    deriv_symbols0 = [symb for symb in expr.atoms() if is_derivative_symbol(symb)]

    for ds in deriv_symbols0:
        if not ds in prov_deriv_symbols and not ds in func_symbols:
            func_symbols.append(ds)

    # replace the func_symbols by the symbols from expr to make sure the the
    # correct symbols (with correct assumptions) are used.
    expr_symbols = atoms(expr, sp.Symbol)
    func_symbols = match_symbols_by_name(expr_symbols, func_symbols, strict=False)

    # convert symbols to functions
    funcs = [ symb_to_time_func(s, arg=t) for s in func_symbols ]

    derivs1 = [[f.diff(t, ord) for f in funcs] for ord in range(order, 0, -1)]

    # TODO: current behavior is inconsistent:
    # time_deriv(x1, [x1], order=5) -> x_1_d5
    # time_deriv(x_2, [x_2], order=5) -> x__2_d5
    # (respective first underscore is obsolete)

    def extended_name_symb(base, ord, assumptions=None, base_difforder=None):
        """
        construct a derivative symbol with an appropriate name and other properties
        like assumptions and the attributes ddt_parent

        Because this function might be called recursively, the optional argument
        base_difforder is used to carry the difforder value of the original symbol
        """

        if assumptions is None:
            assumptions = {}

        if isinstance(base, sp.Symbol):
            assert base_difforder is None  # avoid conflicting information
            if hasattr(base, 'difforder'):
                base_difforder = base.difforder
            else:
                base_difforder = 0
            base = base.name

        assert isinstance(base, str)
        if base_difforder is None:
            base_difforder = 0

        # remove trailing number
        base_order = base.rstrip('1234567890')

        # store trailing number
        trailing_number = str(base[len(base_order):len(base)])

        new_name = []

        # check for 4th derivative
        if base_order[-6:len(base_order)]=='ddddot' and not new_name:
            variable_name = base_order[0:-6]
            underscore = r'' if trailing_number == r'' else r'_'
            new_name = variable_name + underscore + trailing_number + r'_d5'

        # check for 3rd derivative
        elif base_order[-5:len(base_order)]=='dddot':
            variable_name = base_order[0:-5]
            new_name = variable_name + r'ddddot' + trailing_number

        # check for 2nd derivative
        elif base_order[-4:len(base_order)]=='ddot' and not new_name:
            variable_name = base_order[0:-4]
            new_name = variable_name + r'dddot' + trailing_number

        # check for 1st derivative
        elif base_order[-3:len(base_order)]=='dot' and not new_name:
            variable_name = base_order[0:-3]
            new_name = variable_name + r'ddot' + trailing_number

        # check for higher order derivative:
        # x_d5 -> x_d6, etc.
        # x_3_d5 -> x_3_d6 etc.
        elif base_order[-2:len(base_order)]=='_d' and not new_name:
            new_order = int(trailing_number) + 1
            new_name = base_order + str(new_order)

        elif not new_name:
            new_name = base_order + r'dot' + trailing_number

        if ord == 1:
            new_symbol = sp.Symbol(new_name, **assumptions)
            new_symbol.difforder = base_difforder + ord

            return new_symbol
        else:
            return extended_name_symb(new_name, ord - 1, assumptions, base_difforder=base_difforder+1)

    # the user may want to provide their own symbols for the derivatives
    if not prov_deriv_symbols:
        deriv_symbols1 = []
        for ord in range(order, 0, -1):
            tmp_symbol_list = []

            for s in func_symbols:
                ens = extended_name_symb(s, ord, s.assumptions0)
                tmp_symbol_list.append(ens)

            deriv_symbols1.append(tmp_symbol_list)

    else:
        L = len(func_symbols)
        assert len(prov_deriv_symbols) == order*L

        # assume a structure like [xd, yd,  xdd, ydd] (for order = 2)
        # convert in a structure like in the case above
        deriv_symbols1 = []
        for ord in range(order, 0, -1):
            k = ord - 1
            part = prov_deriv_symbols[k*L:ord*L]
            assert len(part) == L

            deriv_symbols1.append(part)

    # flatten the lists:
    derivs = []
    for d_list in derivs1:
        derivs.extend(d_list)

    deriv_symbols = []
    for ds_list in deriv_symbols1:
        deriv_symbols.extend(ds_list)

    subs1 = lzip(func_symbols, funcs)

    # important: begin substitution with highest order
    subs2 = lzip(derivs + funcs, deriv_symbols + func_symbols)

    # ensure that derivs and deriv_symbols are sorted correctly
    subs2 = _sort_func_symb_tuples(subs2, index=0)

    _set_ddt_attributes(subs2)

    expr1 = expr.subs(subs1)
    expr2 = expr1.diff(t, order)
    expr3 = expr2.subs(subs2)

    return expr3


def _sort_func_symb_tuples(tuple_list, index):
    """
    Helper function for time_deriv: ensure that the tuples start with higest order and that there are no duplicates
    :param tuple_list:  list of
    :type index:        index of the sp.Derivative/sp.Function objects w.r.t the 2-tuples, i.e. 0 or 1
                        0 for [(a(t), a), ...]  and 1 for [(a, a(t)), ...]
    :return:
    """

    return sorted(set(tuple_list), key=lambda e: get_sp_deriv_order(e[index]), reverse=True)


def _set_ddt_attributes(rplmts_funcder_to_symb):
    """
    set the following attribute of symbs:
         .ddt_parent
         .ddt_func
         .ddt_child (of matching parent)

    (assuming that all needed symbols are provided and in descending order )

    :param rplmts_funcder_to_symb:
            sequence of tuples (deriv, symb)

    :return: None
    """

    # "funcder" means func or derivative
    funcder_symb_map = dict(rplmts_funcder_to_symb)

    # now use ascending order
    # use descending order
    for funcder, symbol in rplmts_funcder_to_symb:

        symbol.ddt_func = funcder

        if funcder.is_Derivative:
            # funcder.args looks like (x1(t), t, t, t)
            order = get_sp_deriv_order(funcder)
            if order == 1:
                parent_func_der = funcder.args[0]
            else:
                func = funcder.args[0]
                var = func.args[0]
                parent_func_der = sp.Derivative(func, var, order-1)
            try:
                parent_symbol = funcder_symb_map[parent_func_der]
            except KeyError:
                parent_symbol = symbol.ddt_parent
            assert parent_symbol is not None

            parent_symbol.ddt_func = parent_func_der
            parent_symbol.ddt_child = symbol

            symbol.ddt_parent = parent_symbol


def matrix_time_deriv(expr, func_symbols, t_symbol, prov_deriv_symbols=None,
                                                        order=1, **kwargs):
    """
    like time_deriv but for expressions containint MatrixSymbols
    """

    if prov_deriv_symbols is None:
        prov_deriv_symbols = []

    assert isinstance(expr, (sp.MatrixSymbol, sp.MatAdd, sp.MatMul))

    if order == 0:
        return expr

    def matdiff(A, symbol, order):
        assert isinstance(A, sp.MatrixSymbol)
        pseudo_symb = sp.Symbol(A.name)
        diff_symb = time_deriv(pseudo_symb, func_symbols, t_symbol=symbol, order=order)
        if diff_symb == 0:
            return A*0
        else:
            return sp.MatrixSymbol(diff_symb.name, *A.shape)

    # noinspection PyShadowingNames
    def matmuldiff(expr, symbol, order):
        if order > 1:
            # recursively reduce to order 1:
            tmp = matmuldiff(expr, symbol, order-1)

            # last deriv step
            return matrix_time_deriv(tmp, func_symbols, t_symbol,
                                     prov_deriv_symbols, order=1)

        args = expr.args
        res = 0*expr

        for i, a in enumerate(args):
            first_factors = args[:i]
            last_factors = args[i+1:]
            diff_factor = time_deriv(a, func_symbols, t_symbol=symbol)
            product_args = first_factors + (diff_factor,) + last_factors
            res = res + sp.MatMul(*product_args)

        return res

    # noinspection PyShadowingNames
    def matadddiff(expr, symbol, order):
        res = 0*expr

        for i, a in enumerate(expr.args):
            res = res + time_deriv(a, func_symbols, t_symbol=symbol, order=order)
        return res

    if isinstance(expr, sp.MatrixSymbol):
        return matdiff(expr, t_symbol, order)
    elif isinstance(expr, sp.MatAdd):
        return matadddiff(expr, t_symbol, order)
    elif isinstance(expr, sp.MatMul):
        return matmuldiff(expr, t_symbol, order)


def get_sp_deriv_order(deriv_object):

    if isinstance(deriv_object, sp.Function) and not isinstance(deriv_object, sp.Derivative):
        return 0

    assert isinstance(deriv_object, sp.Derivative)

    arg1 = deriv_object.args[1]

    if isinstance(arg1, (tuple, sp.Tuple)):
        # new interface is like Derivative(u1(t), (t, 2))
        if len(deriv_object.args) > 2:
            msg = "only multivariate derivatives are supported yet"
            raise NotImplementedError(msg)
        order = int(arg1[1])
    elif isinstance(arg1, sp.Symbol):
        # old interface was like Derivative(u1(t), t, t)

        order = len(deriv_object.args) - 1
    else:
        msg = "Unexpexted type for arg1 of Derivative: {}".format(type(arg1))
        raise ValueError(msg)

    assert isinstance(order, int)
    return order


def match_symbols_by_name(symbols1, symbols2, strict=True):
    """
    :param symbols1:
    :param symbols2: (might also be a string or a sequence of strings)
    :param strict: determines whether an error is caused if a symbol is not found
                   default: True
    :return: a list of symbols which are those objects from ´symbols1´ where
     the name occurs in ´symbols2´

     ordering is determined by ´symbols2´
    """

    if isinstance(symbols2, str):
        assert " " not in symbols2
        symbols2 = [symbols2]

    if isinstance(symbols1, (sp.Expr, sp.MatrixBase)):
        symbols1 = atoms(symbols1, sp.Symbol)

    str_list1 = [str(s.name) for s in symbols1]
    sdict1 = dict( lzip(str_list1, symbols1) )

    str_list2 = [str(s) for s in symbols2]
    # sympy expects str here (unicode not allowed)

    res = []

    for string2 in str_list2:
        res_symb = sdict1.get(string2)
        if res_symb:
            res.append(res_symb)
        elif strict:
            msg = "Could not find the symbol " + string2
            raise ValueError(msg)

    return res


def symb_to_time_func(symb, arg=None):
    """
    For given symbol x return x.ddt_func (if it exists) or create Function x(t).

    :param symb:    Symbol
    :param arg:     Optional symbol for time argument
    :return:
    """
    assert symb.is_Symbol
    if arg is None:
        arg = t
    assert arg.is_Symbol

    if symb.ddt_func.is_Function or symb.ddt_func.is_Derivative:
        return symb.ddt_func
    else:
        return sp.Function(symb.name)(arg)


def is_derivative_symbol(expr, t_symbol=None):
    """
    Returns whether expr is a derivative symbol (w.r.t. t)

    :param expr:
    :param t_symbol:
    :return: True or False
    """

    if t_symbol is not None:
        # we currently do not distinguish between different independent variables
        raise NotImplementedError

    if hasattr(expr, 'difforder') and expr.difforder > 0:
        return True
    else:
        return False


@recursive_function
def get_all_deriv_childs(thisfunc, expr):
    """
    for each symbol s in expr go down the s.ddt_child-tree and add them to the result

    :param thisfunc:
    :param expr:
    :return:
    """
    symbols = expr.atoms(sp.Symbol)

    res = []
    for s in symbols:
        if isinstance(s.ddt_child, sp.Symbol):
            res.append(s.ddt_child)
            res.extend(thisfunc(s.ddt_child))
        else:
            assert s.ddt_child is None

    return sp.Matrix(res)


@recursive_function
def get_all_deriv_parents(thisfunc, expr):
    """
    for each symbol s in expr go up the s.ddt_parent-tree and add them to the result

    :param thisfunc:
    :param expr:
    :return:
    """
    symbols = expr.atoms(sp.Symbol)

    res = []
    for s in symbols:
        if isinstance(s.ddt_parent, sp.Symbol):
            res.append(s.ddt_parent)
            res.extend(thisfunc(s.ddt_parent))
        else:
            assert s.ddt_parent is None

    return sp.Matrix(res)