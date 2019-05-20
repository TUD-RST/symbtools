
import casadi as cs
from casadi import SX

import os
from sympy.printing.lambdarepr import lambdarepr


def casidify(expr, state_vect, input_vect):
    # source: https://gist.github.com/cklb/60362e1f49ef65f5212fb5eb5904b3fd
    """
    Convert a sympy expression into a casadi expression
    """
    syms = []
    res = ["rhs = vertcat("]
    state_str = ["x = vertcat("]
    input_str = ["u = vertcat("]

    # extract symbols
    for _s in state_vect:
        syms.append("{0} = SX.sym('{0}')".format(str(_s)))
        state_str.append(str(_s) + ", ")

    for _s in input_vect:
        syms.append("{0} = SX.sym('{0}')".format(str(_s)))
        input_str.append(str(_s) + ", ")

    state_str.append(")")
    input_str.append(")")

    # convert expression
    for entry in expr:
        # handle expr
        _expr = lambdarepr(entry)
        res.append(_expr + ", ")

    res.append(")")

    ode_str = os.linesep.join(syms
                              + res
                              + state_str
                              + input_str)

    scope = dict(SX=cs.SX, MX=cs.MX, vertcat=cs.vertcat, sin=cs.sin, cos=cs.cos)
    exec(ode_str, scope)

    return scope["rhs"], scope["x"], scope["u"]


# convenience functions (maybe there is a more elegant way)

# noinspection PyPep8Naming
def seq_to_SX_matrix(seq):
    n = len(seq)

    # leading element:
    e0 = SX(seq[0])
    if e0.shape == (1, 1):
        # we have a sequence of scalars and create a column vector
        res = SX(n, 1)
        for i, elt in enumerate(seq):
            res[i, 0] = elt
        return res
    else:
        # we assume we have a sequence of vectors and want to concatenate them (colstack)
        n1, n2 = e0.shape
        res = SX(n1, n2*n)
        for i, elt in enumerate(seq):
            res[:, i] = elt
        return res


def unpack(sx_matrix):
    """
    convert SX matrix (vector) to list
    """
    n1, n2 = sx_matrix.shape
    assert n2 == 1
    res = [sx_matrix[i, 0] for i in range(n1)]
    return res
