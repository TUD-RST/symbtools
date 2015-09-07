# -*- coding: utf-8 -*-
"""
functions to generate/analyse a model based on Lagrange formalism 2nd kind
with minimal coordinates:

Authors: Thomas Mutzke, Carsten Knoll, 2013
"""

import sympy as sp
import symb_tools as st


def Rz(phi):
    """
    Rotation Matrix in the xy plane
    """
    c = sp.cos(phi)
    s = sp.sin(phi)
    return sp.Matrix([[c, -s], [s, c]])

# 2d coordinate unitvectors
ex = sp.Matrix([1,0])
ey = sp.Matrix([0,1])


# helper function for velocity generation

def point_velocity(point, coord_symbs, velo_symbs, t):
    coord_funcs = []
    for c in coord_symbs:
        coord_funcs.append(sp.Function(c.name + "f")(t))

    coord_funcs = sp.Matrix(coord_funcs)
    p1 = point.subs(zip(coord_symbs, coord_funcs))

    v1_f = p1.diff(t)

    backsubs = zip(coord_funcs, coord_symbs) + zip(coord_funcs.diff(t),
                                                   velo_symbs)

    return v1_f.subs(backsubs)


def symbColVector(n, s='a'):
    """
    returns a column vector with n symbols s, index 1..n
    >>> symbColVector(3)
    [a1]
    [a2]
    [a3]
    >>> symbColVector(2,'k')
    [k1]
    [k2]
    >>> symbRowVector(-1)
    Traceback (most recent call last):
        ...
    ValueError: Positive integer required
    >>> symbColVector(1,3)
    Traceback (most recent call last):
        ...
    TypeError: unsupported operand type(s) for +: 'int' and 'str'
    """
    if n <= 0 or int(n) != n:
        raise ValueError, "Positive integer required"

    A = sp.Matrix(n, 1, lambda i, j: sp.Symbol(s + '%i' % (i + 1)))
    return A


class SymbolicModel(object):
    """ model class """

    def __init__(self):  # , eq_list, var_list, extforce_list, disforce_list):
        self.eq_list = None  #eq_list
        self.qs = None  #var_list
        self.extforce_list = None  #extforce_list
        self.disforce_list = None  #disforce_list

        # for the classical state representation
        self.f = None
        self.g = None
        self.tau = None

        # for the collocated partial linearization
        self.ff = None
        self.gg = None
        self.aa = None

        self.solved_eq = None
        self.zero_equilibrium = None
        self.state_eq = None
        self.A = None
        self.b = None
        self.x = None
        self.xd = None
        self.h = None
        self.M = None

    def substitute_ext_forces(self, old_F, new_F, new_F_symbols):
        """
        Backround: sometimes modeling is easier with forces which are not
        the real control inputs. Then it is necessary to introduce the real
        forces later.
        """

        # nothing fancy has be done yet with the model
        assert self.eq_list != None
        assert (self.f, self.solved_eq, self.state_eq) == (None,) * 3

        subslist = zip(old_F, new_F)
        self.eq_list = self.eq_list.subs(subslist)
        self.extforce_list = new_F_symbols

    def calc_mass_matrix(self):
        """
        calculate and return the mass matrix (without simplification)
        """

        if isinstance(self.M, sp.Matrix):
            assert self.M.is_square
            return self.M
        # Übergangsweise:
        if hasattr(self, 'ttdd'):
            self.M = self.eq_list.jacobian(self.ttdd)
        else:
            self.M =  self.eq_list.jacobian(self.qdds)

        return self.M

    MM = property(calc_mass_matrix)  # short hand

    def solve_for_acc(self, simplify=False):
        self.calc_mass_matrix()
        if simplify:
            self.M.simplify()
        M = self.M

        rhs = self.eq_list.subs(st.zip0(self.ttdd)) * -(1)
        d = M.berkowitz_det()
        adj = M.adjugate()
        if simplify:
            d = d.simplify()
            adj.simplify()
        Minv = adj/d
        res = Minv*rhs

        return res

    def calc_state_eq(self, **kwargs):
        """
        reformulate the second order model to a first order statespace model
        xd = f(x)+g(x)*u
        """
        simplify = kwargs.get('simplify', False)

        self.x = st.row_stack(self.tt, self.ttd)

        eq2nd_order = self.solve_for_acc(**kwargs)
        self.state_eq = st.row_stack(self.ttd, eq2nd_order)

        self.f = self.state_eq.subs(st.zip0(self.tau))
        self.g = self.state_eq.jacobian(self.tau)

        if simplify:
            self.f.simplify()
            self.g.simplify()

    def calc_coll_part_lin_state_eq(self, **kwargs):
        """
        calc vectorfields ff, and gg of collocated linearization
        """
        simplify = kwargs.get('simplify', False)
        self.x = st.row_stack(self.tt, self.ttd)
        nq = len(self.tau)
        np = len(self.tt) - nq
        B = self.eq_list.jacobian(self.tau)
        cond1 = B[:np, :] == sp.zeros(np, nq)
        cond2 = B[np:, :] == -sp.eye(nq)
        if not cond1 and cond2:
            msg = "The jacobian of the equations of motion do not have the expected structure: %s"
            raise NotImplementedError(msg % str(B))

        # set the actuated accelearations as new inputs
        self.aa = self.ttdd[-nq:, :]

        self.calc_mass_matrix()
        if simplify:
            self.M.simplify()
        M11 = self.M[:np, :np]
        M12 = self.M[:np, np:]

        d = M11.berkowitz_det()
        adj = M11.adjugate()
        if simplify:
            d = d.simplify()
            adj.simplify()
        M11inv = adj/d

        # setting input and acceleration to 0
        C1K1 = self.eq_list[:np, :].subs(st.zip0(self.ttdd, self.tau))
        #eq_passive = -M11inv*C1K1 - M11inv*M12*self.aa

        self.ff = st.row_stack(self.ttd, -M11inv*C1K1, self.aa*0)
        self.gg = st.row_stack(sp.zeros(np + nq, nq), -M11inv*M12, sp.eye(nq))

        if simplify:
            self.ff.simplify()
            self.gg.simplify()

"""
Hinweis: 2014-10-15: Verhalten wurde geändert.
 Die Gleichungen werden jetzt in den originalen Zeit-Funktionen und ihren
 Ableitungen zurück gegeben
"""


def generate_model(T, U, qq, F, **kwargs):
    raise DeprecationWarning('generate_symbolic_model should be used')
    """
    T kinetic co-energy
    U potential energy
    q independend deflection variables
    F external forces
    D dissipation function
    """

    n = len(qq)

    # time variable
    t = sp.symbols('t')

    # symbolic Variables (to prevent Functions where we not want them)
    qs = []
    qds = []
    qdds = []

    if not kwargs:
        # assumptions for the symbols (facilitating the postprocessing)
        kwargs ={"real": True}

    for qi in qq:
        # ensure that the same Symbol for t is used
        assert qi.is_Function and qi.args == (t,)
        s = str(qi.func)

        qs.append(sp.Symbol(s, **kwargs))
        qds.append(sp.Symbol(s + "_d",  **kwargs))
        qdds.append(sp.Symbol(s + "_dd",  **kwargs))

    qs, qds, qdds = sp.Matrix(qs), sp.Matrix(qds), sp.Matrix(qdds)

    #derivative of configuration variables
    qd = qq.diff(t)
    qdd = qd.diff(t)

    #lagrange function
    L = T - U

    #substitute functions with symbols for partial derivatives

    #highest derivative first
    subslist = zip(qdd, qdds) + zip(qd, qds) + zip(qq, qs)
    L = L.subs(subslist)

    # partial derivatives of L
    Ldq = st.jac(L, qs)
    Ldqd = st.jac(L, qds)

    # generalised external force
    f = sp.Matrix(F)

    # substitute symbols with functions for time derivative
    subslistrev = st.rev_tuple(subslist)
    Ldq = Ldq.subs(subslistrev)
    Ldqd = Ldqd.subs(subslistrev)
    Ldqd = Ldqd.diff(t)

    #lagrange equation 2nd kind
    model_eq = qq * 0
    for i in range(n):
        model_eq[i] = Ldqd[i] - Ldq[i] - f[i]

    # model equations with symbols
    model_eq = model_eq.subs(subslist)

    # create object of model
    model1 = SymbolicModel()  # model_eq, qs, f, D)
    model1.eq_list = model_eq
    model1.qs = qs
    model1.extforce_list = f
    model1.tau = f

    model1.qds = qds
    model1.qdds = qdds


    # also store kinetic and potential energy
    model1.T = T
    model1.U = U

    # analyse the model

    return model1


def generate_symbolic_model(T, U, tt, F, **kwargs):
    """
    T kinetic energy
    U potential energy
    tt sequence of independent deflection variables ("theta")
    F external forces

    kwargs: might be something like 'real=True'
    """

    # if not kwargs:
    #     # assumptions for the symbols (facilitating the postprocessing)
    #     kwargs ={"real": True}
    #
    n = len(tt)

    for theta_i in tt:
        assert isinstance(theta_i, sp.Symbol)

    F = sp.Matrix(F)
    assert F.shape == (n, 1)

    # introducing symbols for the derivatives
    tt = sp.Matrix(tt)
    ttd = st.perform_time_derivative(tt, tt, **kwargs)
    ttdd = st.perform_time_derivative(tt, tt, order=2, **kwargs)

    #Lagrange function
    L = T - U

    if not T.atoms().intersection(ttd) == set(ttd):
        raise ValueError('Not all velocity symbols do occur in T')

    # partial derivatives of L
    L_diff_tt = st.jac(L, tt)
    L_diff_ttd = st.jac(L, ttd)

    #prov_deriv_symbols = [ttd, ttdd]

    # time-depended_symbols
    tds = list(tt) + list(ttd)
    L_diff_ttd_dt = st.perform_time_derivative(L_diff_ttd, tds, **kwargs)

    #lagrange equation 2nd kind
    model_eq = sp.zeros(n, 1)
    for i in range(n):
        model_eq[i] = L_diff_ttd_dt[i] - L_diff_tt[i] - F[i]

    # create object of model
    mod = SymbolicModel()  # model_eq, qs, f, D)
    mod.eq_list = model_eq

    mod.extforce_list = F
    reduced_F = sp.Matrix([s for s in F if st.is_symbol(s)])
    mod.F = reduced_F
    mod.tau = reduced_F

    # coordinates velocities and accelerations
    mod.tt = tt
    mod.ttd = ttd
    mod.ttdd = ttdd

    mod.qs = tt
    mod.qds = ttd
    mod.qdds = ttdd

    # also store kinetic and potential energy
    mod.T = T
    mod.U = U

    return mod


def solve_eq(model):
    """
    solve model equations for accelerations
    xdd = f(x,xd)+g(x)*u
    """
    eq2_list = model.qdds * 0
    for i in range(len(model.qdds)):
        eq2_list[i] = sp.solve([model.eq_list[i]], model.qdds[i])
        eq2_list[i] = sp.Eq(model.qdds[i], (eq2_list[i][model.qdds[i]]))
    model.solved_eq = eq2_list


def is_zero_equilibrium(model):
    """ checks model for equilibrium point zero """
    # substitution of state variables and their derivatives to zero
    #substitution of input to zero --> stand-alone system
    subslist = st.zip0(model.qs) + st.zip0(model.qds) + st.zip0(
        model.qdds) + st.zip0(model.extforce_list)
    eq_1 = model.eq_list.subs(subslist)
    model.zero_equilibrium = all(eq_1)






def linearise(model):
    """ linearise model at equilibrium point zero """
    if model.zero_equilibrium:
        J = model.f.jacobian(model.x)
        # substitute state varibles with equilibrium point zero
        model.A = J.subs(st.zip0(model.x))
        model.b = model.g.subs(st.zip0(model.x))
