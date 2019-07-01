# -*- coding: utf-8 -*-
"""
functions to generate/analyse a model based on Lagrange formalism 2nd kind
with minimal coordinates:

Authors: Thomas Mutzke (2013), Carsten Knoll (2013-2015)
"""

import sympy as sp
import numpy as np
from scipy.optimize import fmin
import symbtools as st
from symbtools import lzip
from IPython import embed as IPS


def Rz(phi, to_numpy=False):
    """
    Rotation Matrix in the xy plane
    """
    c = sp.cos(phi)
    s = sp.sin(phi)
    M = sp.Matrix([[c, -s], [s, c]])
    if to_numpy:
        return st.to_np(M)
    else:
        return M


# 2d coordinate unitvectors
ex = sp.Matrix([1, 0])
ey = sp.Matrix([0, 1])


# helper function for velocity generation

def point_velocity(point, coord_symbs, velo_symbs, t):
    coord_funcs = []
    for c in coord_symbs:
        coord_funcs.append(sp.Function(c.name + "f")(t))

    coord_funcs = sp.Matrix(coord_funcs)
    p1 = point.subs(lzip(coord_symbs, coord_funcs))

    v1_f = p1.diff(t)

    backsubs = lzip(coord_funcs, coord_symbs) + lzip(coord_funcs.diff(t),
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
        raise ValueError("Positive integer required")

    A = sp.Matrix(n, 1, lambda i, j: sp.Symbol(s + '%i' % (i + 1)))
    return A


def new_model_from_equations_of_motion(eqns, theta, tau):
    """

    :param eqns:    vector of equations of motion
    :param tt:      generalized coordinates
    :param tau:     input
    :return:        SymbolicModel instance
    """

    mod = SymbolicModel()
    mod.eqns = eqns
    mod.tt = theta
    mod.ttd = st.time_deriv(theta, theta)
    mod.ttdd = st.time_deriv(theta, theta, order=2)
    mod.extforce_list = tau
    mod.tau = tau

    return mod


# noinspection PyPep8Naming
class SymbolicModel(object):
    """ model class """

    def __init__(self):
        self.eqns = None  # eq_list
        self.qs = None  # var_list
        self.extforce_list = None  # extforce_list
        self.disforce_list = None  # disforce_list
        self.constraints = None

        # for the classical state representation
        self.f = None
        self.g = None
        self.tau = None

        self.x = None  # deprecated

        # for the collocated partial linearization
        self.xx = None
        self.ff = None
        self.gg = None
        self.aa = None

        # coordinates (and lagrange multipliers)
        self.tt = None
        self.ttd = None
        self.ttdd = None
        self.llmd = None

        self.dae = None  # might become a Container

        # for Lagrange-Byrnes-Normalform
        self.zz = None
        self.fz = None
        self.gz = None
        self.ww = None
        self.ww_def = None

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
        Background: sometimes modeling is easier with forces which are not
        the real control inputs. Then it is necessary to introduce the real
        forces later.
        """

        # nothing fancy has be done yet with the model
        assert self.eqns is not None
        assert (self.f, self.solved_eq, self.state_eq) == (None,) * 3

        subslist = lzip(old_F, new_F)
        self.eqns = self.eqns.subs(subslist)
        self.extforce_list = sp.Matrix(new_F_symbols)
        self.tau = sp.Matrix(new_F_symbols)

    def calc_mass_matrix(self):
        """
        calculate and return the mass matrix (without simplification)
        """

        if isinstance(self.M, sp.Matrix):
            assert self.M.is_square
            return self.M
        # Übergangsweise:
        if hasattr(self, 'ttdd'):
            self.M = self.eqns.jacobian(self.ttdd)
        else:
            self.M = self.eqns.jacobian(self.qdds)

        return self.M

    MM = property(calc_mass_matrix)  # short hand

    def solve_for_acc(self, simplify=False):

        self.calc_mass_matrix()
        if simplify:
            self.M.simplify()
        M = self.M

        rhs = self.eqns.subs(st.zip0(self.ttdd)) * -1
        d = M.berkowitz_det()
        adj = M.adjugate()
        if simplify:
            d = d.simplify()
            adj.simplify()
        Minv = adj/d
        res = Minv*rhs

        return res

    # TODO add remark stating that this methods assumes self.eqns to be
    # affine regarding self.tau
    def calc_state_eq(self, simplify=True):
        """
        reformulate the second order model to a first order statespace model
        xd = f(x)+g(x)*u
        """

        self.xx = st.row_stack(self.tt, self.ttd)
        self.x = self.xx  # xx is preferred now

        eq2nd_order = self.solve_for_acc(simplify=simplify)
        self.state_eq = st.row_stack(self.ttd, eq2nd_order)

        self.f = self.state_eq.subs(st.zip0(self.tau))
        self.g = self.state_eq.jacobian(self.tau)

        if simplify:
            self.f.simplify()
            self.g.simplify()

    def calc_dae_eq(self, parameter_values=None, **kwargs):
        """
        In case of present constraints ode representation is not possible.
        This method constructs a representation F(y, ydot) = 0.

        Such a form can be passed to a DAE solver like IDA (from SUNDIALS / Assimulo)

        :return: dae (Container); also set self.dae = dae , self.dae.yy, self.dae.yyd, self.dae.eqns, ...
        """

        self.dae = DAE_System(self, parameter_values=parameter_values, **kwargs)

        return self.dae

    def calc_coll_part_lin_state_eq(self, simplify=True):
        """
        calc vector fields ff, and gg of collocated linearization.
        self.ff and self.gg are set.
        """
        self.xx = st.row_stack(self.tt, self.ttd)
        self.x = self.xx  # xx is preferred now

        nq = len(self.tau)
        np = len(self.tt) - nq
        B = self.eqns.jacobian(self.tau)
        cond1 = B[:np, :] == sp.zeros(np, nq)
        cond2 = B[np:, :] == -sp.eye(nq)
        if not cond1 and cond2:
            msg = "The jacobian of the equations of motion does not have the expected structure: %s"
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
        C1K1 = self.eqns[:np, :].subs(st.zip0(self.ttdd, self.tau))
        #eq_passive = -M11inv*C1K1 - M11inv*M12*self.aa

        self.ff = st.row_stack(self.ttd, -M11inv*C1K1, self.aa*0)
        self.gg = st.row_stack(sp.zeros(np + nq, nq), -M11inv*M12, sp.eye(nq))

        if simplify:
            self.ff.simplify()
            self.gg.simplify()

    def calc_lbi_nf_state_eq(self, simplify=False):
        """
        calc vectorfields fz, and gz of the Lagrange-Byrnes-Isidori-Normalform

        instead of the state xx
        """

        n = len(self.tt)
        nq = len(self.tau)
        np = n - nq
        nx = 2*n

        # make sure that the system has the desired structure
        B = self.eqns.jacobian(self.tau)
        cond1 = B[:np, :] == sp.zeros(np, nq)
        cond2 = B[np:, :] == -sp.eye(nq)
        if not cond1 and cond2:
            msg = "The jacobian of the equations of motion do not have the expected structure: %s"
            raise NotImplementedError(msg % str(B))

        pp = self.tt[:np,:]
        qq = self.tt[np:,:]
        uu = self.ttd[:np,:]
        vv = self.ttd[np:,:]
        ww = st.symb_vector('w1:{0}'.format(np+1))
        assert len(vv) == nq

        # state w.r.t normal form
        self.zz = st.row_stack(qq, vv, pp, ww)
        self.ww = ww

        # set the actuated accelearations as new inputs
        self.aa = self.ttdd[-nq:, :]

        # input vectorfield
        self.gz = sp.zeros(nx, nq)
        self.gz[nq:2*nq, :] = sp.eye(nq)  # identity matrix for the active coordinates

        # drift vectorfield (will be completed below)
        self.fz = sp.zeros(nx, 1)
        self.fz[:nq, :] = vv

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

        # defining equation for ww: ww := uu + M11inv*M12*vv
        uu_expr = ww - M11inv*M12*vv

        # setting input tau and acceleration to 0 in the equations of motion
        C1K1 = self.eqns[:np, :].subs(st.zip0(self.ttdd, self.tau))

        N = st.time_deriv(M11inv*M12, self.tt)
        ww_dot = -M11inv*C1K1.subs(lzip(uu, uu_expr)) + N.subs(lzip(uu, uu_expr))*vv

        self.fz[2*nq:2*nq+np, :] = uu_expr
        self.fz[2*nq+np:, :] = ww_dot

        # how the new coordinates are defined:
        self.ww_def = uu + M11inv*M12*vv

        if simplify:
            self.fz.simplify()
            self.gz.simplify()
            self.ww_def.simplify()

    @property  # legacy (compatibility with older convention)
    def eq_list(self):
        return self.eqns

    @property  # convenience
    def uu(self):
        return self.tau


# noinspection PyPep8Naming
class DAE_System(object):
    """
    This class encapsulates an differential algebraic equation (DAE).
    """
    
    info = "encapsulate all dae-relevant information"

    def __init__(self, mod, parameter_values=None):
        """
        :param mod:                 SymbolicModel-instance
        :param parameter_values:    subs-compatible sequence
        """

        if parameter_values is None:
            parameter_values = []

        self.parameter_values = parameter_values

        # ensure not None nor empty sequence
        assert mod.constraints
        assert mod.llmd

        mod.xx = st.row_stack(mod.tt, mod.ttd)

        # save a reference to the whole model
        self.mod = mod

        # xxd = st.row_stack(mod.ttd, mod.ttdd)
        self.ntt = len(mod.tt)
        nx = len(mod.xx)
        self.nll = len(mod.llmd)

        # degree of freedom = (number of coordinates) - (number of constraints)
        self.ndof = self.ntt - self.nll

        # time derivative of algebraic variables (only formally needed)
        # llmdd = st.time_deriv(mod.llmd, mod.llmd)

        mod.dae = self
        self.yy = st.row_stack(mod.xx, mod.llmd)

        self.yyd = yyd = st.symb_vector("ydot1:{}".format(1 + nx + self.nll))

        # list of flags whether a variable occurs differentially (1) or only algebraically (0)
        self.diff_alg_vars = [1]*len(mod.xx) + [0]*len(mod.llmd)

        # definiotric equations
        eqns1 = yyd[:self.ntt, :] - mod.ttd

        # dynamic equations (second order; M(tt)*ttdd + ... = 0)
        eqns2 = mod.eqns.subz(mod.ttdd, yyd[self.ntt:2*self.ntt])

        self.constraints = mod.constraints = sp.Matrix(mod.constraints).subs(parameter_values)
        self.eqns = st.row_stack(eqns1, eqns2, mod.constraints).subs(parameter_values)
        self.MM = mod.MM.subs(parameter_values)

        self.eq_func = None

        self.constraints_func = None
        self.generate_eqns_func()

    def generate_eqns_func(self):
        """
        Create a callable function of the form F(ww) which internally represents the lhs of F(t, yy, yyd) = 0
        with ww = (yy, yydot, ttau). Note that the input tau will later be calculated by a controller ttau = k(t, yy).

        :return: None, set self.eq_func
        """

        fvars = st.concat_rows(self.yy, self.yyd, self.mod.tau)

        actual_symbs = self.eqns.atoms(sp.Symbol)
        expected_symbs = set(fvars)
        unexpected_symbs = actual_symbs.difference(expected_symbs)
        if unexpected_symbs:
            msg = "Equations can only converted to numerical func if all parameters are passed for substitution. " \
                  "Unexpected symbols: {}".format(unexpected_symbs)
            raise ValueError(msg)

        self.eq_func = st.expr_to_func(fvars, self.eqns)

    def generate_constraints_func(self):

        if self.constraints_func is not None:
            return

        actual_symbs = self.constraints.atoms(sp.Symbol)
        expected_symbs = set(self.mod.tt)
        if not actual_symbs == expected_symbs:
            msg = "Constraints can only converted to numerical func if all parameters are passed for substitution. " \
                  "Unexpected symbols: {}".format(actual_symbs.difference(expected_symbs))
            raise ValueError(msg)

        self.constraints_func = st.expr_to_func(self.mod.tt, self.constraints)

    def calc_constistent_conf(self, **kwargs):
        """
        Example call: calc_consistent_conf(p1=0.5, _eps=1e-12)

        :param kwargs:
        :return:
        """

        num_requests = []
        estimates = []

        # index lists fot independent and dependent vars
        indep_idcs = []
        dep_idcs = []
        for i, theta_i in enumerate(self.mod.tt):
            assert theta_i.name not in ("_ftol", "_xtol",)
            val = kwargs.get(theta_i.name)
            if val is not None:
                # num_requests.append((theta_i, val))
                num_requests.append(val)
                indep_idcs.append(i)
            else:
                # look for estimate or choose 0 otherwise
                est = kwargs.get(theta_i.name + "_estimate", 0)
                # estimates.append((theta_i, est))
                estimates.append(est)

                # this is an independent var
                dep_idcs.append(i)

        assert len(num_requests) == self.ndof
        assert len(num_requests) + len(estimates) == self.ntt

        self.generate_constraints_func()

        # construct the full argument for the constraints (zeros will be replaced at runtime)
        arg = np.zeros(self.ntt)
        arg[indep_idcs] = num_requests

        def min_target(dep_coords):
            arg[dep_idcs] = dep_coords

            return np.linalg.norm(self.constraints_func(*arg))

        ftol = kwargs.get("_ftol", 1e-8)
        xtol = kwargs.get("_xtol", 1e-8)

        c0 = np.array(estimates)
        sol = fmin(min_target, c0, ftol=ftol, xtol=xtol)

        arg[dep_idcs] = sol
        ttheta_cons = arg

        assert np.allclose(self.constraints_func(*ttheta_cons), 0)

        return ttheta_cons

    def calc_consistent_init_vals(self, xinit, llmd_guess=None):
        """
        Assume yy = (xx, llmd) and xx = (ttheta, ttheta_d)
        -> return yy_0 and and yyd_0 sucht that F(0, yy_0, yyd_0) = 0

        Note that it might be necessary to find a consistent initial configuration first (xx cannot choosen freely)

        :param xinit:       initial state (part of y)
        :param llmd_guess:  guess for initial state of llmd
        """

        if llmd_guess is None:
            llmd_guess = [0] * self.nll

        raise NotImplementedError("Not yet ready")








# TODO: this can be removed soon (2019-06-26)
# def generate_model(T, U, qq, F, **kwargs):
#     raise DeprecationWarning('generate_symbolic_model should be used')
#     """
#     T kinetic energy
#     U potential energy
#     q independend deflection variables
#     F external forces
#     D dissipation function
#     """
#
#     n = len(qq)
#
#     # time variable
#     t = sp.symbols('t')
#
#     # symbolic Variables (to prevent Functions where we not want them)
#     qs = []
#     qds = []
#     qdds = []
#
#     if not kwargs:
#         # assumptions for the symbols (facilitating the postprocessing)
#         kwargs ={"real": True}
#
#     for qi in qq:
#         # ensure that the same Symbol for t is used
#         assert qi.is_Function and qi.args == (t,)
#         s = str(qi.func)
#
#         qs.append(sp.Symbol(s, **kwargs))
#         qds.append(sp.Symbol(s + "_d",  **kwargs))
#         qdds.append(sp.Symbol(s + "_dd",  **kwargs))
#
#     qs, qds, qdds = sp.Matrix(qs), sp.Matrix(qds), sp.Matrix(qdds)
#
#     #derivative of configuration variables
#     qd = qq.diff(t)
#     qdd = qd.diff(t)
#
#     #lagrange function
#     L = T - U
#
#     #substitute functions with symbols for partial derivatives
#
#     #highest derivative first
#     subslist = lzip(qdd, qdds) + lzip(qd, qds) + lzip(qq, qs)
#     L = L.subs(subslist)
#
#     # partial derivatives of L
#     Ldq = st.jac(L, qs)
#     Ldqd = st.jac(L, qds)
#
#     # generalised external force
#     f = sp.Matrix(F)
#
#     # substitute symbols with functions for time derivative
#     subslistrev = st.rev_tuple(subslist)
#     Ldq = Ldq.subs(subslistrev)
#     Ldqd = Ldqd.subs(subslistrev)
#     Ldqd = Ldqd.diff(t)
#
#     #lagrange equation 2nd kind
#     model_eq = qq * 0
#     for i in range(n):
#         model_eq[i] = Ldqd[i] - Ldq[i] - f[i]
#
#     # model equations with symbols
#     model_eq = model_eq.subs(subslist)
#
#     # create object of model
#     model1 = SymbolicModel()  # model_eq, qs, f, D)
#     model1.eqns = model_eq
#     model1.qs = qs
#     model1.extforce_list = f
#     model1.tau = f
#
#     model1.qds = qds
#     model1.qdds = qdds
#
#
#     # also store kinetic and potential energy
#     model1.T = T
#     model1.U = U
#
#     # analyse the model
#
#     return model1


# TODO add remark stating that due to construction, the equations stored in
# the returned `SymbolicModel` under `eqns`
# noinspection PyPep8Naming
def generate_symbolic_model(T, U, tt, F, simplify=True, constraints=None, **kwargs):
    """
    T:             kinetic energy
    U:             potential energy
    tt:            sequence of independent deflection variables ("theta")
    F:             external forces
    simplify:      determines whether the equations of motion should be simplified
                   (default=True)
    constraints:   None (default) or sequence of constraints (will introduce lagrange multipliers)

    kwargs: optional assumptions like 'real=True'
    """
    n = len(tt)

    for theta_i in tt:
        assert isinstance(theta_i, sp.Symbol)

    if constraints is None:
        constraints_flag = False
        # ensure well define calculations (jacobian of empty matrix would not be possible)
        constraints = [0]
    else:
        constraints_flag = True
    assert len(constraints) > 0
    constraints = sp.Matrix(constraints)
    assert constraints.shape[1] == 1
    nC = constraints.shape[0]
    jac_constraints = constraints.jacobian(tt)

    llmd = st.symb_vector("lambda_1:{}".format(nC+1))

    F = sp.Matrix(F)

    if F.shape[0] == 1:
        # convert to column vector
        F = F.T
    if not F.shape == (n, 1):
        msg = "Vector of external forces has the wrong length. Should be " + \
        str(n) + " but is %i!"  % F.shape[0]
        raise ValueError(msg)

    # introducing symbols for the derivatives
    tt = sp.Matrix(tt)
    ttd = st.time_deriv(tt, tt, **kwargs)
    ttdd = st.time_deriv(tt, tt, order=2, **kwargs)

    # Lagrange function
    L = T - U

    if not T.atoms().intersection(ttd) == set(ttd):
        raise ValueError('Not all velocity symbols do occur in T')

    # partial derivatives of L
    L_diff_tt = st.jac(L, tt)
    L_diff_ttd = st.jac(L, ttd)

    # prov_deriv_symbols = [ttd, ttdd]

    # time-depended_symbols
    tds = list(tt) + list(ttd)
    L_diff_ttd_dt = st.time_deriv(L_diff_ttd, tds, **kwargs)

    # constraints

    constraint_terms = list(llmd.T*jac_constraints)

    # lagrange equation 1st kind (which include 2nd kind as special case if constraints are empty)

    model_eq = sp.zeros(n, 1)
    for i in range(n):
        model_eq[i] = L_diff_ttd_dt[i] - L_diff_tt[i] - F[i] - constraint_terms[i]

    # create object of model
    mod = SymbolicModel()  # model_eq, qs, f, D)
    mod.eqns = model_eq

    mod.extforce_list = F
    reduced_F = sp.Matrix([s for s in F if st.is_symbol(s)])
    mod.F = reduced_F
    mod.tau = reduced_F
    if constraints_flag:
        mod.llmd = llmd
        mod.constraints = constraints
    else:
        # omit fake constraint [0]
        mod.constraints = None
        mod.llmd = None

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

    if simplify:
        mod.eqns.simplify()

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


def transform_2nd_to_1st_order_matrices(P0, P1, P2, xx):
    """Transforms an implicit second order (tangential) representation of a mechanical system to
    an first order representation (needed to apply the "Franke-approach")

    :param P0:      eqns.jacobian(theta)
    :param P1:      eqns.jacobian(theta_d)
    :param P2:      eqns.jacobian(theta_dd)
    :param xx:      vector of state variables

    :return:   P0_bar, P1_bar

    with P0_bar = implicit_state_equations.jacobian(x)
    and  P1_bar = implicit_state_equations.jacobian(x_d)
    """

    assert P0.shape == P1.shape == P2.shape
    assert xx.shape == (P0.shape[1]*2, 1)

    xxd = st.time_deriv(xx, xx)

    N = int(xx.shape[0]/2)

    # definitional equations like xdot1 - x3 = 0 (for N=2)
    eqns_def = xx[N:, :] - xxd[:N, :]
    eqns_mech = P2*xxd[N:, :] + P1*xxd[:N, :] + P0*xx[:N, :]

    F = st.row_stack(eqns_def, eqns_mech)

    P0_bar = F.jacobian(xx)
    P1_bar = F.jacobian(xxd)

    return P0_bar, P1_bar

