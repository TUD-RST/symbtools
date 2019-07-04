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


# noinspection PyPep8Naming
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


# noinspection PyPep8Naming
def symbColVector(n, s='a'):
    # noinspection PyTypeChecker
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
        >>> symbColVector(1, 3)
        Traceback (most recent call last):
            ...
        TypeError: unsupported operand type(s) for +: 'int' and 'str'
        """
    if n <= 0 or int(n) != n:
        raise ValueError("Positive integer required")

    A = sp.Matrix(n, 1, lambda i, j: sp.Symbol(s + '%i' % (i + 1)))
    return A


def new_model_from_equations_of_motion(eqns, ttheta, tau):
    """

    :param eqns:    vector of equations of motion
    :param ttheta:  generalized coordinates
    :param tau:     input
    :return:        SymbolicModel instance
    """

    mod = SymbolicModel()
    mod.eqns = eqns
    mod.tt = ttheta
    mod.ttd = st.time_deriv(ttheta, ttheta)
    mod.ttdd = st.time_deriv(ttheta, ttheta, order=2)
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
        # for backward compatibility (can be dropped in 2020):
        if hasattr(self, 'ttdd'):
            self.M = self.eqns.jacobian(self.ttdd)
        else:
            # noinspection PyUnresolvedReferences
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

    def calc_dae_eq(self, parameter_values=None):
        """
        In case of present constraints ode representation is not possible.
        This method constructs a representation F(y, ydot) = 0.

        Such a form can be passed to a DAE solver like IDA (from SUNDIALS / Assimulo)

        :return: dae (Container); also set self.dae = dae , self.dae.yy, self.dae.yyd, self.dae.eqns, ...
        """

        self.dae = DAE_System(self, parameter_values=parameter_values)

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
        self.constraints_d = None
        self.constraints_dd = None

        self.eqns = st.row_stack(eqns1, eqns2, mod.constraints).subs(parameter_values)
        self.MM = mod.MM.subs(parameter_values)

        self.eq_func = None  # internal representation (with signature F(ww) with ww = (yy, yyd, tau)
        self.deq_func = None  # internal representation of ode-part (with signature F(ww) with ww = (yy, yyd, tau)

        self.model_func = None  # solver-friendly representation (with signature F(t, yy, yyd))

        self.constraints_func = None
        self.constraints_d_func = None
        self.constraints_dd_func = None
        self.leqs_acc_lmd_func = None
        self.acc_of_lmd_func = None

        # default input-function
        u_zero = np.zeros((len(self.mod.tau)))

        # noinspection PyUnusedLocal
        def input_func(t):
            return u_zero

        self.input_func = input_func

    def generate_eqns_funcs(self, parameter_values=None):
        """
        Creates two callable functions.

        The first of the form F_tilde(ww) which *internally* represents the lhs of F(t, yy, yyd) = 0
        with ww = (yy, yydot, ttau). Note that the input tau will later be calculated by a controller ttau = k(t, yy).

        The second: F itself with the signature as above.

        :return: None, set self.eq_func
        """

        if self.eq_func is not None and self.model_func is not None:
            return

        if parameter_values is None:
            parameter_values = []

        # also respect those values, which have been passed to the constructor
        parameter_values = list(self.parameter_values) + list(parameter_values)

        # create needed helper function:

        self.gen_leqs_for_acc_llmd(parameter_values=parameter_values)

        # avoid dot access in the internal function below
        ntt, nll, acc_of_lmd_func = self.ntt, self.nll, self.acc_of_lmd_func

        fvars = st.concat_rows(self.yy, self.yyd, self.mod.tau)

        # keep self.eqns unchanged (maybe not necessary)
        eqns = self.eqns.subs(parameter_values)

        actual_symbs = eqns.atoms(sp.Symbol)
        expected_symbs = set(fvars)
        unexpected_symbs = actual_symbs.difference(expected_symbs)
        if unexpected_symbs:
            msg = "Equations can only be converted to numerical func if all parameters are passed for substitution. " \
                  "Unexpected symbols: {}".format(unexpected_symbs)
            raise ValueError(msg)

        # full equations in classical formulation (currently not needed internally)
        self.eq_func = st.expr_to_func(fvars, eqns)

        # only the ode part
        self.deq_func = st.expr_to_func(fvars, eqns[:2*self.ntt, :])

        def model_func(t, yy, yyd):
            """
            This function is intended to be passed to a DAE solver like IDA.

            The model consists of two coupled parts: ODE part F_ode(yy, yydot)=0 and algebraic C(yy)=0 part.

            Problem is, that for mechanical systems with constraints we have differential index=3,
            i.e. C and C_dot not depend on llmd. C_ddot can be formulated to depend on llmd (if F_ode is plugged in).

            Idea: instead fo returning just C we return C**2 + C_dot**2 + C_ddot**2

            :param t:
            :param yy:
            :param yyd:
            :return: F(t, yy, yyd) (should be 0 with shape (2*ntt + nll,))
            """

            # to use a controller, this needs to be more sophisticated
            external_forces = self.input_func(t)
            args = np.concatenate((yy, yyd, external_forces))

            ttheta = yy[:ntt]
            ttheta_d = yy[ntt:2*ntt]

            # not needed, just for comprehension
            # llmd = yy[2*ntt:2*ntt + nll]

            ode_part = self.deq_func(*args)

            # now calculate the accelerations in depencency of yy (and thus in dependency of llmd)
            # Note: signature: acc_of_lmd_func(yy, ttau)
            ttheta_dd = acc_of_lmd_func(*np.concatenate((yy, external_forces)))
            c2 = self.constraints_dd_func(*np.concatenate((ttheta, ttheta_d, ttheta_dd)))
            c2 = np.atleast_1d(c2)

            res = np.concatenate((ode_part, c2))

            return res

        self.model_func = model_func

    def generate_constraints_funcs(self):

        if self.constraints_func is not None:
            return

        actual_symbs = self.constraints.atoms(sp.Symbol)
        expected_symbs = set(self.mod.tt)
        if not actual_symbs == expected_symbs:
            msg = "Constraints can only converted to numerical func if all parameters are passed for substitution. " \
                  "Unexpected symbols: {}".format(actual_symbs.difference(expected_symbs))
            raise ValueError(msg)

        self.constraints_func = st.expr_to_func(self.mod.tt, self.constraints)

        # now we need also the differentiated constraints (e.g. to calculate consistent velocities and accelerations)

        self.constraints_d = st.time_deriv(self.constraints, self.mod.tt)
        self.constraints_dd = st.time_deriv(self.constraints_d, self.mod.tt)

        xx = st.row_stack(self.mod.tt, self.mod.ttd)

        # this function depends on coordinates ttheta and velocities ttheta_dot
        self.constraints_d_func = st.expr_to_func(xx, self.constraints_d)

        zz = st.row_stack(self.mod.tt, self.mod.ttd, self.mod.ttdd)

        # this function depends on coordinates ttheta and velocities ttheta_dot and accel
        self.constraints_dd_func = st.expr_to_func(zz, self.constraints_dd)

    def gen_leqs_for_acc_llmd(self, parameter_values=None):
        """
        Create a callable function which returns A, bnum of the linear eqn-system A*ww = bnum, where ww = (ttheta_dd, llmnd)


        :return: None, set self.leqs_acc_lmd_func
        """

        if self.leqs_acc_lmd_func is not None and self.acc_of_lmd_func is not None:
            return

        if parameter_values is None:
            parameter_values = []

        ntt = self.ntt
        nll = self.nll

        self.generate_constraints_funcs()

        # also respect those values, which have been passed to the constructor
        parameter_values = list(self.parameter_values) + list(parameter_values)

        # we use mod.eqns here because we do not want ydot-vars inside
        eqns = st.concat_rows(self.mod.eqns.subs(parameter_values), self.constraints_dd)

        ww = st.concat_rows(self.mod.ttdd, self.mod.llmd)

        A = eqns.jacobian(ww)
        b = -eqns.subz0(ww)  # rhs of the leqs

        Ab = st.concat_cols(A, b)

        fvars = st.concat_rows(self.mod.tt, self.mod.ttd, self.mod.tau)

        actual_symbs = Ab.atoms(sp.Symbol)
        expected_symbs = set(fvars)
        unexpected_symbs = actual_symbs.difference(expected_symbs)
        if unexpected_symbs:
            msg = "Equations can only converted to numerical func if all parameters are passed for substitution. " \
                  "Unexpected symbols: {}".format(unexpected_symbs)
            raise ValueError(msg)

        A_fnc = st.expr_to_func(fvars, A, keep_shape=True)
        b_fnc = st.expr_to_func(fvars, b)

        nargs = len(fvars)

        # noinspection PyShadowingNames
        def leqs_acc_lmd_func(*args):
            """
            Calculate the matrices of the linear equation system for ttheta and llmd.
            Assume args = (ttheta, theta_d, ttau)
            :param args:
            :return:
            """
            assert len(args) == nargs
            Anum = A_fnc(*args)
            bnum = b_fnc(*args)

            # theese arrays can now be passed to a linear equation solver
            return Anum, bnum

        self.leqs_acc_lmd_func = leqs_acc_lmd_func

        def acc_of_lmd_func(*args):
            """
            Calculate ttheta in dependency of args= (yy, ttau) = ((ttheta, ttheta_d, llmd), ttau)

            :param args:
            :return:
            """

            ttheta = args[:ntt]
            ttheta_d = args[ntt:2*ntt]
            llmd = args[2*ntt:2*ntt+nll]
            ttau = args[2*ntt+nll:]

            args1 = np.concatenate((ttheta, ttheta_d, ttau))

            Anum = A_fnc(*args1)
            A1 = Anum[:ntt, :ntt]
            A2 = Anum[:ntt, ntt:]

            b1 = b_fnc(*args1)[:ntt]

            ttheta_dd_res = np.linalg.solve(A1, b1 - np.dot(A2, llmd))

            return ttheta_dd_res

        self.acc_of_lmd_func = acc_of_lmd_func

    def calc_constistent_conf_vel(self, **kwargs):
        """
        Example call: calc_consistent_conf(p1=0.5, pdot1=-1.2, p2_estimate=-2, _ftol=1e-12)

        Notes,
            - There must self.ndof coordinates be specified. These will be the independent variables.
            - If velocities (of indep. vars) are not given explicitly, they are assumed to be zero.
            - If estimates (guess) for coords or velocities (of dep. vars) are not given explicitly,
              they are assumed to be zero.

        :param kwargs:
        :return:
        """

        # for coordinates and velocities
        c_num_requests = []
        c_estimates = []
        v_num_requests = []
        v_estimates = []

        disp_flag = kwargs.get("_disp", True)

        option_names = {"_ftol", "_xtol", "_disp"}

        # check if (accidentally) wrong symbol names have been passed (like 'q2d' instead of 'qdot2')
        # or if symbol names clash with additional options
        keys = set(kwargs.keys())
        valid_symbol_names = set([s.name for s in list(self.mod.tt) + list(self.mod.ttd)])
        valid_estimate_names = set([n+"_estimate" for n in valid_symbol_names])

        assert not valid_symbol_names.intersection(option_names)
        assert not valid_estimate_names.intersection(option_names)

        valid_keys = option_names.union(valid_symbol_names, valid_estimate_names)

        invalid_keys = keys - valid_keys

        if invalid_keys:
            msg = "The following invalid keywords were passed: {}. " \
                  "Valid keywords are a subset of {}".format(invalid_keys, valid_keys)
            raise ValueError(msg)

        # index lists for independent and dependent vars
        indep_idcs = []
        dep_idcs = []
        for i, (theta_i, theta_dot_i) in enumerate(zip(self.mod.tt, self.mod.ttd)):

            # first handle coordinates
            c_val = kwargs.get(theta_i.name)
            if c_val is not None:
                c_num_requests.append(c_val)
                indep_idcs.append(i)
            else:
                # look for estimate or choose 0 otherwise
                c_est = kwargs.get(theta_i.name + "_estimate", 0)
                c_estimates.append(c_est)

                # this is an independent var
                dep_idcs.append(i)

            # now look for velocities
            if c_val is not None:
                # this is an indep. var
                v_val = kwargs.get(theta_dot_i.name, 0)
                v_num_requests.append(v_val)
            else:
                assert theta_dot_i.name not in kwargs
                v_est = kwargs.get(theta_dot_i.name + "_estimate", 0)
                v_estimates.append(v_est)

        assert len(c_num_requests) == self.ndof
        assert len(c_num_requests) + len(c_estimates) == self.ntt
        assert len(v_num_requests) == self.ndof
        assert len(v_num_requests) + len(v_estimates) == self.ntt

        self.generate_constraints_funcs()

        # construct the full argument for the constraints (zeros will be replaced at runtime)
        arg_c = np.zeros(self.ntt)
        arg_c[indep_idcs] = c_num_requests

        def min_target_c(dep_coords):
            # write the dependent values at the corresponding places in the array
            arg_c[dep_idcs] = dep_coords

            return np.linalg.norm(self.constraints_func(*arg_c))

        ftol = kwargs.get("_ftol", 1e-8)
        xtol = kwargs.get("_xtol", 1e-8)

        c0 = np.array(c_estimates)
        sol_c = fmin(min_target_c, c0, ftol=ftol, xtol=xtol, disp=disp_flag)

        arg_c[dep_idcs] = sol_c
        ttheta_cons = arg_c

        assert np.allclose(self.constraints_func(*ttheta_cons), 0)

        # now calculate velocities

        arg_v = np.zeros(self.ntt)
        arg_v[indep_idcs] = v_num_requests

        def min_target_v(dep_vels):
            # write the dependent values at the corresponding places in the velocity-array
            arg_v[dep_idcs] = dep_vels

            arg_cv = np.concatenate((ttheta_cons, arg_v))
            return np.linalg.norm(self.constraints_d_func(*arg_cv))

        v0 = np.array(v_estimates)
        sol_v = fmin(min_target_v, v0, ftol=ftol, xtol=xtol, disp=disp_flag)

        # ensure that solution was successful
        assert np.allclose(min_target_v(sol_v), 0)

        arg_v[dep_idcs] = sol_v
        ttheta_dot_cons = arg_v

        return ttheta_cons, ttheta_dot_cons

    def calc_consistent_accel_lmd(self, xx, t=0):
        """
        This function solves numerically the the equations system
         M(ttheta) * ttheta_dd + ... + H(llmd) = 0
         Constraints_dd(ttheta, ttheta_d, ttheta_dd) = 0

        It consists of ntt + nll equations in the same number of variables: (ttheta_dd, llmd).

        External forces are assumed to be given by a function (open or closed loop controller).

        :param xx:              numerical values for (ttheta, ttheta_dot)
        :param t:               time (default: 0)
        :return:
        """

        # speed up trick (omit dots in inner loops)
        ntt = self.ntt
        nll = self.nll

        if isinstance(xx, (tuple, list)) and len(xx) == 2 \
           and isinstance(xx[0], np.ndarray) and isinstance(xx[1], np.ndarray):

            xx = np.concatenate(xx)

        assert xx.shape == (2*ntt,)

        external_forces = self.input_func(t)

        self.gen_leqs_for_acc_llmd()
        A, b = self.leqs_acc_lmd_func(*np.r_[xx, external_forces])
        sol = np.linalg.solve(A, b)

        acc = sol[:ntt]
        llmd = sol[-nll:]

        return acc, llmd

    def calc_consistent_init_vals(self, t=0, **kwargs):
        """
        Assume yy = (xx, llmd) and xx = (ttheta, ttheta_d)
        -> return yy_0 and and yyd_0 sucht that F(0, yy_0, yyd_0) = 0

        Note that it might be necessary to find a consistent initial configuration first (xx cannot choosen freely)

        :param t:           time variabel (needed to evaluate the external_input_func)
        :param kwargs:      conditions (and estimates) for independent (and dependent) coordinates and velocities
                            see calc_constistent_conf_vel
        """

        ttheta, ttheta_d = self.calc_constistent_conf_vel(**kwargs)

        # noinspection PyTypeChecker
        acc, llmd = self.calc_consistent_accel_lmd((ttheta, ttheta_d), t=t)

        yy = np.concatenate((ttheta, ttheta_d, llmd))
        yyd = np.concatenate((ttheta_d, acc, llmd*0))

        # a b c # write unit test for this function, then try IDA algorithm

        return yy, yyd


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

