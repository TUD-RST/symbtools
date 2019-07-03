# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 11:25:00 2014

@author: Carsten Knoll
"""

import unittest
import sympy as sp
from sympy import sin, cos, Matrix
import symbtools as st
import numpy as npy
import symbtools.modeltools as mt
from symbtools.modeltools import Rz
import sys


from IPython import embed as IPS


# noinspection PyPep8Naming
class ModelToolsTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_simple1(self):
        q1, = qq  = sp.Matrix(sp.symbols('q1,'))
        F1, = FF  = sp.Matrix(sp.symbols('F1,'))

        m = sp.Symbol('m')

        q1d = st.time_deriv(q1, qq)
        q1dd = st.time_deriv(q1, qq, order=2)

        T = q1d**2*m/2
        V = 0

        mod = mt.generate_symbolic_model(T, V, qq, FF)

        eq = m*q1dd - F1

        self.assertEqual(mod.eqns[0], eq)

        # test the application of the @property
        M = mod.MM
        self.assertEqual(M[0], m)

    def test_simple_constraints(self):
        q1, q2 = qq = sp.Matrix(sp.symbols('q1, q2'))
        F1, = FF = sp.Matrix(sp.symbols('F1,'))

        m = sp.Symbol('m')

        qdot1, qdot2 = st.time_deriv(qq, qq)
        # q1dd, q2dd = st.time_deriv(qq, qq, order=2)

        T = qdot1**2*m/4 + qdot2**2*m/4
        V = 0

        mod = mt.generate_symbolic_model(T, V, qq, [F1, 0], constraints=[q1-q2])
        self.assertEqual(mod.llmd.shape, (1, 1))
        self.assertEqual(mod.constraints[0], q1 - q2)

        # noinspection PyUnusedLocal
        with self.assertRaises(ValueError) as cm:
            # no parameters passed
            dae = mod.calc_dae_eq()
            dae.generate_eqns_funcs()

        dae = mod.calc_dae_eq(parameter_values=[(m, 1)])

        ttheta_1, ttheta_d_1 = dae.calc_constistent_conf_vel(q1=0.123, _disp=False)

        self.assertTrue(npy.allclose(ttheta_1, npy.array([0.123, 0.123])))
        self.assertTrue(npy.allclose(ttheta_d_1, npy.array([0.0, 0.0])))

        ttheta_2, ttheta_d_2 = dae.calc_constistent_conf_vel(q2=567, qdot2=-100, _disp=False)

        self.assertTrue(npy.allclose(ttheta_2, npy.array([567., 567.])))
        self.assertTrue(npy.allclose(ttheta_d_2, npy.array([-100.0, -100.0])))

        # noinspection PyUnusedLocal
        with self.assertRaises(ValueError) as cm:
            # wrong argument
            dae.calc_constistent_conf_vel(q2=567, q2d=-100, _disp=False)

        # noinspection PyUnusedLocal
        with self.assertRaises(ValueError) as cm:
            # unexpected argument
            dae.calc_constistent_conf_vel(q2=567, qdot2=-100, foobar="fnord", _disp=False)

        dae.gen_leqs_for_acc_llmd()
        A, b = dae.leqs_acc_lmd_func(*npy.r_[ttheta_1, ttheta_1, 1])

        self.assertEqual(A.shape, (dae.ntt+dae.nll,)*2)
        self.assertEqual(b.shape, (dae.ntt+dae.nll,))
        self.assertTrue(npy.allclose(b, [1, 0, 0]))

        # in that situation there is no force
        acc_1, llmd_1 = dae.calc_consistent_accel_lmd((ttheta_1, ttheta_d_1))
        self.assertTrue(npy.allclose(acc_1, [0, 0]))
        self.assertTrue(npy.allclose(llmd_1, [0, 0]))

        # in that situation there is also no force
        acc_2, llmd_2 = dae.calc_consistent_accel_lmd((ttheta_2, ttheta_d_2))
        self.assertTrue(npy.allclose(acc_2, [0, 0]))
        self.assertTrue(npy.allclose(llmd_2, [0, 0]))

        yy_1, yyd_1 = dae.calc_consistent_init_vals(q2=12.7, qdot2=100)

        dae.generate_eqns_funcs()
        res = dae.model_func(0, yy_1, yyd_1)

        self.assertEqual(res.shape, (dae.ntt*2 + dae.nll,))
        self.assertTrue(npy.allclose(res, 0))

        # End of test_simple_constraints

    def test_four_bar_constraints(self):

        # 2 passive Gelenke
        np = 2
        nq = 1
        p1, p2 = pp = st.symb_vector("p1:{0}".format(np + 1))
        q1, = qq = st.symb_vector("q1:{0}".format(nq + 1))

        ttheta = st.row_stack(pp, qq)
        pdot1, pdot2, qdot1 = tthetad = st.time_deriv(ttheta, ttheta)
        tthetadd = st.time_deriv(ttheta, ttheta, order=2)
        st.make_global(ttheta, tthetad, tthetadd)

        params = sp.symbols('s1, s2, s3, m1, m2, m3, J1, J2, J3, l1, l2, l3, kappa1, kappa2, g')
        s1, s2, s3, m1, m2, m3, J1, J2, J3, l1, l2, l3, kappa1, kappa2, g = params
        st.make_global(params)

        parameter_values = dict(s1=1/2, s2=1/2, s3=1/2, m1=1, m2=1, m3=3, J1=1/12, J2=1/12, J3=1/12, l1=1, l2=1.5,
                                l3=1.5, kappa1=3/2, kappa2=14.715, g=9.81).items()

        # ttau = sp.symbols('tau')
        tau1, tau2 = st.symb_vector("tau1, tau2")

        # unit vectors
        ex = sp.Matrix([1, 0])

        # Basis 1 and 2
        # B1 = sp.Matrix([0, 0])
        B2 = sp.Matrix([2*l1, 0])

        # Koordinaten der Schwerpunkte und Gelenke
        S1 = Rz(q1)*ex*s1
        G1 = Rz(q1)*ex*l1
        S2 = G1 + Rz(q1 + p1)*ex*s2
        G2 = G1 + Rz(q1 + p1)*ex*l2

        # one link manioulator
        G2b = B2 + Rz(p2)*ex*l3
        S3 = B2 + Rz(p2)*ex*s3

        # timederivatives of centers of masses
        Sd1, Sd2, Sd3 = st.col_split(st.time_deriv(st.col_stack(S1, S2, S3), ttheta))

        # kinetic energy

        T_rot = (J1*qdot1 ** 2 + J2*(qdot1 + pdot1) ** 2 + J3*(pdot2) ** 2)/2
        T_trans = (m1*Sd1.T*Sd1 + m2*Sd2.T*Sd2 + m3*Sd3.T*Sd3)/2

        T = T_rot + T_trans[0]

        # potential energy
        V = m1*g*S1[1] + m2*g*S2[1] + m3*g*S3[1]

        # # Kinetische Energie mit Platzhaltersymbolen einführen (jetzt nicht):
        #
        # M1, M2, M3 = MM = st.symb_vector('M1:4')
        # MM_subs = [(J1 + m1*s1 ** 2 + m2*l1 ** 2, M1), (J2 + m2*s2 ** 2, M2), (m2*l1*s2, M3)]
        # # MM_rplm = st.rev_tuple(MM_subs)  # Umkehrung der inneren Tupel -> [(M1, J1+... ), ...]

        external_forces = [0, 0, tau1]

        mod = mt.generate_symbolic_model(T, V, ttheta, external_forces, constraints=[G2 - G2b])

        self.assertEqual(mod.llmd.shape, (2, 1))
        self.assertEqual(mod.constraints.shape, (2, 1))

        dae = mod.calc_dae_eq(parameter_values=parameter_values)

        nc = len(mod.llmd)
        self.assertEqual(dae.eqns[-nc:, :], mod.constraints)

        # qdot1 = 0
        ttheta_1, ttheta_d_1 = dae.calc_constistent_conf_vel(q1=npy.pi/4, _disp=False)

        eres_c = npy.array([-0.2285526,  1.58379902,  0.78539816])
        eres_v = npy.array([0, 0, 0])

        self.assertTrue(npy.allclose(ttheta_1, eres_c))
        self.assertTrue(npy.allclose(ttheta_d_1, eres_v))

        # in that situation there is no force
        acc_1, llmd_1 = dae.calc_consistent_accel_lmd((ttheta_1, ttheta_d_1))
        # these values seem reasonable but have yet not been checked analytically
        self.assertTrue(npy.allclose(acc_1, [13.63475466, -1.54473017, -8.75145644]))
        self.assertTrue(npy.allclose(llmd_1, [-0.99339947,  0.58291489]))

        # qdot1 ≠ 0
        ttheta_2, ttheta_d_2 = dae.calc_constistent_conf_vel(q1=npy.pi/8*7, qdot1=3, _disp=False)

        eres_c = npy.array([-0.85754267,  0.89969149,  0.875])*npy.pi
        eres_v = npy.array([-3.42862311,  2.39360715,  3.])

        self.assertTrue(npy.allclose(ttheta_2, eres_c))
        self.assertTrue(npy.allclose(ttheta_d_2, eres_v))

        yy_1, yyd_1 = dae.calc_consistent_init_vals(q1=12.7, qdot1=100)

        dae.generate_eqns_funcs()
        res = dae.model_func(0, yy_1, yyd_1)

        self.assertEqual(res.shape, (dae.ntt*2 + dae.nll,))
        self.assertTrue(npy.allclose(res, 0))

    def test_cart_pole(self):
        p1, q1 = ttheta = sp.Matrix(sp.symbols('p1, q1'))
        F1, = FF = sp.Matrix(sp.symbols('F1,'))

        params = sp.symbols('m0, m1, l1, g')
        m0, m1, l1, g = params

        pdot1 = st.time_deriv(p1, ttheta)
        # q1dd = st.time_deriv(q1, ttheta, order=2)

        ex = Matrix([1, 0])
        ey = Matrix([0, 1])

        S0 = ex*q1  # joint of the pendulum
        S1 = S0 - mt.Rz(p1)*ey*l1  # center of mass

        # velocity
        S0d = st.time_deriv(S0, ttheta)
        S1d = st.time_deriv(S1, ttheta)

        T_rot = 0  # no moment of inertia (mathematical pendulum)
        T_trans = ( m0*S0d.T*S0d + m1*S1d.T*S1d )/2
        T = T_rot + T_trans[0]

        V = m1*g*S1[1]

        # noinspection PyUnusedLocal
        with self.assertRaises(ValueError) as cm:
            # wrong length of external forces
            mt.generate_symbolic_model(T, V, ttheta, [F1])

        # noinspection PyUnusedLocal
        with self.assertRaises(ValueError) as cm:
            # wrong length of external forces
            mt.generate_symbolic_model(T, V, ttheta, [F1, 0, 0])

        QQ = sp.Matrix([0, F1])
        mod = mt.generate_symbolic_model(T, V, ttheta, [0, F1])
        mod_1 = mt.generate_symbolic_model(T, V, ttheta, QQ)
        mod_2 = mt.generate_symbolic_model(T, V, ttheta, QQ.T)
        
        self.assertEqual(mod.eqns, mod_1.eqns)
        self.assertEqual(mod.eqns, mod_2.eqns)
        
        mod.eqns.simplify()

        M = mod.MM
        M.simplify()

        M_ref = Matrix([[l1**2*m1, l1*m1*cos(p1)], [l1*m1*cos(p1), m1 + m0]])
        self.assertEqual(M, M_ref)

        # noinspection PyUnusedLocal
        rest = mod.eqns.subs(st.zip0(mod.ttdd))
        # noinspection PyUnusedLocal
        rest_ref = Matrix([[g*l1*m1*sin(p1)], [-F1 - l1*m1*pdot1**2*sin(p1)]])

        self.assertEqual(M, M_ref)

        mod.calc_state_eq(simplify=True)
        mod.calc_coll_part_lin_state_eq(simplify=True)

        pdot1, qdot1 = mod.ttd

        ff_ref = sp.Matrix([[pdot1],[qdot1], [-g*sin(p1)/l1], [0]])
        gg_ref = sp.Matrix([[0], [0], [-cos(p1)/l1], [1]])

        self.assertEqual(mod.ff, ff_ref)
        self.assertEqual(mod.gg, gg_ref)

    def test_cart_pole_with_dissi(self):
        p1, q1 = ttheta = sp.Matrix(sp.symbols('p1, q1'))
        F1, tau_dissi = sp.Matrix(sp.symbols('F1, tau_dissi'))

        params = sp.symbols('m0, m1, l1, g, d1')
        m0, m1, l1, g, d1 = params

        pdot1 = st.time_deriv(p1, ttheta)
        # q1dd = st.time_deriv(q1, ttheta, order=2)

        ex = Matrix([1, 0])
        ey = Matrix([0, 1])

        S0 = ex*q1  # joint of the pendulum
        S1 = S0 - mt.Rz(p1)*ey*l1  # center of mass

        # velocity
        S0d = st.time_deriv(S0, ttheta)
        S1d = st.time_deriv(S1, ttheta)

        T_rot = 0  # no moment of inertia (mathematical pendulum)
        T_trans = (m0*S0d.T*S0d + m1*S1d.T*S1d)/2
        T = T_rot + T_trans[0]

        V = m1*g*S1[1]

        QQ = sp.Matrix([tau_dissi, F1])
        mod = mt.generate_symbolic_model(T, V, ttheta, QQ)

        mod.substitute_ext_forces(QQ, [d1*pdot1, F1], [F1])
        self.assertEqual(len(mod.tau), 1)
        self.assertEqual(len(mod.extforce_list), 1)
        self.assertEqual(mod.tau[0], F1)

        mod.calc_coll_part_lin_state_eq()

    def test_simple_pendulum_with_actuated_mountpoint(self):

        np = 1
        nq = 2
        n = np + nq
        pp = st.symb_vector("p1:{0}".format(np+1))
        qq = st.symb_vector("q1:{0}".format(nq+1))

        p1, q1, q2 = ttheta = st.row_stack(pp, qq)
        pdot1, qdot1, qdot2 = st.time_deriv(ttheta, ttheta)
        params = sp.symbols('l3, l4, s3, s4, J3, J4, m1, m2, m3, m4, g')
        l3, l4, s3, s4, J3, J4, m1, m2, m3, m4, g = params

        tau1, tau2 = st.symb_vector("tau1, tau2")

        # Geometry

        ex = sp.Matrix([1,0])
        ey = sp.Matrix([0,1])

        # Coordinates of centers of masses (com) and joints
        S1 = ex*q1
        S2 = ex*q1 + ey*q2
        G3 = S2  # Joints

        # com of pendulum (points upwards)
        S3 = G3 + mt.Rz(p1)*ey*s3

        # timederivatives
        Sd1, Sd2, Sd3 = st.col_split(st.time_deriv(st.col_stack(S1, S2, S3), ttheta)) ##

        # Energy
        T_rot = ( J3*pdot1**2 )/2
        T_trans = ( m1*Sd1.T*Sd1 + m2*Sd2.T*Sd2 + m3*Sd3.T*Sd3 )/2
        T = T_rot + T_trans[0]
        V = m1*g*S1[1] + m2*g*S2[1] + m3*g*S3[1]

        external_forces = [0, tau1, tau2]
        assert not any(external_forces[:np])
        mod = mt.generate_symbolic_model(T, V, ttheta, external_forces)
        mod.calc_coll_part_lin_state_eq(simplify=True)

        # Note: pdot1, qdot1, qdot2 = mod.ttd

        ff_ref = sp.Matrix([[pdot1], [qdot1], [qdot2], [g*m3*s3*sin(p1)/(J3 + m3*s3**2)], [0], [0]])
        gg_ref_part = sp.Matrix([m3*s3*cos(p1)/(J3 + m3*s3**2), m3*s3*sin(p1)/(J3 + m3*s3**2)]).T

        self.assertEqual(mod.ff, ff_ref)
        self.assertEqual(mod.gg[-3, :], gg_ref_part)

    def test_two_link_manipulator(self):
        p1, q1 = ttheta = sp.Matrix(sp.symbols('p1, q1'))
        pdot1, qdot1 = st.time_deriv(ttheta, ttheta)
        tau1, = sp.Matrix(sp.symbols('F1,'))

        params = sp.symbols('m1, m2, l1, l2, s1, s2, J1, J2')
        m1, m2, l1, l2, s1, s2, J1, J2 = params

        ex = Matrix([1, 0])
        # noinspection PyUnusedLocal
        ey = Matrix([0, 1])

        S1 = mt.Rz(q1)*ex*s1  # center of mass (first link)
        G1 = mt.Rz(q1)*ex*l1  # first joint
        S2 = G1 + mt.Rz(q1+p1)*ex*s2

        # velocity
        S1d = st.time_deriv(S1, ttheta)
        S2d = st.time_deriv(S2, ttheta)

        T_rot = (J1*qdot1**2 + J2*(qdot1 + pdot1)**2) / 2

        T_trans = ( m1*S1d.T*S1d + m2*S2d.T*S2d )/2
        T = T_rot + T_trans[0]

        V = 0

        mod = mt.generate_symbolic_model(T, V, ttheta, [tau1, 0])
        mod.calc_lbi_nf_state_eq(simplify=True)

        w1 = mod.ww[0]

        kappa = l1*m2*s2 / (J2 + m2*s2**2)
        fz4 = - kappa * qdot1*sin(p1)*(w1 - kappa*qdot1*cos(p1))
        fzref = sp.Matrix([[                               qdot1],
                           [                                   0],
                           [ (-qdot1*(1 + kappa*cos(p1) ) ) + w1],
                           [                                 fz4]])

        w_def_ref = pdot1 + qdot1*(1+kappa*cos(p1))
        self.assertEqual(mod.gz, sp.Matrix([0, 1, 0, 0]))

        fz_diff = mod.fz - fzref
        fz_diff.simplify()
        self.assertEqual(fz_diff, sp.Matrix([0, 0, 0, 0]))

        diff = mod.ww_def[0,0] - w_def_ref
        self.assertEqual(diff.simplify(), 0)

    def test_unicycle(self):
        # test the generation of Lagrange-Byrnes-Isidori-Normal form

        theta = st.symb_vector("p1, q1, q2")
        p1, q1, q2 = theta

        params = sp.symbols('l1, l2, s1, s2, delta0, delta1, delta2, J0, J1, J2, m0, m1, m2, r, g')
        l1, l2, s1, s2, delta0, delta1, delta2, J0, J1, J2, m0, m1, m2, r, g = params

        QQ = sp.symbols("Q0, Q1, Q2")
        Q0, Q1, Q2 = QQ

        mu = st.time_deriv(theta, theta)
        p1d, q1d, q2d = mu

        # Geometry
        # noinspection PyUnusedLocal
        ex = Matrix([1, 0])
        ey = Matrix([0, 1])

        M0 = Matrix([-r*p1, r])

        S1 = M0 + mt.Rz(p1+q1)*ey*s1
        S2 = M0 + mt.Rz(p1+q1)*ey*l1+mt.Rz(p1+q1+q2)*ey*s2

        M0d = st.time_deriv(M0, theta)
        S1d = st.time_deriv(S1, theta)
        S2d = st.time_deriv(S2, theta)

        # Energy
        T_rot = ( J0*p1d**2 + J1*(p1d+q1d)**2 + J2*(p1d+ q1d+q2d)**2 )/2
        T_trans = ( m0*M0d.T*M0d  +  m1*S1d.T*S1d  +  m2*S2d.T*S2d )/2

        T = T_rot + T_trans[0]

        V = m1*g*S1[1] + m2*g*S2[1]

        mod = mt.generate_symbolic_model(T, V, theta, [0, Q1, Q2])
        self.assertEqual(mod.fz, None)
        self.assertEqual(mod.gz, None)

        mod.calc_lbi_nf_state_eq()

        self.assertEqual(mod.fz.shape, (6, 1))
        self.assertEqual(mod.gz, sp.eye(6)[:, 2:4])

    def test_triple_pendulum(self):

        np = 1
        nq = 2
        pp = sp.Matrix( sp.symbols("p1:{0}".format(np+1) ) )
        qq = sp.Matrix( sp.symbols("q1:{0}".format(nq+1) ) )
        ttheta = st.row_stack(pp, qq)
        Q1, Q2 = sp.symbols('Q1, Q2')

        p1_d, q1_d, q2_d = st.time_deriv(ttheta, ttheta)
        p1_dd, q1_dd, q2_dd = st.time_deriv(ttheta, ttheta, order=2)

        p1, q1, q2 = ttheta

        # reordering according to chain
        kk = sp.Matrix([q1, q2, p1])
        kd1, kd2, kd3 = q1_d, q2_d, p1_d

        params = sp.symbols('l1, l2, l3, l4, s1, s2, s3, s4, J1, J2, J3, J4, m1, m2, m3, m4, g')
        l1, l2, l3, l4, s1, s2, s3, s4, J1, J2, J3, J4, m1, m2, m3, m4, g = params

        # geometry
        mey = -Matrix([0,1])

        # coordinates for centers of inertia and joints
        S1 = mt.Rz(kk[0])*mey*s1
        G1 = mt.Rz(kk[0])*mey*l1

        S2 = G1 + mt.Rz(sum(kk[:2]))*mey*s2
        G2 = G1 + mt.Rz(sum(kk[:2]))*mey*l2

        S3 = G2 + mt.Rz(sum(kk[:3]))*mey*s3
        # noinspection PyUnusedLocal
        G3 = G2 + mt.Rz(sum(kk[:3]))*mey*l3

        # velocities of joints and center of inertia
        Sd1 = st.time_deriv(S1, ttheta)
        Sd2 = st.time_deriv(S2, ttheta)
        Sd3 = st.time_deriv(S3, ttheta)

        # energy
        T_rot = ( J1*kd1**2 + J2*(kd1 + kd2)**2 + J3*(kd1 + kd2 + kd3)**2)/2
        T_trans = ( m1*Sd1.T*Sd1 + m2*Sd2.T*Sd2 + m3*Sd3.T*Sd3)/2

        T = T_rot + T_trans[0]
        V = m1*g*S1[1] + m2*g*S2[1] + m3*g*S3[1]

        external_forces = [0, Q1, Q2]
        mod = mt.generate_symbolic_model(T, V, ttheta, external_forces, simplify=False)

        eqns_ref = Matrix([[J3*(2*p1_dd + 2*q1_dd + 2*q2_dd)/2 + g*m3*s3*sin(p1 + q1 + q2) + m3*s3*(-l1*q1_d**2*sin(q1) + l1*q1_dd*cos(q1) - l2*(q1_d + q2_d)**2*sin(q1 + q2) + l2*(q1_dd + q2_dd)*cos(q1 + q2) - s3*(p1_d + q1_d + q2_d)**2*sin(p1 + q1 + q2) + s3*(p1_dd + q1_dd + q2_dd)*cos(p1 + q1 + q2))*cos(p1 + q1 + q2) + m3*s3*(l1*q1_d**2*cos(q1) + l1*q1_dd*sin(q1) + l2*(q1_d + q2_d)**2*cos(q1 + q2) + l2*(q1_dd + q2_dd)*sin(q1 + q2) + s3*(p1_d + q1_d + q2_d)**2*cos(p1 + q1 + q2) + s3*(p1_dd + q1_dd + q2_dd)*sin(p1 + q1 + q2))*sin(p1 + q1 + q2)], [J1*q1_dd + J2*(2*q1_dd + 2*q2_dd)/2 + J3*(2*p1_dd + 2*q1_dd + 2*q2_dd)/2 - Q1 + g*m1*s1*sin(q1) + \
        g*m2*(l1*sin(q1) + s2*sin(q1 + q2)) + g*m3*(l1*sin(q1) + l2*sin(q1 + q2) + s3*sin(p1 + q1 + q2)) + m1*q1_dd*s1**2*sin(q1)**2 + m1*q1_dd*s1**2*cos(q1)**2 + m2*(2*l1*sin(q1) + 2*s2*sin(q1 + q2))*(l1*q1_d**2*cos(q1) + l1*q1_dd*sin(q1) + s2*(q1_d + q2_d)**2*cos(q1 + q2) + s2*(q1_dd + q2_dd)*sin(q1 + q2))/2 + m2*(2*l1*cos(q1) + 2*s2*cos(q1 + q2))*(-l1*q1_d**2*sin(q1) + l1*q1_dd*cos(q1) - s2*(q1_d + q2_d)**2*sin(q1 + q2) + s2*(q1_dd + q2_dd)*cos(q1 + q2))/2 + m3*(2*l1*sin(q1) + 2*l2*sin(q1 + q2) + 2*s3*sin(p1 + q1 + q2))*(l1*q1_d**2*cos(q1) + l1*q1_dd*sin(q1) + l2*(q1_d + q2_d)**2*cos(q1 + q2) + l2*(q1_dd + q2_dd)*sin(q1 + q2) + s3*(p1_d + q1_d + q2_d)**2*cos(p1 + q1 + q2) + s3*(p1_dd + q1_dd + q2_dd)*sin(p1 + q1 + q2))/2 + m3*(2*l1*cos(q1) + 2*l2*cos(q1 + q2) + 2*s3*cos(p1 + q1 + q2))*(-l1*q1_d**2*sin(q1) + l1*q1_dd*cos(q1) - l2*(q1_d + q2_d)**2*sin(q1 + q2) + l2*(q1_dd + q2_dd)*cos(q1 + q2) - \
        s3*(p1_d + q1_d + q2_d)**2*sin(p1 + q1 + q2) + s3*(p1_dd + q1_dd + q2_dd)*cos(p1 + q1 + q2))/2], [J2*(2*q1_dd + 2*q2_dd)/2 + J3*(2*p1_dd + 2*q1_dd + 2*q2_dd)/2 - Q2 + g*m2*s2*sin(q1 + q2) + g*m3*(l2*sin(q1 + q2) + s3*sin(p1 + q1 + q2)) + m2*s2*(-l1*q1_d**2*sin(q1) + l1*q1_dd*cos(q1) - s2*(q1_d + q2_d)**2*sin(q1 + q2) + s2*(q1_dd + q2_dd)*cos(q1 + q2))*cos(q1 + q2) + \
        m2*s2*(l1*q1_d**2*cos(q1) + l1*q1_dd*sin(q1) + s2*(q1_d + q2_d)**2*cos(q1 + q2) + s2*(q1_dd + q2_dd)*sin(q1 + q2))*sin(q1 + q2) + m3*(2*l2*sin(q1 + q2) + 2*s3*sin(p1 + q1 + q2))*(l1*q1_d**2*cos(q1) + l1*q1_dd*sin(q1) + l2*(q1_d + q2_d)**2*cos(q1 + q2) + l2*(q1_dd + q2_dd)*sin(q1 + q2) + s3*(p1_d + q1_d + q2_d)**2*cos(p1 + q1 + q2) + s3*(p1_dd + q1_dd + q2_dd)*sin(p1 + q1 + q2))/2 + m3*(2*l2*cos(q1 + q2) + 2*s3*cos(p1 + q1 + q2))*(-l1*q1_d**2*sin(q1) + l1*q1_dd*cos(q1) - l2*(q1_d + q2_d)**2*sin(q1 + q2) + l2*(q1_dd + q2_dd)*cos(q1 + q2) - s3*(p1_d + q1_d + q2_d)**2*sin(p1 + q1 + q2) + s3*(p1_dd + q1_dd + q2_dd)*cos(p1 + q1 + q2))/2]])

        self.assertEqual(eqns_ref, mod.eqns)


# noinspection PyPep8Naming
class ModelToolsTest2(unittest.TestCase):

    def setUp(self):
        pass

    def testRz(self):
        Rz1 = mt.Rz(sp.pi*0.4)
        Rz2 = mt.Rz(npy.pi*0.4, to_numpy=True)

        self.assertTrue(npy.allclose(st.to_np(Rz1), Rz2))

    def test_transform_2nd_to_1st(self):

        def intern_test(np, nq):
            N = np + nq
            xx = st.symb_vector('x1:{0}'.format(N*2+1))

            P0 = st.symbMatrix(np, N, s='A')
            P1 = st.symbMatrix(np, N, s='B')
            P2 = st.symbMatrix(np, N, s='C')

            P0bar, P1bar = mt.transform_2nd_to_1st_order_matrices(P0, P1, P2, xx)

            self.assertEqual(P0bar[:N, N:], sp.eye(N))
            self.assertEqual(P1bar[:N, :N], -sp.eye(N))

            self.assertEqual(P0bar[N:, :N], P0)
            self.assertEqual(P1bar[N:, :], P1.row_join(P2))

        intern_test(1, 3)
        intern_test(2, 2)
        intern_test(3, 1)


def main():
    # remove command line args which should not be passed to the testframework
    if 'all' in sys.argv:
        sys.argv.remove('all')
    
    unittest.main()


if __name__ == '__main__':
    main()
