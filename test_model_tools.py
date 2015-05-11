# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 11:25:00 2014

@author: Carsten Knoll
"""

import unittest
import sympy as sp
from sympy import sin, cos, Matrix
import symb_tools as st
import model_tools as mt


from IPython import embed as IPS

"""
tmp-TODO: 2 weitere Tests für symbolic Model
cart-pole und 3-Fach-Pendel.

Wenn das mit bisherigem Ergebnis übereinstimmt -> merge -> master
"""


class ModelToolsTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_simple1(self):
        q1, = qq  = sp.Matrix(sp.symbols('q1,'))
        F1, = FF  = sp.Matrix(sp.symbols('F1,'))

        m = sp.Symbol('m')

        q1d = st.perform_time_derivative(q1, qq)
        q1dd = st.perform_time_derivative(q1, qq, order=2)

        T = q1d**2*m/2
        V = 0

        mod = mt.generate_symbolic_model(T, V, qq, FF)

        eq = m*q1dd - F1

        self.assertEqual(mod.eq_list[0], eq)

        # test the application of the @property
        M = mod.calc_mass_matrix
        self.assertEqual(M[0], m)

    def test_cart_pole(self):
        p1, q1 = ttheta = sp.Matrix(sp.symbols('p1, q1'))
        F1, = FF = sp.Matrix(sp.symbols('F1,'))

        params = sp.symbols('m0, m1, l1, g')
        m0, m1, l1, g = params

        pdot1 = st.perform_time_derivative(p1, ttheta)
        # q1dd = st.perform_time_derivative(q1, ttheta, order=2)

        ex = Matrix([1,0])
        ey = Matrix([0,1])

        S0 = ex*q1  # joint of the pendulum
        S1 = S0 - mt.Rz(p1)*ey*l1  # center of mass

        #velocity
        S0d = st.perform_time_derivative(S0, ttheta)
        S1d = st.perform_time_derivative(S1, ttheta)

        T_rot = 0  # no moment of inertia (mathematical pendulum)
        T_trans = ( m0*S0d.T*S0d + m1*S1d.T*S1d )/2
        T = T_rot + T_trans[0]

        V = m1*g*S1[1]

        mod = mt.generate_symbolic_model(T, V, ttheta, [0, F1])
        mod.eq_list.simplify()

        M = mod.MM
        M.simplify()

        M_ref = Matrix([[l1**2*m1, l1*m1*cos(p1)], [l1*m1*cos(p1), m1 + m0]])
        self.assertEqual(M, M_ref)

        rest = mod.eq_list.subs(st.zip0((mod.ttdd)))
        rest_ref = Matrix([[g*l1*m1*sin(p1)], [-F1 - l1*m1*pdot1**2*sin(p1)]])

        self.assertEqual(M, M_ref)

    def test_triple_pendulum(self):

        np = 1
        nq = 2
        pp = sp.Matrix( sp.symbols("p1:{0}".format(np+1) ) )
        qq = sp.Matrix( sp.symbols("q1:{0}".format(nq+1) ) )
        ttheta = st.row_stack(pp, qq)
        Q1, Q2 = sp.symbols('Q1, Q2')


        p1_d, q1_d, q2_d = mu = st.perform_time_derivative(ttheta, ttheta)
        p1_dd, q1_dd, q2_dd = mu_d = st.perform_time_derivative(ttheta, ttheta, order=2)

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
        G3 = G2 + mt.Rz(sum(kk[:3]))*mey*l3

        # velocities of joints and center of inertia
        Sd1 = st.perform_time_derivative(S1, ttheta)
        Sd2 = st.perform_time_derivative(S2, ttheta)
        Sd3 = st.perform_time_derivative(S3, ttheta)

        # energy
        T_rot = ( J1*kd1**2 + J2*(kd1 + kd2)**2 + J3*(kd1 + kd2 + kd3)**2)/2
        T_trans = ( m1*Sd1.T*Sd1 + m2*Sd2.T*Sd2 + m3*Sd3.T*Sd3)/2

        T = T_rot + T_trans[0]
        V = m1*g*S1[1] + m2*g*S2[1] + m3*g*S3[1]

        external_forces = [0, Q1, Q2]
        mod = mt.generate_symbolic_model(T, V, ttheta, external_forces)

        eq_list_ref = Matrix([[J3*(2*p1_dd + 2*q1_dd + 2*q2_dd)/2 + g*m3*s3*sin(p1 + q1 + q2) + m3*s3*(-l1*q1_d**2*sin(q1) + l1*q1_dd*cos(q1) - l2*(q1_d + q2_d)**2*sin(q1 + q2) + l2*(q1_dd + q2_dd)*cos(q1 + q2) - s3*(p1_d + q1_d + q2_d)**2*sin(p1 + q1 + q2) + s3*(p1_dd + q1_dd + q2_dd)*cos(p1 + q1 + q2))*cos(p1 + q1 + q2) + m3*s3*(l1*q1_d**2*cos(q1) + l1*q1_dd*sin(q1) + l2*(q1_d + q2_d)**2*cos(q1 + q2) + l2*(q1_dd + q2_dd)*sin(q1 + q2) + s3*(p1_d + q1_d + q2_d)**2*cos(p1 + q1 + q2) + s3*(p1_dd + q1_dd + q2_dd)*sin(p1 + q1 + q2))*sin(p1 + q1 + q2)], [J1*q1_dd + J2*(2*q1_dd + 2*q2_dd)/2 + J3*(2*p1_dd + 2*q1_dd + 2*q2_dd)/2 - Q1 + g*m1*s1*sin(q1) + g*m2*(l1*sin(q1) + s2*sin(q1 + q2)) + g*m3*(l1*sin(q1) + l2*sin(q1 + q2) + s3*sin(p1 + q1 + q2)) + m1*q1_dd*s1**2*sin(q1)**2 + m1*q1_dd*s1**2*cos(q1)**2 + m2*(2*l1*sin(q1) + 2*s2*sin(q1 + q2))*(l1*q1_d**2*cos(q1) + l1*q1_dd*sin(q1) + s2*(q1_d + q2_d)**2*cos(q1 + q2) + s2*(q1_dd + q2_dd)*sin(q1 + q2))/2 + m2*(2*l1*cos(q1) + 2*s2*cos(q1 + q2))*(-l1*q1_d**2*sin(q1) + l1*q1_dd*cos(q1) - s2*(q1_d + q2_d)**2*sin(q1 + q2) + s2*(q1_dd + q2_dd)*cos(q1 + q2))/2 + m3*(2*l1*sin(q1) + 2*l2*sin(q1 + q2) + 2*s3*sin(p1 + q1 + q2))*(l1*q1_d**2*cos(q1) + l1*q1_dd*sin(q1) + l2*(q1_d + q2_d)**2*cos(q1 + q2) + l2*(q1_dd + q2_dd)*sin(q1 + q2) + s3*(p1_d + q1_d + q2_d)**2*cos(p1 + q1 + q2) + s3*(p1_dd + q1_dd + q2_dd)*sin(p1 + q1 + q2))/2 + m3*(2*l1*cos(q1) + 2*l2*cos(q1 + q2) + 2*s3*cos(p1 + q1 + q2))*(-l1*q1_d**2*sin(q1) + l1*q1_dd*cos(q1) - l2*(q1_d + q2_d)**2*sin(q1 + q2) + l2*(q1_dd + q2_dd)*cos(q1 + q2) - s3*(p1_d + q1_d + q2_d)**2*sin(p1 + q1 + q2) + s3*(p1_dd + q1_dd + q2_dd)*cos(p1 + q1 + q2))/2], [J2*(2*q1_dd + 2*q2_dd)/2 + J3*(2*p1_dd + 2*q1_dd + 2*q2_dd)/2 - Q2 + g*m2*s2*sin(q1 + q2) + g*m3*(l2*sin(q1 + q2) + s3*sin(p1 + q1 + q2)) + m2*s2*(-l1*q1_d**2*sin(q1) + l1*q1_dd*cos(q1) - s2*(q1_d + q2_d)**2*sin(q1 + q2) + s2*(q1_dd + q2_dd)*cos(q1 + q2))*cos(q1 + q2) + m2*s2*(l1*q1_d**2*cos(q1) + l1*q1_dd*sin(q1) + s2*(q1_d + q2_d)**2*cos(q1 + q2) + s2*(q1_dd + q2_dd)*sin(q1 + q2))*sin(q1 + q2) + m3*(2*l2*sin(q1 + q2) + 2*s3*sin(p1 + q1 + q2))*(l1*q1_d**2*cos(q1) + l1*q1_dd*sin(q1) + l2*(q1_d + q2_d)**2*cos(q1 + q2) + l2*(q1_dd + q2_dd)*sin(q1 + q2) + s3*(p1_d + q1_d + q2_d)**2*cos(p1 + q1 + q2) + s3*(p1_dd + q1_dd + q2_dd)*sin(p1 + q1 + q2))/2 + m3*(2*l2*cos(q1 + q2) + 2*s3*cos(p1 + q1 + q2))*(-l1*q1_d**2*sin(q1) + l1*q1_dd*cos(q1) - l2*(q1_d + q2_d)**2*sin(q1 + q2) + l2*(q1_dd + q2_dd)*cos(q1 + q2) - s3*(p1_d + q1_d + q2_d)**2*sin(p1 + q1 + q2) + s3*(p1_dd + q1_dd + q2_dd)*cos(p1 + q1 + q2))/2]])

        IPS()

        self.assertEqual(eq_list_ref, mod.eq_list)



def main():
    unittest.main()

if __name__ == '__main__':
    main()