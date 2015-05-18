# -*- coding: utf-8 -*-
"""
Created on Fri 2015-03-20

@author: Carsten Knoll
"""

import unittest

import sympy as sp
from sympy import sin, cos, exp

import symb_tools as st
from IPython import embed as IPS
import non_commutative_tools as nct

# Noncommutative tools
t = nct.t
s = nct.s


class NonCommToolsTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_apply_deriv1(self):
        a, b = sp.symbols("a, b")
        f1 = sp.Function('f1')(t)
        F1 = a*f1

        res1 = nct.apply_deriv(F1, 1, s, t)
        self.assertEqual(res1, F1.diff(t) + F1*s)

        res2 = nct.apply_deriv(F1, 3, s, t)
        self.assertEqual(res2, F1.diff(t, 3) + 3*F1.diff(t, 2)*s + 3*F1.diff(t)*s**2 + F1*s**3)

    def test_apply_deriv2(self):
        y1, y2 = yy = sp.Matrix( sp.symbols('y1, y2', commutative=False) )

        ydot1 = st.perform_time_derivative(y1, yy)
        ydot2 = st.perform_time_derivative(y2, yy)
        yddot1 = st.perform_time_derivative(y1, yy, order=2)
        ydddot1 = st.perform_time_derivative(y1, yy, order=3)

        res1 = nct.apply_deriv(y1, 1, s, t, func_symbols=yy)
        self.assertEqual(res1, ydot1 + y1*s)

        res3 = nct.apply_deriv(y1*s, 1, s, t, func_symbols=yy)
        self.assertEqual(res3, ydot1*s + y1*s**2)

        res4 = nct.apply_deriv(y2 + y1, 1, s, t, func_symbols=yy)
        self.assertEqual(res4, ydot1 + ydot2 + y1*s + y2*s)

        res5 = nct.apply_deriv(ydot1 + y1*s, 1, s, t, func_symbols=yy)
        self.assertEqual(res5, yddot1 + 2*ydot1*s + y1*s**2)

        res6 = nct.apply_deriv(y1, 2, s, t, func_symbols=yy)
        self.assertEqual(res5, res6)

        res2 = nct.apply_deriv(y1, 3, s, t, func_symbols=yy)
        self.assertEqual(res2, ydddot1 + 3*yddot1*s + 3*ydot1*s**2 + y1*s**3)

    def test_right_shift(self):
        a, b = sp.symbols("a, b")
        f1 = sp.Function('f1')(t)
        f1d = f1.diff(t)
        f2 = sp.Function('f2')(t)

        res1 = nct.right_shift(s*f1, s, t)
        ex1 = f1.diff(t) + f1*s

        self.assertEquals(res1, ex1)

        res2 = nct.right_shift(f2*s*f1, s, t)
        ex2= f2*f1.diff(t) + f2*f1*s

        self.assertEquals(res2, ex2)

        res3 = nct.right_shift(a*f2*s*f1d, s, t)
        ex3= a*f2*f1.diff(t, 2) + a*f2*f1d*s

        self.assertEquals(res3, ex3)

        res4 = nct.right_shift(s*f1*f2, s, t)
        ex4 = f1.diff(t)*f2 + f1*f2*s + f1*f2.diff(t)

        self.assertEquals(res4, ex4)

        self.assertRaises( ValueError, nct.right_shift, s*f1*(f2+1), s, t )

    def test_right_shift2(self):
        a, b = sp.symbols("a, b", commutative = False)
        f1 = sp.Function('f1')(t)
        f1d = f1.diff(t)
        f2 = sp.Function('f2')(t)

        res1 = nct.right_shift(s*t, s, t)
        ex1 = 1 + t*s

        self.assertEquals(res1, ex1)

        res2 = nct.right_shift(s, s, t)
        ex2 = s
        self.assertEquals(res2, ex2)

        res3 = nct.right_shift(s**4, s, t)
        ex3 = s**4
        self.assertEquals(res3, ex3)

        res4 = nct.right_shift(s**4*a*b, s, t)
        ex4 = a*b*s**4
        self.assertEquals(res4, ex4)

        res5 = nct.right_shift(s**2*a*s*b*s, s, t)
        ex5 = a*b*s**4
        self.assertEquals(res5, ex5)

        res6 = nct.right_shift(s**2*(a*t**3), s, t)
        ex6 = a*(6*t + 6*t**2*s + t**3*s**2)
        self.assertEquals(res6, ex6)

    def test_right_shift3(self):
        a, b = sp.symbols("a, b", commutative = False)
        f1 = sp.Function('f1')(t)
        f2 = sp.Function('y2')(t)
        f1d = f1.diff(t)
        f1dd = f1.diff(t, 2)
        f2d = f2.diff(t)
        f2dd = f2.diff(t, 2)

        res1 = nct.right_shift(s*f1d*f2d, s, t)
        ex1 = f1dd*f2d + f1d*f2dd + f1d*f2d*s

        self.assertEquals(res1, ex1)

        test = s*f2*f2d
        res2 = nct.right_shift(test, s, t)
        ex2 = f2d**2 + f2*f2dd + f2*f2d*s

        self.assertEquals(res2, ex2)

    def _test_right_shift4(self):

        y1, y2 = yy = sp.Matrix( sp.symbols('y1, y2', commutative=False) )

        ydot1, ydot2 = st.perform_time_derivative(yy, yy)
        res1 = nct.right_shift(s*y1, s, t, yy)

        self.assertEqual(res1, ydot1 + y1*s)

    def _test_right_shift_all(self):
        1/0
        a, b = sp.symbols("a, b")
        f1 = sp.Function('f1')(t)
        f1d = f1.diff(t)
        f2 = sp.Function('f2')(t)

    @unittest.expectedFailure
    def test_nc_sympy_multiplication_bug(self):
    # This seems to be a sympy bug
        a, b = sp.symbols("a, b", commutative = False)
        E = sp.eye(2)

        Mb = b*E
        Mab = a*b*E

        res = a*Mb - Mab

        self.assertEqual(res, 0*E)

    def test_nc_multiplication(self):
        a, b = sp.symbols("a, b", commutative = False)
        E = sp.eye(2)

        Mb = b*E
        Mab = a*b*E

        res = nct.nc_mul(a, Mb) - Mab
        self.assertEqual(res, 0*E)

        res2 = nct.nc_mul(a*E, b*E)
        self.assertEqual(res2, Mab)

        res3 = nct.nc_mul(Mb, Mab)
        self.assertEqual(res3, b*a*b*E)


def main():
    unittest.main()

if __name__ == '__main__':
    main()