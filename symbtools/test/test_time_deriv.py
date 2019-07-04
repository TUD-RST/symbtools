# -*- coding: utf-8 -*-
"""
Created on 2019-06-01 17:28:34 (copy from test_core1)

@author: Carsten Knoll
"""

import sympy as sp
from sympy import sin, cos, exp

import symbtools as st
import ipydex as ipd


import unittest


# noinspection PyShadowingNames,PyPep8Naming,PySetFunctionToLiteral
class Tests1(unittest.TestCase):

    def setUp(self):
        st.init_attribute_store(reinit=True)

    def test_time_deriv1(self):
        a, b, t = sp.symbols("a, b, t")
        f1 = a ** 2 + 10 * sin(b)
        f1_dot = st.time_deriv(f1, (a, b))
        self.assertEqual(str(f1_dot), "2*a*adot + 10*bdot*cos(b)")

        f2 = sin(a)
        f2_dot2 = st.time_deriv(f2, [a, b], order=2)
        f2_dot = st.time_deriv(f2, [a])
        f2_ddot = st.time_deriv(f2_dot, [a])
        self.assertEqual(f2_ddot, f2_dot2)

    def test_time_deriv2(self):
        """
        test matrix compatibility
        """

        a, b, t = sp.symbols("a, b, t")

        # not time dependent
        x, y = sp.symbols("x, y")

        adot, bdot = st.time_deriv(sp.Matrix([a, b]), (a, b))

        # has no associated function -> return the symbol itself
        self.assertEqual(x.ddt_func, x)

        # due to time_deriv now the .ddt_func attribute is set for a, b, adot, bdot
        self.assertTrue(isinstance(adot.ddt_func, sp.Derivative))
        self.assertEqual(type(type(a.ddt_func)), sp.function.UndefinedFunction)
        self.assertEqual(type(a.ddt_func).__name__, a.name)

        self.assertEqual(a.ddt_child, adot)
        self.assertEqual(bdot.ddt_parent, b)

        addot1 = st.time_deriv(adot, [a, b])
        addot2 = st.time_deriv(a, [a, b], order=2)
        self.assertEqual(addot1, addot2)

        A = sp.Matrix([sin(a), exp(a * b), -t ** 2 * 7 * 0, x + y]).reshape(2, 2)
        A_dot = st.time_deriv(A, (a, b))

        A_dot_manual = \
            sp.Matrix([adot * cos(a), b * adot * exp(a * b) + a * bdot * exp(a * b),
                       -t ** 2 * 7 * 0, 0]).reshape(2, 2)

        self.assertEqual(A_dot.expand(), A_dot_manual)

    def test_perform_time_deriv3(self):
        """
        test to provide deriv_symbols
        """

        a, b, adot, bdot, addot, bddot = sp.symbols("a, b, adot, bdot, addot, bddot")

        f1 = 8 * a * b ** 2
        f1d = st.time_deriv(f1, (a, b), (adot, bdot))
        self.assertEqual(f1d, 8 * adot * b ** 2 + 16 * a * bdot * b)

        f1d_2 = st.time_deriv(f1, (a, b), (adot, bdot) + (addot, bddot),
                              order=2)

        f1d_2_altntv = st.time_deriv(f1d, (a, b, adot, bdot),
                                     (adot, bdot) + (addot, bddot))
        self.assertEqual(sp.expand(f1d_2 - f1d_2_altntv), 0)

    def test_perform_time_deriv4(self):
        # test higher order derivatives

        a, b = sp.symbols("a, b")

        f1 = 8 * a * b ** 2

        res_a1 = st.time_deriv(f1, (a, b), order=5)

        a_str = 'a adot addot adddot addddot a_d5'
        b_str = a_str.replace('a', 'b')

        expected_symbol_names = a_str.split() + b_str.split()

        res_list = [res_a1.has(sp.Symbol(e)) for e in expected_symbol_names]

        self.assertTrue(all(res_list))

        l1 = len(res_a1.atoms(sp.Symbol))
        self.assertEqual(len(expected_symbol_names), l1)

    def test_perform_time_deriv5(self):
        # test numbered symbols

        x1, x2 = xx = sp.symbols("x1, x_2")

        res_a1 = st.time_deriv(x1, xx)
        self.assertEqual(str(res_a1), 'xdot1')

        res_a2 = st.time_deriv(x1, xx, order=2)
        self.assertEqual(str(res_a2), 'xddot1')

        res_a3 = st.time_deriv(x1, xx, order=3)
        self.assertEqual(str(res_a3), 'xdddot1')

        res_a4 = st.time_deriv(x1, xx, order=4)
        self.assertEqual(str(res_a4), 'xddddot1')

        # FIXME:
        res_a5 = st.time_deriv(x1, xx, order=5)
        # self.assertEqual(str(res_a5), 'x1_d5')

        # test attributes ddt_child and ddt_parent
        tmp = x1
        for i in range(res_a5.difforder):
            tmp = tmp.ddt_child
        self.assertEqual(tmp, res_a5)

        tmp = res_a5
        for i in range(res_a5.difforder):
            tmp = tmp.ddt_parent
        self.assertEqual(tmp, x1)

        self.assertEqual(st.time_deriv(x1.ddt_child, xx, order=2),
                         x1.ddt_child.ddt_child.ddt_child)

        res_b1 = st.time_deriv(x2, xx)
        self.assertEqual(str(res_b1), 'x_dot2')

        res_b2 = st.time_deriv(x2, xx, order=2)
        self.assertEqual(str(res_b2), 'x_ddot2')

        res_b3 = st.time_deriv(x2, xx, order=3)
        self.assertEqual(str(res_b3), 'x_dddot2')

        res_b4 = st.time_deriv(x2, xx, order=4)
        self.assertEqual(str(res_b4), 'x_ddddot2')

        # FIXME
        res_b5 = st.time_deriv(x2, xx, order=5)
        # self.assertEqual(str(res_b5), 'x_2_d5')

    @unittest.expectedFailure
    def test_perform_time_deriv5f(self):
        # test numbered symbols

        x1, x2 = xx = sp.symbols("x1, x_2")

        # TODO: These two assertions should pass
        # Then the above FIXME-issues can be resolved and this test is obsolete

        res_a5 = st.time_deriv(x1, xx, order=5)
        self.assertEqual(str(res_a5), 'x1_d5')

        res_b5 = st.time_deriv(x2, xx, order=5)
        self.assertEqual(str(res_b5), 'x_2_d5')

    def test_time_deriv7(self):
        a, b, t = sp.symbols("a, b, t", commutative=False)

        f1 = sp.Function('f1')(t)
        f2 = sp.Function('f2')(t)

        res1 = st.time_deriv(f1, [a])
        self.assertEqual(res1, f1.diff(t))

        res2 = st.time_deriv(f1, [])
        self.assertEqual(res2, f1.diff(t))

        res3 = st.time_deriv(a * f1, [a, b])

        # we must provide the noncommutative t_symbols here (because a.ddt_function depends on it)
        adot = st.time_deriv(a, [a], t_symbol=t)
        self.assertEqual(res3, a * f1.diff(t) + adot * f1)

    def test_time_deriv8(self):

        y1, y2 = yy = sp.Matrix(sp.symbols('y1, y2', commutative=False))

        ydot1 = st.time_deriv(y1, yy)
        ydot2 = st.time_deriv(y2, yy)

        yddot1 = st.time_deriv(y1, yy, order=2)
        ydddot1 = st.time_deriv(y1, yy, order=3)

        res1 = st.time_deriv(ydot1, yy)
        self.assertEqual(res1, yddot1)

        res2 = st.time_deriv(ydot1, yy, order=2)
        self.assertEqual(res2, ydddot1)

        res3 = st.time_deriv(yddot1, yy)
        self.assertEqual(res3, ydddot1)

    def test_time_deriv_matrix_symbol1(self):
        A = sp.MatrixSymbol("A", 3, 3)
        B = sp.MatrixSymbol("B", 3, 5)
        C = sp.MatrixSymbol("C", 5, 5)

        Adot = st.time_deriv(A, [A])
        Cdot = st.time_deriv(C, [C])

        self.assertEqual(Adot.shape, A.shape)
        self.assertEqual(Adot.name, "Adot")

        res1 = st.time_deriv(3 * A, [A])
        self.assertEqual(res1, 3 * Adot)

        res2 = st.time_deriv(A * B * C, [A])
        self.assertEqual(res2, Adot * B * C)

        res3 = st.time_deriv(A * B * C + B * C, [A, C])
        self.assertEqual(res3, B * Cdot + A * B * Cdot + Adot * B * C)

    def test_time_deriv_matrix_symbol2(self):
        A = sp.MatrixSymbol("A", 3, 3)
        B = sp.MatrixSymbol("B", 3, 5)
        C = sp.MatrixSymbol("C", 5, 5)

        # test higher order

        Adot = st.time_deriv(A, [A])
        Cdot = st.time_deriv(C, [C])

        Addot = st.time_deriv(A, [A], order=2)
        Cddot = st.time_deriv(C, [C], order=2)

        Adddot = st.time_deriv(A, [A], order=3)
        Cdddot = st.time_deriv(C, [C], order=3)

        Addddot = st.time_deriv(A, [A], order=4)
        Cddddot = st.time_deriv(C, [C], order=4)

        self.assertEqual(Addddot.shape, A.shape)
        self.assertEqual(Addddot.name, "Addddot")

        res1 = st.time_deriv(3 * A, [A], order=3)
        self.assertEqual(res1, 3 * Adddot)

        res2 = st.time_deriv(A * B * C, [A], order=2)
        self.assertEqual(res2, Addot * B * C)

        res3 = st.time_deriv(A * B * C + B * C, [A, C], order=4)
        eres3 = B * Cddddot + A * B * Cddddot + Addddot * B * C + 4 * Adddot * B * Cdot + \
                4 * Adot * B * Cdddot + 6 * Addot * B * Cddot
        self.assertEqual(res3, eres3)

        res4 = st.time_deriv(Adot, [A]) - Addot
        self.assertEqual(res4, 0 * A)

        res5 = st.time_deriv(Adot, [A], order=3)
        self.assertEqual(res5, Addddot)

        res6 = st.time_deriv(A * B, [A], order=0)
        self.assertEqual(res6, A * B)

    def test_get_all_deriv_childs_and_parents(self):

        x1, x2 = xx = st.symb_vector("x1, x2")
        xdot1, xot2 = xxd = st.time_deriv(xx, xx)
        xddot1, xdot2 = xxdd = st.time_deriv(xx, xx, order=2)

        expr = x1*x2

        E2 = st.time_deriv(expr, xx, order=2)

        dc = st.get_all_deriv_childs(xx)

        self.assertEqual(len(dc), 4)

        xdot1, xdot2 = st.time_deriv(xx, xx)
        xddot1, xddot2 = st.time_deriv(xx, xx, order=2)

        self.assertTrue( xdot1 in dc)
        self.assertTrue( xddot1 in dc)
        self.assertTrue( xdot2 in dc)
        self.assertTrue( xddot2 in dc)

        dp1 = st.get_all_deriv_parents(xdot1)
        dp2 = st.get_all_deriv_parents(xddot2)

        self.assertEqual(dp1, sp.Matrix([x1]))
        self.assertEqual(dp2, sp.Matrix([xdot2, x2]))
