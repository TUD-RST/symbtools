# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 11:35:00 2014

@author: Carsten Knoll
"""

import unittest

import sympy as sp
from sympy import sin, cos, exp

import symb_tools as st
from IPython import embed as IPS


class SymbToolsTest(unittest.TestCase):

    def setUp(self):
        pass


    def test_symbs_to_func(self):
        a, b, t = sp.symbols("a, b, t")
        x = a + b
        #M = sp.Matrix([x, t, a**2])

        f_x = st.symbs_to_func(x, [a, b], t)
        self.assertEquals(str(f_x), "a(t) + b(t)")

    def test_perform_time_deriv1(self):
        a, b, t = sp.symbols("a, b, t")
        f1 = a**2 + 10*sin(b)
        f1_dot = st.perform_time_derivative(f1, (a, b))
        self.assertEquals(str(f1_dot), "2*a*a_d + 10*b_d*cos(b)")

        f2 = sin(a)
        f2_dot2 = st.perform_time_derivative(f2, [a, b], order=2)
        f2_dot = st.perform_time_derivative(f2, [a])
        f2_ddot = st.perform_time_derivative(f2_dot, [a, sp.Symbol('a_d')])
        self.assertEqual(f2_ddot, f2_dot2)

    def test_perform_time_deriv2(self):
        """
        test matrix compatibility
        """

        a, b, t = sp.symbols("a, b, t")
        x, y = sp.symbols("x, y")

        ad, bd = st.perform_time_derivative( sp.Matrix([a, b]), (a, b) )

        A = sp.Matrix([sin(a), exp(a*b), -t**2*7*0, x + y]).reshape(2, 2)
        A_dot = st.perform_time_derivative(A, (a, b))

        A_dot_manual = \
            sp.Matrix([ad*cos(a), b*ad*exp(a*b) + a*bd*exp(a*b),
                       -t**2*7*0, 0]).reshape(2, 2)

        self.assertEqual(A_dot.expand(), A_dot_manual)

    def test_perform_time_deriv3(self):
        """
        test to provide deriv_symbols
        """

        a, b, ad, bd, add, bdd = sp.symbols("a, b, ad, bd, add, bdd")

        f1 = 8*a*b**2
        f1d = st.perform_time_derivative(f1, (a, b), (ad, bd))
        self.assertEquals(f1d, 8*ad*b**2 + 16*a*bd*b)

        f1d_2 = st.perform_time_derivative(f1, (a, b), (ad, bd)+ (add, bdd),
                                           order=2)

        f1d_2_altntv = st.perform_time_derivative(f1d, (a, b, ad, bd),
                                                 (ad, bd)+ (add, bdd) )
        self.assertEquals(f1d_2, f1d_2_altntv)

    def test_match_symbols_by_name(self):
        a, b, c = abc0 = sp.symbols('a, b, c', real=True)
        a1, b1, c1 = abc1 = sp.symbols('a, b, c')

        self.assertFalse(a == a1 or b == b1 or c == c1)

        abc2 = st.match_symbols_by_name(abc0, abc1)
        self.assertEquals(abc0, tuple(abc2))

        input3 = [a1, b, "c", "x"]
        a3, b3, c3, x3 = st.match_symbols_by_name(abc0, input3)

        self.assertEquals(abc0, (a3, b3, c3))
        self.assertEquals(x3, sp.Symbol('x'))

    def test_symbs_to_func(self):
        a, b, t = sp.symbols("a, b, t")
        x, y = sp.symbols("x, y")

        at = sp.Function('a')(t)
        bt = sp.Function('b')(t)

        xt = sp.Function('x')(t)
        yt = sp.Function('y')(t)

        M = sp.Matrix([x, y])
        Mt = sp.Matrix([xt, yt])

        expr1 = a**2 + b**2
        f1 = st.symbs_to_func(expr1, (a,b), t)
        self.assertEqual(f1, at**2 + bt**2)

        at2 = st.symbs_to_func(a, arg=t)
        self.assertEqual(at, at2)

        # Matrix
        Mt2 = st.symbs_to_func(M, arg=t)
        self.assertEqual(Mt, Mt2)

        # TODO: assert raises

    def test_poly_expr_coeffs(self):
        a, b, c, d = sp.symbols("a, b, c, d")
        x, y = sp.symbols("x, y")

        p1 = a*x + b*x**2 + c*y - d + 2*a*x*y
        sample_solution = [((0, 0), -d), ((1, 0), a), ((2, 0), b),
                           ((0, 1), c), ((1, 1), 2*a), ((0, 2), 0)]

        # coeff-dict
        cd1 = st.poly_expr_coeffs(p1, [x,y])
        self.assertEqual(cd1, dict(sample_solution))

    def test_get_diffterms(self):
        x1, x2, x3, = xx = sp.symbols('x1:4')
        res2 = st.get_diffterms(xx, 2)
        expected_res2 = [(x1, x1), (x1, x2), (x1, x3), (x2, x2),
                         (x2, x3), (x3, x3)]

        expected_indices2 = [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0),
                             (0, 1, 1), (0, 0, 2)]
        self.assertEqual(res2, expected_res2)

        res3 = st.get_diffterms(xx, 3)
        res23 = st.get_diffterms(xx, (2, 3))

        self.assertEqual(res23, res2 + res3)

        res2b, indices2 = st.get_diffterms(xx, 2, order_list=True)
        self.assertEqual(res2b, res2)
        self.assertEqual(expected_indices2, indices2)

        res3b, indices3 = st.get_diffterms(xx, 3, order_list=True)
        res23b, indices23 = st.get_diffterms(xx, (2, 3), order_list=True)

        self.assertEqual(res23b, res2b + res3b)
        self.assertEqual(indices23, indices2 + indices3)

    def test_get_diffterms_bug1(self):
        xx = sp.symbols('x, y')
        res, orders = st.get_diffterms(xx, 2, order_list=True)
        self.assertEqual(orders, [(2, 0), (1, 1), (0, 2)])

    def test_monomial_from_signature(self):
        x1, x2, x3, = xx = sp.symbols('x1:4')

        s1 = (0, 0, 0)
        p1 = st.monomial_from_signature(s1, xx)
        self.assertEqual(p1, 1)

        s2 = (3, 2, 1)
        p2 = st.monomial_from_signature(s2, xx)
        self.assertEqual(p2, x1**3 * x2**2 * x3 )

    def test_zip0(self):
        aa = sp.symbols('a1:5')
        bb = sp.symbols('b1:4')
        xx = sp.symbols('x1:3')

        s1 = sum(aa)
        self.assertEqual(s1.subs(st.zip0(aa)), 0)
        self.assertEqual(s1.subs(st.zip0(aa, arg=3.5)), 3.5*len(aa))
        self.assertEqual(s1.subs(st.zip0(aa, arg=xx[0])), xx[0]*len(aa))

        s2 = s1 + sum(bb) + sum(xx)
        self.assertEqual(s1.subs(st.zip0(aa, bb, xx)), 0)
        t2 = s2.subs(st.zip0(aa, bb, xx, arg=-2.1))
        self.assertEqual( t2, -2.1*len(aa + bb + xx) )

    def test_is_number(self):
        x1, x2, x3 = xx = sp.symbols('x1:4')

        self.assertTrue(st.is_number(x1/x1))
        self.assertTrue(st.is_number(1))
        self.assertTrue(st.is_number(3.4))
        self.assertTrue(st.is_number(-10.0000001))

        self.assertFalse(st.is_number(x1))
        self.assertFalse(st.is_number(x1/x2))
        self.assertFalse(st.is_number(float('nan')))
        self.assertFalse(st.is_number(float('inf')))
        self.assertFalse(st.is_number(-float('inf')))

    def test_rnd_number_tuples(self):
        x1, x2, x3 = xx = sp.symbols('x1:4')

        s = sum(xx)
        res_a1 = st.rnd_number_subs_tuples(s)
        self.assertTrue(isinstance(res_a1, list))
        self.assertEqual(len(res_a1), len(xx))

        c1 = [len(e)==2 and e[0].is_Symbol and st.is_number(e[1])
              for e in res_a1]

        self.assertTrue( all(c1) )

        t = sp.Symbol('t')
        f = sp.Function('f')(t)

        fdot = f.diff(t)
        fddot = f.diff(t, 2)

        ff = sp.Matrix([f, fdot, fddot, x1*x2])

        res_b1 = st.rnd_number_subs_tuples(ff)

        expct_b1_set = set([f, fdot, fddot, t, x1, x2])
        res_b1_atom_set = set( zip(*res_b1)[0] )

        self.assertEqual(expct_b1_set, res_b1_atom_set)
        self.assertEqual(res_b1[0][0], fddot)
        self.assertEqual(res_b1[1][0], fdot)
        self.assertTrue( all( [st.is_number(e[1]) for e in res_b1] ) )

    def test_rnd_number_tuples2(self):
        x1, x2, x3 = xx = sp.symbols('x1:4')

        s = sum(xx)
        res_a1 = st.rnd_number_subs_tuples(s, seed=1)
        res_a2 = st.rnd_number_subs_tuples(s, seed=2)
        self.assertNotEqual(res_a1, res_a2)

        res_b1 = st.rnd_number_subs_tuples(s, seed=2)
        self.assertEqual(res_b1, res_a2)

    def test_lie_deriv_cartan(self):
        x1, x2, x3 = xx = sp.symbols('x1:4')
        u1, u2 = uu = sp.Matrix(sp.symbols('u1:3'))

        # ordinary lie_derivative

        # source: inspired by the script of Prof. Kugi (TU-Wien)
        f = sp.Matrix([-x1**3, cos(x1)*cos(x2), x2])
        g = sp.Matrix([cos(x2), 1, exp(x1)])
        h = x3
        Lfh = x2
        Lf2h = f[1]
        Lgh = exp(x1)

        res1 = st.lie_deriv_cartan(h, f, xx)
        res2 = st.lie_deriv_cartan(h, f, xx, order=2)

        self.assertEqual(res1, Lfh)
        self.assertEqual(res2, Lf2h)

        # incorporating the input
        h2 = u1

        udot1, udot2 = uudot = st.perform_time_derivative(uu, uu, order=1)
        uddot1, uddot2 = st.perform_time_derivative(uu, uu, order=2)

        res_a1 = st.lie_deriv_cartan(h2, f, xx, uu, order=1)
        res_a2 = st.lie_deriv_cartan(h2, f, xx, uu, order=2)

        self.assertEqual(res_a1, udot1)
        self.assertEqual(res_a2, uddot1)

        res_a3 = st.lie_deriv_cartan(udot1, f, xx, [uu, uudot], order=1)
        self.assertEqual(res_a3, uddot1)

        # more complex examples
        h3 = x3 + u1
        fg = f + g * u2

        res_b1 = st.lie_deriv_cartan(h3, fg, xx, uu, order=1)
        res_b2 = st.lie_deriv_cartan(h3, fg, xx, uu, order=2)
        res_b3 = st.lie_deriv_cartan(res_b1, fg, xx, [uu, uudot], order=1)

        self.assertEqual(res_b1, Lfh + Lgh*u2 + udot1)
        self.assertEqual(sp.expand(res_b2 - res_b3), 0)

        h4 = x3 * sin(x2)
        fg = f + g * u2

        res_c1 = st.lie_deriv_cartan(h4, fg, xx, uu, order=1)
        res_c2 = st.lie_deriv_cartan(res_c1, fg, xx, uu, order=1)
        res_c3 = st.lie_deriv_cartan(h4, fg, xx, uu, order=2)

        self.assertEqual(sp.expand(res_c2 - res_c3), 0)

    def test_deriv_2nd_order_chain_rule(self):
        a, b, x = sp.symbols('a, b, x')

        f1 = a**3 + a*b**2 + 7*a*b
        f2 = -2*a**2 + b*a*b**2/(2+a**2 * b**2) + 12*a*b
        f3 = -3*a**2 + b*a*b**2 + 7*a*b/(2+a**2 * b**2)


        f = f1
        aa = sp.cos(3*x)
        bb = sp.exp(5*x)

        ab_subs = [(a, aa), (b, bb)]
        fab = f.subs(ab_subs)
        fab_d2 = fab.diff(x, 2)

        res1 = st.deriv_2nd_order_chain_rule(f, (a,b), [aa, bb], x)

        res1a = sp.simplify(res1 - fab_d2)
        self.assertEqual(res1a, 0)

        # multiple functions

        ff = sp.Matrix([f1, f2, f3])

        ff_ab_d2 = ff.subs(ab_subs).diff(x, 2)

        res2 = st.deriv_2nd_order_chain_rule(ff, (a, b), [aa, bb], x)
        res2a = sp.expand(res2 - ff_ab_d2)
        self.assertEqual(res2a, res2a*0)

    def test_update_cse(self):
        a, b, c, x1, x2, x3 = sp.symbols('a, b, c, x1, x2, x3')

        L1 = [(x1, a + b), (x2, x1*b**2)]
        new_subs1 = [(b, 2)]
        new_subs2 = [(a, b + 5), (b, 3),]

        res1_exp = [(x1, a + 2), (x2, (a + 2)*4)]
        res2_exp = [(x1, 11), (x2, 99)]

        res1 = st.update_cse(L1, new_subs1)
        res2 = st.update_cse(L1, new_subs2)

        self.assertEqual(res1, res1_exp)
        self.assertEqual(res2, res2_exp)


def main():
    unittest.main()

if __name__ == '__main__':
    main()